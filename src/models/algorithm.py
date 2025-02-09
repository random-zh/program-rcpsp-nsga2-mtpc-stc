import math
import random
import numpy as np
from collections import defaultdict, deque
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any, Set
from .problem import Project, Activity, Program
from utils.config_loader import config
from gurobipy import Model, GRB, quicksum


class Schedule:
    """调度类：核心调度逻辑，管理活动优先级、资源分配及目标计算"""

    def __init__(self, project: Project):
        # === 核心属性 ===
        self.project = project  # 关联的项目对象
        self.priority_order: Dict[int, int] = {}  # {活动ID: 优先级值}
        self.start_times: Dict[int, int] = {}  # {活动ID: 开始时间}
        self.total_duration: Optional[int] = None  # 总工期
        self.robustness: Optional[float] = None  # 鲁棒性指标
        self.resource_usage: Dict[str, List[int]] = {}  # 资源时间轴占用 {资源名: [时间轴]}

        # === 鲁棒性计算缓存 ===
        self.ff_cache: Dict[int, int] = {}  # {活动ID: 自由时差}
        self.path_length_cache: Dict[int, int] = {}  # {活动ID: 到虚活动的路径长度}
        self.virtual_activity_id: int = self._find_virtual_activity()

        # === 初始化资源时间轴为0 ===
        max_duration_guess = sum(act.duration for act in self.project.activities.values())  # 所有活动工期之和
        for res in self.project.local_resources:
            self.resource_usage[res] = [0] * (max_duration_guess + 1)

    # === 检测优先级顺序是否违背紧后关系约束 ===
    def validate_priority_order(self) -> None:
        """
        验证优先级顺序是否满足紧后关系约束。
        规则：若活动A是活动B的紧前活动（B必须在A之后执行），则A的优先级必须高于B。
        若存在冲突，抛出 E。
        """
        violations = []
        for act in self.project.activities.values():
            current_priority = self.priority_order.get(act.activity_id, 0)
            # 检查所有后继活动的优先级是否不高于当前活动（越小优先级越高）
            for succ_id in act.successors:
                succ_act = self.project.activities.get(succ_id)
                if not succ_act:
                    raise Exception(
                        f" 活动 {act.activity_id} 的紧后活动 {succ_id} 不存在 "
                    )
                succ_priority = self.priority_order.get(succ_id, 0)
                if succ_priority <= current_priority:
                    violations.append(
                        f"活动 {act.activity_id} (优先级={current_priority}) -> "
                        f"活动 {succ_id} (优先级={succ_priority}) 违反紧后约束"
                    )
        if violations:
            raise Exception(
                "优先级顺序违背紧后关系约束：\n" + "\n".join(violations)
            )

    # === 调度生成方法（优先顺序需要满足活动紧后要求） ===
    def generate_schedule(self) -> None:
        """基于优先级顺序生成调度（串行调度生成方案，SSGS）"""

        # 1. 检查优先级顺序是否合法
        self.validate_priority_order()

        # 按优先级排序活动
        sorted_activities = sorted(
            self.project.activities.values(),
            key=lambda act: self.priority_order.get(act.activity_id, 0),
            reverse=True
        )

        for act in sorted_activities:
            # 1. 计算最早开始时间（考虑前驱完成时间）
            earliest_start = 0
            for pred_id in act.predecessors:
                pred_end = self.start_times.get(pred_id, 0) + self.project.activities[pred_id].duration
                earliest_start = max(earliest_start, pred_end)

            # 2. 找到满足资源约束的最早可行时间
            start_time = earliest_start
            while True:
                if self._is_resource_available(act, start_time):
                    break
                start_time += 1

            # 3. 更新资源占用和开始时间
            self._allocate_resources(act, start_time)
            self.start_times[act.activity_id] = start_time

        # 计算总工期和鲁棒性
        self.calculate_total_duration()
        self.calculate_robustness()

    # === 私有资源管理方法 ===
    def _is_resource_available(self, act: Activity, start_time: int) -> bool:
        """检查资源在时间段 [start_time, start_time+duration) 是否可用"""
        for res, demand in act.resource_request.items():
            # 本地资源检查
            if res in self.project.local_resources:
                for t in range(start_time, start_time + act.duration):
                    # if t >= len(self.resource_usage[res]):
                    #     return True  # 假设超出预测范围时资源足够
                    if self.resource_usage[res][t] + demand > self.project.local_resources[res]:
                        return False
        return True

    def _allocate_resources(self, act: Activity, start_time: int) -> None:
        """分配占用资源"""
        for res, demand in act.resource_request.items():
            if res in self.project.local_resources:
                for t in range(start_time, start_time + act.duration):
                    # if t >= len(self.resource_usage[res]):
                    # 动态扩展时间轴
                    # self.resource_usage[res].extend([0] * (t - len(self.resource_usage[res]) + 1))
                    self.resource_usage[res][t] += demand

    # === 目标计算方法 ===
    def calculate_total_duration(self) -> None:
        """计算总工期（最大结束时间）"""
        self.total_duration = max(
            start + act.duration
            for act_id, start in self.start_times.items()
            for act in [self.project.activities[act_id]]
        )

    def calculate_robustness(self) -> None:
        """计算基于CIW的鲁棒性指标"""
        # 预计算路径长度和自由时差
        self._precompute_robustness_data()

        # 计算总鲁棒性
        total_robustness = 0.0
        for act in self.project.activities.values():
            total_robustness += self.calculate_ciw(act)
        self.robustness = total_robustness

    def _precompute_robustness_data(self) -> None:
        """预计算路径长度和自由时差"""
        # 计算所有活动的路径长度
        for act in self.project.activities.values():
            _ = self.calculate_max_path_length(act)

        # 计算所有活动的自由时差
        for act in self.project.activities.values():
            self.ff_cache[act.activity_id] = self.calculate_free_float(act)

    # === 计算自由时差（考虑资源约束） ===
    def calculate_free_float(self, act: Activity) -> int:
        """
        计算活动的自由时差（FF）：
        - 时间约束：FF_time = min(后继活动ES) - 当前活动EF
        - 资源约束：在[EF, EF+Δ)期间资源不超限
        - FF = min(FF_time, Δ_max)
        """
        # 1. 计算时间约束允许的最大Δ
        es = self.start_times[act.activity_id]
        ef = es + act.duration

        if act.successors:
            # 获取所有紧后活动的最早开始时间的最小值
            min_succ_es = min(
                self.start_times[succ_id] for succ_id in act.successors
                if succ_id in self.start_times
            )
            ff_time = max(0, min_succ_es - ef)
        else:
            # 无后继活动，FF_time为0
            return 0

        if ff_time == 0:
            return 0

        # 2. 计算资源约束允许的最大Δ
        delta_max = 0

        # 遍历Δ的候选值（从0到ff_time）
        for delta in range(0, ff_time + 1):
            feasible = True
            # 检查每个资源类型
            for res, demand in act.resource_request.items():
                if res not in self.project.local_resources:
                    continue  # 忽略非本地资源
                res_limit = self.project.local_resources[res]

                # 检查时间段 [ef, ef + delta)
                for t in range(ef, ef + delta):
                    # 处理时间轴越界（假设后续时间资源未被占用）
                    if t >= len(self.resource_usage[res]):
                        available = 0
                    else:
                        available = self.resource_usage[res][t]
                    # 如果资源需求超过限额，标记为不可行
                    if available + demand > res_limit:
                        feasible = False
                        break
                if not feasible:
                    break  # 当前资源类型不满足，跳出循环

            if feasible:
                delta_max = delta  # 更新可行Δ
            else:
                break  # 后续更大的Δ也不可行，提前终止

        return delta_max

    def _find_virtual_activity(self) -> int:
        """识别虚活动（没有后继活动的活动）"""
        for act in self.project.activities.values():
            if not act.successors:
                return act.activity_id
        return max(self.project.activities.keys())  # 默认取最大ID

    def calculate_max_path_length(self, activity: Activity) -> int:
        """动态规划计算到虚活动的最大路径长度"""
        if activity.activity_id in self.path_length_cache:
            return self.path_length_cache[activity.activity_id]

        if not activity.successors:
            # 虚活动自身路径长度为0
            self.path_length_cache[activity.activity_id] = 0
            return 0

        max_length = 0
        for succ_id in activity.successors:
            succ_act = self.project.activities[succ_id]
            path_length = 1 + self.calculate_max_path_length(succ_act)
            max_length = max(max_length, path_length)

        self.path_length_cache[activity.activity_id] = max_length
        return max_length

    def calculate_ciw(self, act: Activity) -> float:
        """计算单个活动的CIW值"""
        ff = self.ff_cache.get(act.activity_id, 0)
        if ff <= 0:
            return 0.0

        # 计算指数衰减项
        alpha = 1 + (act.duration - 1) / 10
        sum_exp = sum(math.exp(-i / alpha) for i in range(1, ff + 1))

        # 获取路径长度
        path_length = self.path_length_cache.get(act.activity_id, 0)

        return path_length * sum_exp


class Individual:
    """NSGA-II 个体类：表示一个调度解，支持多目标优化和约束处理"""

    def __init__(self, project: Project, chromosome: Optional[List[int]] = None):
        """
        参数:
            project: 关联的项目对象
            chromosome: 优先级顺序编码（若为None则随机生成）
        """
        self.project = deepcopy(project)  # 项目副本（避免修改原始项目）
        self.mutation_rate = config["algorithm"]["mutation_probability"]  # 变异概率
        self.chromosome = chromosome or self._generate_constrained_chromosome()  # 染色体编码（活动优先级顺序）
        self.schedule: Optional[Schedule] = None
        self.fitness: Optional[float, float] = None  # (工期, -鲁棒性)

        self._init_schedule()  # 初始化调度并计算适应度

    def _init_schedule(self) -> None:
        """初始化调度并计算适应度"""
        # 1. 将染色体转换为优先级字典
        priority_order = {act_id: idx + 1 for idx, act_id in enumerate(self.chromosome)}
        # 将染色体转换为优先级字典并按活动ID排序
        # priority_order = {act_id: idx + 1 for idx, act_id in sorted(enumerate(self.chromosome), key=lambda x: x[1])}
        # 2. 生成调度
        self.schedule = Schedule(self.project)
        self.schedule.priority_order = priority_order
        self.schedule.generate_schedule()

        # 3. 计算适应度
        makespan = self.schedule.total_duration
        robustness = self.schedule.robustness
        self.fitness = (makespan, -robustness)  # 鲁棒性需最大化，取负转为最小化

    @property
    def objectives(self) -> Tuple[float, float]:
        """获取目标值（工期, -鲁棒性）"""
        return self.fitness

    def __lt__(self, other: 'Individual') -> bool:
        """定义支配关系比较"""
        return (self.fitness[0] <= other.fitness[0]) and (self.fitness[1] <= other.fitness[1]) \
               and (self.fitness[0] < other.fitness[0] or self.fitness[1] < other.fitness[1])

    def _generate_constrained_chromosome(self) -> List[int]:
        """
        生成满足紧后关系约束的随机优先级顺序编码（拓扑排序的随机变体）
        :param self: 项目对象，包含活动及其依赖关系
        :return: 符合依赖关系的活动ID列表
        """
        # 1. 构建依赖图和入度字典
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        for act in self.project.activities.values():
            for succ_id in act.successors:
                graph[act.activity_id].append(succ_id)
                in_degree[succ_id] += 1

        # 2. 初始化队列（入度为0的活动）
        queue = deque([act_id for act_id in self.project.activities if in_degree[act_id] == 0])
        chromosome = []

        # 3. 随机拓扑排序
        while queue:
            # 随机选择队列中的一个活动
            current = random.choice(queue)
            queue.remove(current)
            chromosome.append(current)

            # 更新紧后活动的入度
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 4. 检查是否所有活动都被处理
        if len(chromosome) != len(self.project.activities):
            raise ValueError("依赖关系存在环，无法生成合法优先级顺序")

        return chromosome

    def mutate(self) -> None:
        """变异操作：对每个基因位按概率进行安全位置调整"""
        new_chromosome = self.chromosome.copy()
        n = len(new_chromosome)
        # 从后向前遍历，避免插入操作影响未处理的基因位索引
        for i in reversed(range(n)):
            if random.random() > config["algorithm"]["mutation_probability"]:
                continue

            act_id = new_chromosome[i]
            act = self.project.activities[act_id]

            # === 步骤1：确定可行区域 ===
            # 左边界：所有紧前活动的最大位置（右侧）
            left = max(
                [new_chromosome.index(pred_id) for pred_id in act.predecessors],
                default=0
            )
            # 右边界：所有紧后活动的最小位置（左侧）
            right = min(
                [new_chromosome.index(succ_id) for succ_id in act.successors],
                default=n - 1
            ) if act.successors else n - 1

            # 修正可行区域为 [left, right]
            feasible_region = list(range(left, right + 1))
            # 移除当前位置
            if i in feasible_region:
                feasible_region.remove(i)

            if not feasible_region:
                continue  # 无可插入位置

            # === 步骤2：随机选择新位置 ===
            new_pos = random.choice(feasible_region)

            # === 步骤3：移动基因并调整序列 ===
            # 删除原位置基因
            del new_chromosome[i]
            # 插入新位置
            new_chromosome.insert(new_pos, act_id)

        # 更新染色体
        self.chromosome = new_chromosome


class NSGA2Algorithm:
    """NSGA-II 算法核心类，支持多目标优化和精英保留策略"""

    def __init__(self, project, program):
        self.program = program
        self.project = project
        self.population: List[Individual] = []
        self.pop_size = config["algorithm"]["population_size"]
        self.max_generations = config["algorithm"]["generations"]
        self.crossover_prob = config["algorithm"]["crossover_probability"]

        self.history_knee_points = []  # 记录每次迭代的Knee点
        self.best_knee = None  # 全局最优Knee点

        # 初始化种群
        self._initialize_population()

    def _initialize_population(self) -> None:
        """初始化种群（随机生成个体）"""
        self.population = [Individual(self.project) for _ in range(self.pop_size)]

    def _non_dominated_sort(self, individuals: List[Individual]) -> List[List[int]]:
        """非支配排序"""
        objectives = np.array([ind.objectives for ind in individuals])
        return NonDominatedSorting().do(objectives)

    @staticmethod
    def _crowding_distance(front: List[Individual]) -> List[float]:
        """拥挤度计算"""
        num_objs = len(front[0].objectives)
        distances = [0.0] * len(front)

        for obj_idx in range(num_objs):
            # 按目标值排序
            sorted_front = sorted(front, key=lambda x: x.objectives[obj_idx])
            min_obj = sorted_front[0].objectives[obj_idx]
            max_obj = sorted_front[-1].objectives[obj_idx]

            # 边界个体拥挤度设为无穷大
            distances[0] = distances[-1] = np.inf
            for i in range(1, len(front) - 1):
                if max_obj - min_obj == 0:
                    continue
                distances[i] += (sorted_front[i + 1].objectives[obj_idx] -
                                 sorted_front[i - 1].objectives[obj_idx]) / (max_obj - min_obj)
        return distances

    @staticmethod
    def _tournament_selection(population: List[Individual], k: int = 2) -> Individual:
        """二元锦标赛选择"""
        candidates = random.sample(population, k)
        return min(candidates, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """执行交叉操作（按概率）"""
        if random.random() > self.crossover_prob:
            return parent1, parent2  # 不发生交叉

        return self._single_point_crossover(parent1, parent2)

    @staticmethod
    def _single_point_crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        单点交叉操作：生成女儿和儿子个体
        :param parent1: 父代个体1（母亲）
        :param parent2: 父代个体2（父亲）
        :return: (daughter, son) 两个新个体
        """
        # 检查染色体长度一致
        assert len(parent1.chromosome) == len(parent2.chromosome), "染色体长度不一致"
        J = len(parent1.chromosome)

        # 随机选择交叉点 r (1 ≤ r ≤ J-1)
        r = random.randint(1, J - 1)

        # === 生成女儿D ===
        # 前r个基因来自母亲
        daughter_chromo = parent1.chromosome[:r].copy()
        # 后J-r个基因按父亲顺序填充未被选中的活动
        for gene in parent2.chromosome:
            if gene not in daughter_chromo and len(daughter_chromo) < J:
                daughter_chromo.append(gene)
        daughter = Individual(parent1.project, daughter_chromo)

        # === 生成儿子S ===
        # 前r个基因来自父亲
        son_chromo = parent2.chromosome[:r].copy()
        # 后J-r个基因按母亲顺序填充未被选中的活动
        for gene in parent1.chromosome:
            if gene not in son_chromo and len(son_chromo) < J:
                son_chromo.append(gene)
        son = Individual(parent2.project, son_chromo)

        return daughter, son

    def evolve(self) -> None:
        """执行进化循环"""
        for gen in range(self.max_generations):
            # 1. 生成子代种群
            offspring = []
            while len(offspring) < self.pop_size:
                # 选择父代
                parent1 = self._tournament_selection(self.population)
                parent2 = self._tournament_selection(self.population)
                # 交叉和变异
                daughter, son = self._crossover(parent1, parent2)
                # 变异并添加到子代种群
                daughter.mutate()
                son.mutate()
                offspring.extend([daughter, son])

            # 2. 合并父代和子代
            combined = self.population + offspring

            # 3. 非支配排序和拥挤度计算
            fronts = self._non_dominated_sort(combined)
            next_population = []
            front_idx = 0
            while len(next_population) + len(fronts[front_idx]) <= self.pop_size:
                next_population += [combined[i] for i in fronts[front_idx]]
                front_idx += 1

            # 4. 填充剩余位置（按拥挤度）
            if len(next_population) < self.pop_size:
                last_front = [combined[i] for i in fronts[front_idx]]
                crowding_dist = self._crowding_distance(last_front)
                sorted_indices = sorted(
                    range(len(last_front)),
                    key=lambda x: crowding_dist[x],
                    reverse=True
                )
                needed = self.pop_size - len(next_population)
                next_population += [last_front[i] for i in sorted_indices[:needed]]

            self.population = next_population

            # === 新增：计算当前代的帕累托前沿并更新Knee点 ===
            current_front = self._get_current_pareto_front()
            current_knee = self._select_knee_point(current_front)
            self._update_best_knee(current_knee)
            self.history_knee_points.append({
                "generation": gen,
                "makespan": current_knee.objectives[0],
                "robustness": -current_knee.objectives[1],
                "solution": current_knee
            })

        # 最终输出最优Knee点
        self._print_final_knee_info()
        # 迭代结束后，将最优Knee点的共享资源需求注入Project
        self._update_project_with_knee_solution()

    def _get_current_pareto_front(self) -> List[Individual]:
        """获取当前代的帕累托前沿第一层"""
        fronts = self._non_dominated_sort(self.population)
        return [self.population[i] for i in fronts[0]]

    def _select_knee_point(self, front: List[Individual]) -> Individual:
        """使用MMD方法选择当前前沿的Knee点"""
        # 归一化目标值
        makespans = [ind.objectives[0] for ind in front]
        robustness = [-ind.objectives[1] for ind in front]  # 鲁棒性需最大化

        # 计算理想点和最差点
        ideal = [min(makespans), max(robustness)]
        nadir = [max(makespans), min(robustness)]

        # 归一化公式
        normalized = []
        for ind in front:
            norm_makespan = (ind.objectives[0] - ideal[0]) / (nadir[0] - ideal[0])
            norm_robustness = ((-ind.objectives[1]) - ideal[1]) / (nadir[1] - ideal[1])
            normalized.append(norm_makespan + norm_robustness)  # MMD

        # 选择MMD最小的个体
        return front[normalized.index(min(normalized))]

    def _update_best_knee(self, candidate: Individual) -> None:
        """更新全局最优Knee点（支配关系比较）"""
        if self.best_knee is None:
            self.best_knee = candidate
            return

        # 判断候选解是否支配当前最优解
        if not (candidate.objectives[0] <= self.best_knee.objectives[0] and
                candidate.objectives[1] <= self.best_knee.objectives[1] and
                (candidate.objectives[0] < self.best_knee.objectives[0] or
                 candidate.objectives[1] < self.best_knee.objectives[1])):
            self.best_knee = candidate

    def _print_final_knee_info(self) -> None:
        """输出最优Knee点的调度详情"""
        best_schedule = self.best_knee.schedule
        print("\n=== 最优Knee点解 ===")
        print(f"总工期: {best_schedule.total_duration}")
        print(f"鲁棒性: {best_schedule.robustness:.2f}")
        print("共享资源需求:")
        for proj_id, proj in self.best_knee.project.projects.items():
            shared_res = proj.shared_resources_request
            print(f"  项目 {proj_id}: {shared_res}")

    def _update_project_with_knee_solution(self) -> None:
        """将最优Knee点的共享资源需求写入Project"""
        if not self.best_knee:
            return

        # 获取调度中的共享资源需求
        shared_resources = defaultdict(int)
        for proj_id, proj in self.best_knee.project.projects.items():
            for act in proj.activities.values():
                for res, demand in act.resource_request.items():
                    if res.startswith("global "):  # 假设共享资源以"global"为前缀
                        shared_resources[res] += demand

        # 更新Program中所有Project的 shared_resources_request
        for proj in self.program.projects.values():
            proj.shared_resources_request = dict(shared_resources)


class GurobiAlgorithm:
    """精确解类"""

    def __init__(self, program: Program):
        self.program = program
        self.model = Model("ProgramScheduling")
        self._build_model()

    def _build_model(self) -> None:
        """构建Gurobi模型求解最短工期精确解"""
        # === 定义变量 ===
        projects = self.program.projects
        T = self._calculate_upper_bound()  # 总工期上界（各项目工期之和）

        # 变量1：项目开始时间（整数变量）
        start_vars = self.model.addVars(
            projects.keys(), lb=0, ub=T, vtype=GRB.INTEGER, name="start"
        )
        # 变量2：项目是否在时间t占用全局资源（0-1变量）
        resource_vars = self.model.addVars(
            self.program.global_resources.keys(),
            projects.keys(),
            range(T),
            vtype=GRB.BINARY,
            name="resource_usage"
        )

        # === 目标函数：最小化总工期 ===
        makespan = self.model.addVar(name="makespan")
        self.model.addConstr(
            makespan == quicksum(
                start_vars[proj_id] + proj.total_duration
                for proj_id, proj in projects.items()
            ),
            "makespan_def"
        )
        self.model.setObjective(makespan, GRB.MINIMIZE)

        # === 约束1：项目间依赖关系 ===
        for proj_id, proj in projects.items():
            for pred_id in proj.predecessors:
                if pred_id in projects:
                    self.model.addConstr(
                        start_vars[proj_id] >= start_vars[pred_id] + projects[pred_id].total_duration,
                        f"proj_precedence_{pred_id}_{proj_id}"
                    )

        # === 约束2：全局资源限制 ===
        for res in self.program.global_resources:
            res_capacity = self.program.global_resources[res]
            for t in range(T):
                # 所有项目在时间t对资源res的占用量
                total_usage = quicksum(
                    resource_vars[res, proj_id, t] * proj.shared_resources_request.get(res, 0)
                    for proj_id, proj in projects.items()
                )
                self.model.addConstr(
                    total_usage <= res_capacity,
                    f"global_res_{res}_at_{t}"
                )

        # === 约束3：资源占用时间窗 ===
        for proj_id, proj in projects.items():
            duration = proj.total_duration
            for res, demand in proj.shared_resources_request.items():
                if demand == 0:
                    continue
                for t in range(T):
                    # 如果项目在时间s开始，则在[s, s+duration)期间占用资源
                    self.model.addGenConstrIndicator(
                        resource_vars[res, proj_id, t],
                        True,
                        start_vars[proj_id] <= t,
                        name=f"res_{res}_proj_{proj_id}_start_le_{t}"
                    )
                    self.model.addGenConstrIndicator(
                        resource_vars[res, proj_id, t],
                        True,
                        start_vars[proj_id] + duration > t,
                        name=f"res_{res}_proj_{proj_id}_end_gt_{t}"
                    )

    def _calculate_upper_bound(self) -> int:
        """计算总工期上界（各项目工期之和）"""
        return sum(proj.total_duration for proj in self.program.projects.values())

    def solve(self) -> Dict[str, Any]:
        """求解并返回基准调度计划，同时更新Program中的项目开始时间"""
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            # 获取求解结果中的项目开始时间
            start_times = {
                proj_id: int(self.model.getVarByName(f"start[{proj_id}]").x)
                for proj_id in self.program.projects
            }
            makespan = int(self.model.getVarByName("makespan").x)

            # === 将开始时间更新到Program的各个Project中 ===
            for proj_id, start in start_times.items():
                self.program.projects[proj_id].start_time = start
                # 更新项目总工期（可选）
                self.program.projects[proj_id].total_duration = (
                    self.program.projects[proj_id].total_duration  # 假设已通过其他方法计算
                )

            # 更新Program的总工期和资源使用
            self.program.total_duration = makespan
            self.program.resource_usage = self._extract_resource_usage()

            return {
                "start_times": start_times,
                "makespan": makespan,
                "resource_usage": self.program.resource_usage
            }
        else:
            raise Exception("未找到可行解")

    def _extract_resource_usage(self) -> Dict[str, List[int]]:
        """提取全局资源时间轴占用情况"""
        T = self._calculate_upper_bound()
        resource_usage = {res: [0] * T for res in self.program.global_resources}

        for res in self.program.global_resources:
            for proj_id in self.program.projects:
                for t in range(T):
                    var = self.model.getVarByName(f"resource_usage[{res},{proj_id},{t}]")
                    if var.x > 0.5:
                        resource_usage[res][t] += self.program.projects[proj_id].shared_resources_request.get(res, 0)
        return resource_usage


class MTPCAlgorithm:
    """
    MTPC算法类：基于基准调度计划优化资源分配，最小化拖期惩罚成本
    核心功能：
    1. 根据基准调度计划生成活动序列 LIST_1
    2. 按顺序分配资源，动态添加资源弧
    3. 计算拖期惩罚成本（TPC）
    """

    def __init__(self, program: Program):
        self.program = program
        self.A_R: Set[Tuple[str, str]] = set()  # 资源弧集合（格式：(来源项目ID, 目标项目ID)）
        self.alloc: Dict[str, Dict[str, int]] = {}  # 资源分配量 {项目ID: {资源类型: 数量}}
        self.tpc: float = 0.0  # 总拖期惩罚成本

    def run(self) -> Dict[str, Any]:
        """执行MTPC主流程"""
        # 初始化资源分配（虚拟项目0拥有全部全局资源）
        self._initialize_alloc()

        # 生成活动序列 LIST_1（按基准开始时间、权重、项目ID排序）
        sorted_projects = self._generate_project_list()

        # 遍历项目进行资源分配
        for proj_id in sorted_projects:
            self._allocate_resources_for_project(proj_id)

        # 更新Program的全局资源占用
        self.program.resource_usage = self._get_global_resource_usage()

        return {
            "resource_arcs": self.A_R,
            "allocations": self.alloc,
            "total_tpc": self.tpc,
            "global_resource_usage": self.program.resource_usage
        }

    def _initialize_alloc(self) -> None:
        """初始化资源分配：虚拟项目拥有全部全局资源"""
        # 初始化虚拟项目1资源分配为全局资源容量
        self.alloc["1"] = {res: cap for res, cap in self.program.global_resources.items()}
        # 初始化其他项目资源分配为0
        for proj_id in self.program.projects:
            self.alloc[proj_id] = {res: 0 for res in self.program.global_resources}

    def _generate_project_list(self) -> List[str]:
        """生成项目序列 LIST_1（排序规则：基准开始时间↑ → 权重↓ → 项目ID↑）"""
        projects = list(self.program.projects.values())
        # 假设Project类有权重属性（若无可替换为其他业务逻辑）
        sorted_projects = sorted(
            projects,
            key=lambda p: (
                p.start_time,  # 基准开始时间升序
                -p.priority,  # 权重降序（需在Project类中添加priority字段）
                p.project_id  # 项目ID升序
            )
        )
        return [p.project_id for p in sorted_projects]

    def _allocate_resources_for_project(self, proj_id: str) -> None:
        """为项目分配资源（核心逻辑）"""
        proj = self.program.projects[proj_id]
        required_resources = proj.shared_resources_request

        # 步骤1：计算可用资源（来自紧前项目）
        avail = self._calculate_available_resources(proj)
        if all(avail[res] >= required_resources.get(res, 0) for res in self.program.global_resources):
            self._direct_allocation(proj, avail)
        else:
            # 步骤2：添加额外资源弧
            H_j = self._find_minimal_Hj(proj)
            self.A_R.update(H_j)
            self._allocate_with_priority(proj, H_j)

    def _calculate_available_resources(self, proj: Project) -> Dict[str, int]:
        """计算可用资源（来自紧前项目和资源弧）"""
        avail = {res: 0 for res in self.program.global_resources}
        for pred_id in proj.predecessors:
            if pred_id in self.program.projects:
                avail = {
                    res: avail[res] + self.alloc[pred_id][res]
                    for res in self.program.global_resources
                }
        return avail

    def _find_minimal_Hj(self, proj: Project) -> Set[Tuple[str, str]]:
        """寻找最小备选集 H_j*（简化实现：选择第一个可行解）"""
        candidates = []
        # 遍历所有可能的前驱项目（非紧前且时间不冲突）
        for h_proj in self.program.projects.values():
            if (h_proj.project_id not in proj.predecessors and
                    h_proj.start_time + h_proj.total_duration <= proj.start_time):
                candidates.append((h_proj.project_id, proj.project_id))
        return set(candidates[:1])  # 实际需按TPC最小化选择

    def _direct_allocation(self, proj: Project, avail: Dict[str, int]) -> None:
        """直接分配资源（无需添加额外弧）"""
        for res in self.program.global_resources:
            demand = proj.shared_resources_request.get(res, 0)
            self.alloc[proj.project_id][res] = demand
            # 从虚拟项目0扣除资源
            self.alloc["0"][res] -= demand

    def _allocate_with_priority(self, proj: Project, H_j: Set[Tuple[str, str]]) -> None:
        """根据优先准则分配资源（简化实现）"""
        # 按准则排序（此处简化，实际需实现6个准则）
        sorted_Hj = sorted(
            H_j,
            key=lambda arc: (
                -len(self.program.projects[arc[0]].successors),  # 准则1：紧后数量升序
                -self.program.projects[arc[0]].weight,  # 准则2：权重降序
                -(self.program.projects[arc[0]].start_time +  # 准则3：结束时间降序
                  self.program.projects[arc[0]].total_duration),
                -self.alloc[arc[0]][res]  # 准则4：可提供量降序
            )
        )
        # 分配资源
        for res in self.program.global_resources:
            demand = proj.shared_resources_request.get(res, 0)
            for h_proj_id, _ in sorted_Hj:
                flow = min(self.alloc[h_proj_id][res], demand)
                self.alloc[proj.project_id][res] += flow
                self.alloc[h_proj_id][res] -= flow
                demand -= flow
                if demand <= 0:
                    break

    def _get_global_resource_usage(self) -> Dict[str, List[int]]:
        """提取全局资源时间轴占用"""
        T = self.program.total_duration
        resource_usage = {res: [0] * T for res in self.program.global_resources}
        for proj in self.program.projects.values():
            start = proj.start_time
            end = start + proj.total_duration
            for res in self.program.global_resources:
                demand = proj.shared_resources_request.get(res, 0)
                for t in range(start, end):
                    if t < T:
                        resource_usage[res][t] += demand
        return resource_usage

    def _calculate_tpc(self, proj: Project) -> float:
        """计算单个项目的拖期惩罚成本（修正版本）"""
        tpc = 0.0
        for pred_id in proj.predecessors:
            if (pred_id, proj.project_id) in self.A_R:
                pred_project = self.program.projects.get(pred_id)
                if not pred_project:
                    continue

                # 计算时间差（实际工期 - 计划时间差）
                planned_time_window = proj.start_time - pred_project.start_time
                time_diff = pred_project.total_duration - planned_time_window
                penalty = max(0, time_diff)

                # 拖期惩罚成本 = 项目权重 * 拖期概率 * 惩罚值
                tpc += proj.weight * 0.2 * penalty  # 假设拖期概率为0.2

        self.tpc += tpc
        return tpc


class ArtiguesAlgorithm:
    """
    Artigues算法（项目群层面）：生成可行性资源流网络（不优化鲁棒性）
    """

    def __init__(self, program: Program):
        self.program = program
        self.alloc: Dict[str, Dict[str, int]] = {}  # {项目ID: {资源类型: 量}}
        self.resource_arcs: Set[Tuple[str, str, str]] = set()  # 资源弧 (i, j, k)

    def run(self) -> Dict[str, Any]:
        """执行算法主流程"""
        self._initialize_alloc()
        sorted_projects = self._get_sorted_projects()

        for proj in sorted_projects:
            self._allocate_resources(proj)

        return {
            "allocations": self.alloc,
            "resource_arcs": self.resource_arcs
        }

    def _initialize_alloc(self) -> None:
        """初始化资源分配：虚拟项目0拥有全部资源"""
        self.alloc["0"] = {res: cap for res, cap in self.program.global_resources.items()}
        for proj_id in self.program.projects:
            self.alloc[proj_id] = {res: 0 for res in self.program.global_resources}

    def _get_sorted_projects(self) -> List[Project]:
        """按基准开始时间升序返回项目"""
        return sorted(
            self.program.projects.values(),
            key=lambda p: p.start_time
        )

    def _allocate_resources(self, proj: Project) -> None:
        """为项目分配资源"""
        for res in self.program.global_resources:
            req = proj.shared_resources_request.get(res, 0)
            if req <= 0:
                continue

            # 从虚拟项目0分配资源
            if self.alloc["0"][res] >= req:
                flow = req
                self.alloc[proj.project_id][res] += flow
                self.alloc["0"][res] -= flow
                self.resource_arcs.add(("0", proj.project_id, res))
            else:
                # 从已完成项目分配资源（简化实现）
                for h_proj in self._get_completed_projects(proj.start_time):
                    if self.alloc[h_proj.project_id][res] > 0:
                        flow = min(req, self.alloc[h_proj.project_id][res])
                        self.alloc[proj.project_id][res] += flow
                        self.alloc[h_proj.project_id][res] -= flow
                        self.resource_arcs.add((h_proj.project_id, proj.project_id, res))
                        req -= flow
                        if req == 0:
                            break

    def _get_completed_projects(self, current_time: int) -> List[Project]:
        """获取在当前时间前已完成的项目"""
        return [
            p for p in self.program.projects.values()
            if p.start_time + p.total_duration <= current_time
        ]


class STCAlgorithm:
    """
    STC算法类：在资源分配基础上插入缓冲以提升鲁棒性
    核心步骤：
    1. 计算每个活动的开始时间关键度（stc值）
    2. 按stc值降序尝试插入缓冲
    3. 验证缓冲插入后的可行性与鲁棒性提升
    """

    def __init__(self, program: Program):
        self.program = program
        self.best_schedule = None  # 存储最优调度计划
        self.current_schedule = None  # 当前调度计划
        self.buffer_added = []  # 记录已插入的缓冲

    def run(self, max_iter: int = 100) -> Dict[str, Any]:
        """执行STC主流程"""
        # 初始化：复制基准调度计划
        self.current_schedule = deepcopy(self.program)
        self.best_schedule = deepcopy(self.program)
        best_robustness = self._calculate_robustness(self.best_schedule)

        # 迭代优化
        for _ in range(max_iter):
            # 计算所有活动的stc值
            stc_values = self._calculate_stc_values()
            # 按stc值降序排序活动
            sorted_activities = self._sort_activities_by_stc(stc_values)
            improvement_found = False

            for act in sorted_activities:
                # 跳过stc值为0的活动
                if stc_values[act.activity_id] <= 0:
                    continue

                # 尝试在活动前插入缓冲
                original_start = act.es
                new_start = original_start + 1  # 插入1单位缓冲
                self._update_schedule(act, new_start)

                # 检查资源可行性和鲁棒性提升
                if self._is_schedule_feasible() and self._is_robustness_improved():
                    self.best_schedule = deepcopy(self.current_schedule)
                    improvement_found = True
                    self.buffer_added.append(act.activity_id)
                    break  # 进入下一轮迭代
                else:
                    # 回退缓冲插入
                    self._update_schedule(act, original_start)

            if not improvement_found:
                break  # 无进一步优化可能

        return {
            "best_schedule": self.best_schedule,
            "buffers_added": self.buffer_added,
            "final_robustness": self._calculate_robustness(self.best_schedule)
        }

    def _calculate_stc_values(self) -> Dict[int, float]:
        """计算所有活动的stc值（公式4-9, 4-10）"""
        stc_values = {}
        for proj in self.current_schedule.projects.values():
            for act in proj.activities.values():
                gamma_j = 0.0
                # 遍历所有前驱活动
                for pred_id in act.predecessors:
                    pred_act = self._find_activity(pred_id)
                    lpl = self._calculate_lpl(pred_act, act)
                    # 简化假设：P(d_j > s_j - s_i - LPL) = 0.2（需根据实际数据调整）
                    prob = 0.2 if (act.es - pred_act.es - lpl) < pred_act.duration else 0.0
                    gamma_j += prob
                stc = gamma_j * act.priority
                stc_values[act.activity_id] = stc
        return stc_values

    def _calculate_lpl(self, pred_act: Activity, succ_act: Activity) -> int:
        """计算最长路径LPL(i,j)（简化为前驱活动工期之和）"""
        return pred_act.duration  # 可扩展为关键路径计算

    def _sort_activities_by_stc(self, stc_values: Dict[int, float]) -> List[Activity]:
        """按stc值降序返回活动列表"""
        all_activities = []
        for proj in self.current_schedule.projects.values():
            all_activities.extend(proj.activities.values())
        return sorted(
            all_activities,
            key=lambda x: stc_values.get(x.activity_id, 0),
            reverse=True
        )

    def _update_schedule(self, act: Activity, new_start: int) -> None:
        """更新活动的开始时间，并调整后续活动的时序"""
        # 调整当前活动开始时间
        act.es = new_start
        act.ef = new_start + act.duration
        # 递归调整所有后继活动（简化实现，实际需处理资源冲突）
        for succ_id in act.successors:
            succ_act = self._find_activity(succ_id)
            if succ_act.es < act.ef:
                self._update_schedule(succ_act, act.ef)

    def _is_schedule_feasible(self) -> bool:
        """检查调度计划是否可行（资源约束满足）"""
        # 简化实现：假设资源已由MTPC/Artigues分配完成，仅检查时间冲突
        for proj in self.current_schedule.projects.values():
            for act in proj.activities.values():
                for pred_id in act.predecessors:
                    pred_act = self._find_activity(pred_id)
                    if act.es < pred_act.ef:
                        return False
        return True

    def _is_robustness_improved(self) -> bool:
        """检查鲁棒性是否提升（简化实现）"""
        current_robustness = self._calculate_robustness(self.current_schedule)
        best_robustness = self._calculate_robustness(self.best_schedule)
        return current_robustness > best_robustness

    def _calculate_robustness(self, program: Program) -> float:
        """计算调度计划的鲁棒性（示例：基于缓冲总长度）"""
        robustness = 0.0
        for proj in program.projects.values():
            for act in proj.activities.values():
                robustness += (act.es - self.program.projects[proj.project_id].activities[act.activity_id].es)
        return robustness

    def _find_activity(self, act_id: int) -> Optional[Activity]:
        """根据ID查找活动"""
        for proj in self.current_schedule.projects.values():
            if act_id in proj.activities:
                return proj.activities[act_id]
        return None
