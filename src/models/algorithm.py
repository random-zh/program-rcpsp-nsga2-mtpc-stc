# model/algorithm.py
import math
import random
from collections import defaultdict, deque
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
from scipy.stats import lognorm

from .problem import Project, Activity, Program
from utils.config_loader import config
from gurobipy import Model, GRB, quicksum, max_, LinExpr
from tqdm import trange


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
        self.global_resources_request: Optional[int] = None  # 最大共享资源需求

        # === 鲁棒性计算缓存 ===
        self.ff_cache: Dict[int, int] = {}  # {活动ID: 自由时差}
        self.path_length_cache: Dict[int, int] = {}  # {活动ID: 到虚活动的路径长度}

        # === 初始化资源时间轴为0 ===
        max_duration_guess = sum(act.duration for act in self.project.activities.values())  # 所有活动工期之和
        for res in self.project.local_resources:
            self.resource_usage[res] = [0] * (max_duration_guess + 1)

    def to_dict(self) -> dict:
        return {
            "priority_order": self.priority_order,
            "start_times": self.start_times,
            "total_duration": self.total_duration,
            "robustness": self.robustness,
            "global_resources_request": self.global_resources_request,
            "project": self.project.to_dict(),  # 使用Project的序列化方法
        }

    # === 计算最大共享资源需求 ===
    def _calculate_max_global_resources_request(self) -> None:
        """计算最大共享资源需求"""
        if not self.resource_usage:
            raise Exception("资源时间轴未初始化")

        # 获取第一个资源的名称
        first_resource = next(iter(self.resource_usage.keys()))

        # 计算该资源的最大占用量
        max_demand = max(self.resource_usage[first_resource])

        self.global_resources_request = max_demand

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
            reverse=False
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

        # 计算总工期和鲁棒性和最大共享资源需求
        self.calculate_total_duration()
        self.calculate_robustness()
        self._calculate_max_global_resources_request()

    # === 私有资源管理方法 ===
    def _is_resource_available(self, act: Activity, start_time: int) -> bool:
        """检查资源在时间段 [start_time, start_time+duration) 是否可用"""
        for res, demand in act.resource_request.items():
            # 本地资源检查
            if res in self.project.local_resources:
                for t in range(start_time, start_time + act.duration):
                    if t >= len(self.resource_usage[res]):
                        raise Exception(f"资源时间轴长度不足：{res} @ {t}")
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
                # if res not in self.project.local_resources:
                #     continue  # 忽略非本地资源
                res_limit = self.project.local_resources[res]

                # 检查时间段 [ef, ef + delta)
                for t in range(ef, ef + delta):
                    # 处理时间轴越界（假设后续时间资源未被占用）
                    if t >= len(self.resource_usage[res]):
                        raise Exception(f"资源时间轴长度不足：{res} @ {t}")
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
        self.schedule: [Schedule] = None
        self.fitness: Optional[float, float] = None  # (工期, -鲁棒性)
        self.rank = None  # 非支配等级（第几层）
        self.crowding_distance = 0.0  # 拥挤度

        self._init_schedule()  # 初始化调度并计算适应度

    def to_dict(self) -> dict:
        return {
            "chromosome": self.chromosome,
            "fitness": self.fitness,
            "schedule": self.schedule.to_dict() if self.schedule else None,
            "rank": self.rank,
            "crowding_distance": self.crowding_distance
        }

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

    # === 新增支配关系判断方法 ===
    def dominates(self, other: 'Individual') -> bool:
        """判断当前个体是否支配另一个个体（目标为最小化）"""
        obj_self = self.objectives
        obj_other = other.objectives
        return (obj_self[0] <= obj_other[0] and
                obj_self[1] <= obj_other[1] and
                (obj_self[0] < obj_other[0] or
                 obj_self[1] < obj_other[1]))

    # # === 个体支配关系比较 ===
    # def __lt__(self, other: 'Individual') -> bool:
    #     """定义支配关系比较"""
    #     return (self.fitness[0] <= other.fitness[0]) and (self.fitness[1] <= other.fitness[1]) \
    #            and (self.fitness[0] < other.fitness[0] or self.fitness[1] < other.fitness[1])

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
            feasible_region = list(range(left + 1, right))
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

        # 无变异直接返回
        if new_chromosome == self.chromosome:
            return

        # 更新染色体
        self.chromosome = new_chromosome
        # 重新初始化调度
        self._init_schedule()


def _non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """高效非支配排序 O(MN^2)"""
    n = len(population)
    fronts = [[]]  # 存储各前沿层的个体对象
    domination_counts = [0] * n  # 每个个体被支配的次数（使用索引）
    dominated_map = defaultdict(list)  # 键: 个体索引, 值: 被其支配的个体索引列表

    # === 步骤1: 构建支配关系 ===
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # 判断支配关系
            if population[i].dominates(population[j]):
                dominated_map[i].append(j)
            elif population[j].dominates(population[i]):
                domination_counts[i] += 1

        # 第一前沿层（无被支配的个体）
        if domination_counts[i] == 0:
            population[i].rank = 0
            fronts[0].append(population[i])

    # === 步骤2: 分层处理 ===
    current_front = 0
    while fronts[current_front]:
        next_front = []
        # 遍历当前前沿层的个体索引（注意：这里用个体对象反向找索引）
        for ind in fronts[current_front]:
            idx = population.index(ind)  # 获取个体在population中的索引
            for dominated_idx in dominated_map[idx]:
                domination_counts[dominated_idx] -= 1
                if domination_counts[dominated_idx] == 0:
                    population[dominated_idx].rank = current_front + 1
                    next_front.append(population[dominated_idx])
        if 1 == 1:
            fronts.append(next_front)
            current_front += 1
    return fronts[:-1]  # 去除最后一个空层


class NSGA2Algorithm:
    """NSGA-II 算法核心类，支持多目标优化和精英保留策略"""

    def __init__(self, project, program):
        self.program = program
        self.project = project
        self.population: List[Individual] = []
        self.pop_size = config["algorithm"]["population_size"]
        self.max_generations = config["algorithm"]["generations"]
        self.crossover_prob = config["algorithm"]["crossover_probability"]

        self.history_best_points = []  # 记录每次迭代的最值点
        self.best_knee: Optional[Individual] = None  # 全局最优Knee点
        self.fronts = None  # 最终的非支配层级结果

        # 初始化种群
        self._initialize_population()

    def _initialize_population(self) -> None:
        """初始化种群（随机生成个体）"""
        self.population = [Individual(self.project) for _ in range(self.pop_size)]

    def evolve(self) -> None:
        """执行进化循环"""
        # 步骤1: 初始化种群 (已通过构造函数完成)
        # 步骤2: 目标函数计算 (已在个体初始化时完成)
        for gen in trange(self.max_generations, desc="NSGA2 Progress", unit="gen", leave=False, position=0,
                          dynamic_ncols=True):
            # 步骤3: 非支配排序
            fronts = _non_dominated_sort(self.population)

            # 步骤4: 拥挤度计算
            for front in fronts:
                self._calculate_crowding_distance(front)

            # 步骤5: 父代选择
            selected_parents = self._select_parents(fronts)

            # 步骤6: 生成子代
            offspring = self._generate_offspring(selected_parents)

            # 步骤7: 合并种群
            combined_population = self.population + offspring

            # 步骤8: 环境选择
            # 步骤8.1: 重新非支配排序
            combined_fronts = _non_dominated_sort(combined_population)

            # 步骤8.2: 构建新一代种群
            new_population = []
            remaining = self.pop_size
            for front in combined_fronts:

                if len(front) <= remaining:
                    new_population.extend(front)
                    remaining -= len(front)
                else:
                    # 计算当前前沿的拥挤度
                    self._calculate_crowding_distance(front)
                    # 按拥挤度排序选择
                    front_sorted = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
                    new_population.extend(front_sorted[:remaining])
                    break

            self.population = new_population

            # 记录knee迭代信息
            self._update_iteration_stats(gen)

        # 选择最优Knee点
        front0 = [ind for ind in self.population if ind.rank == 0]
        if front0:
            self.best_knee = self._select_knee_point(front0)

        # 最终非支配排序
        self.fronts = _non_dominated_sort(self.population)

    def _update_iteration_stats(self, gen: int) -> None:
        """记录迭代统计信息"""
        front0 = [ind for ind in self.population if ind.rank == 0]
        if front0:
            # 计算每个目标的最小值
            min_makespan = min(ind.objectives[0] for ind in front0)
            min_robustness = min(ind.objectives[1] for ind in front0)
            self.history_best_points.append({
                "generation": gen,
                "makespan": min_makespan,
                "robustness": -min_robustness,
                "front_size": len(front0)
            })

    def _calculate_crowding_distance(self, front: List[Individual]) -> None:
        """改进的拥挤度计算（直接修改个体属性）"""

        num_objs = len(front[0].objectives)

        # 初始化拥挤度
        for ind in front:
            ind.crowding_distance = 0.0

        for obj_idx in range(num_objs):
            # 按目标值排序
            front_sorted = sorted(front, key=lambda x: x.objectives[obj_idx])
            min_obj = front_sorted[0].objectives[obj_idx]
            max_obj = front_sorted[-1].objectives[obj_idx]

            # 处理边界情况
            if max_obj - min_obj < 1e-6:
                continue

            # 设置边界个体的拥挤度
            front_sorted[0].crowding_distance = float('inf')
            front_sorted[-1].crowding_distance = float('inf')

            # 计算中间个体的拥挤度
            for i in range(1, len(front_sorted) - 1):
                delta = (front_sorted[i + 1].objectives[obj_idx] -
                         front_sorted[i - 1].objectives[obj_idx]) / (max_obj - min_obj)
                front_sorted[i].crowding_distance += delta

    def _select_parents(self, fronts: List[List[Individual]]) -> List[Individual]:
        """二元锦标赛选择"""
        parents = []
        tournament_size = 2

        while len(parents) < self.pop_size:
            # 从整个种群中选择候选者
            candidates = random.sample(self.population, tournament_size)

            # 选择规则:
            # 1. 优先低非支配层级
            # 2. 同层级选择高拥挤度
            winner = min(candidates, key=lambda x: (x.rank, -x.crowding_distance))
            parents.append(deepcopy(winner))
            # TODO: 优化选择策略
            # parents.append(Individual(self.project, winner.chromosome))

        return parents

    def _generate_offspring(self, parents: List[Individual]) -> List[Individual]:
        """交叉变异生成子代"""
        offspring = []

        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                break  # 处理奇数情况

            parent1 = parents[i]
            parent2 = parents[i + 1]

            # 交叉操作
            daughter, son = self._crossover(parent1, parent2)

            # 变异操作
            daughter.mutate()
            son.mutate()

            offspring.append(daughter)
            offspring.append(son)

        return offspring

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
            # 计算 makespan 的归一化值
            denominator_m = nadir[0] - ideal[0]
            if denominator_m == 0:
                norm_makespan = 0.0  # 所有个体的 makespan 相同，归一化为 0
            else:
                norm_makespan = (ind.objectives[0] - ideal[0]) / denominator_m

            # 计算 robustness 的归一化值
            denominator_r = nadir[1] - ideal[1]
            if denominator_r == 0:
                norm_robustness = 0.0  # 所有个体的 robustness 相同，归一化为 0
            else:
                norm_robustness = ((-ind.objectives[1]) - ideal[1]) / denominator_r

            normalized.append(norm_makespan + norm_robustness)  # MMD

        # 选择 MMD 最小的个体
        return front[normalized.index(min(normalized))]


# ------------------------------------------------------------------------------------------------
# 以下为其他算法实现
#
#
#
#
#
#
#
# ------------------------------------------------------------------------------------------------


class GurobiAlgorithm:
    """使用Gurobi求解RCPSP问题"""

    def __init__(self, program):
        self.program = program
        self.model = Model("RCPSP_Solver")
        self.project_vars = {}  # 项目开始时间变量
        self.x = {}  # 时间周期变量

        # 从配置获取全局资源限制
        self.global_res_limits = program.global_resources
        self.resource_usage = {}  # 资源使用记录

    def solve(self) -> Dict:
        """求解并返回调度方案"""
        max_makespan = sum(p.total_duration for p in self.program.projects.values())
        self._add_time_variables(max_makespan)
        self._add_dependency_constraints(max_makespan)
        self._add_resource_constraints(max_makespan)

        # === 修改目标函数 ===
        # 计算总开始时间
        total_start = quicksum(
            self.project_vars[proj.project_id]
            for proj in self.program.projects.values()
        )

        # 分层目标：优先最小化最大完成时间，其次最小化总开始时间
        self.model.ModelSense = GRB.MINIMIZE
        self.model.setObjectiveN(self.max_completion, index=0, priority=2)
        self.model.setObjectiveN(total_start, index=1, priority=1)

        # 求解模型
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            # 获取调度方案
            schedule = {}
            resource_usage_info = {}
            for proj in self.program.projects.values():
                start_time = sum(t * self.x[proj.project_id][t].X for t in range(max_makespan + 1))
                start_time = int(start_time)

                schedule[proj.project_id] = start_time

                # 记录资源使用情况
                for res in self.global_res_limits:
                    if res not in resource_usage_info:
                        resource_usage_info[res] = {}
                    resource_usage = quicksum(
                        self.x[proj.project_id][t].X * proj.global_resources_request.get(res, 0)
                        for t in range(max_makespan + 1)
                    )
                    finish_time = start_time + proj.total_duration
                    for t in range(start_time, finish_time):
                        if t not in resource_usage_info[res]:
                            resource_usage_info[res][t] = 0
                        resource_usage_info[res][t] += proj.global_resources_request.get(res, 0)

            # 整理资源使用信息
            max_resource_usage = {}
            for res, usage in resource_usage_info.items():
                max_usage = max(usage.values()) if usage else 0
                max_resource_usage[res] = max_usage

            return {
                "schedule": schedule,
                "resource_usage": max_resource_usage
            }
        else:
            raise Exception("No feasible solution found")


    def _add_time_variables(self, T):
        """添加时间周期变量"""
        for proj in self.program.projects.values():
            # 每个项目的开始时间变量
            self.project_vars[proj.project_id] = self.model.addVar(
                lb=0, ub=T, vtype=GRB.INTEGER, name=f"start_{proj.project_id}"
            )
            # 时间周期变量
            self.x[proj.project_id] = {}
            for t in range(T + 1):
                self.x[proj.project_id][t] = self.model.addVar(
                    vtype=GRB.BINARY, name=f"x_{proj.project_id}_{t}"
                )

    def _add_dependency_constraints(self, T):
        """添加前驱约束"""
        # 每个项目只能在一个时间周期启动
        for proj in self.program.projects.values():
            self.model.addConstr(
                quicksum(self.x[proj.project_id][t] for t in range(T + 1)) == 1,
                name=f"single_start_{proj.project_id}"
            )

        # 前驱约束
        for proj in self.program.projects.values():
            for pred_id in proj.predecessors:
                pred = self.program.projects[pred_id]
                self.model.addConstr(
                    quicksum(t * self.x[proj.project_id][t] for t in range(T + 1)) >=
                    quicksum(t * self.x[pred.project_id][t] for t in range(T + 1)) +
                    pred.total_duration,
                    name=f"predecessor_{pred.project_id}_{proj.project_id}"
                )

    def _add_resource_constraints(self, T):
        """添加全局资源约束"""
        number_resource = len(self.global_res_limits)
        for k, res in enumerate(self.global_res_limits):
            limit = self.global_res_limits[res]
            for t in range(T + 1):
                use_resource = 0
                for proj in self.program.projects.values():
                    duration = proj.total_duration
                    start_pred = t - duration + 1
                    if start_pred < 0:
                        start_pred = 0
                    use_resource += quicksum(
                        self.x[proj.project_id][tt] * proj.global_resources_request.get(res, 0)
                        for tt in range(start_pred, t + 1)
                    )
                self.model.addConstr(
                    use_resource <= limit,
                    name=f"resource_{res}_{t}"
                )

        # 计算最大完成时间
        max_completion = self.model.addVar(name="max_completion")
        for proj in self.program.projects.values():
            self.model.addConstr(
                quicksum(t * self.x[proj.project_id][t] for t in range(T + 1)) + proj.total_duration <=
                max_completion,
                name=f"max_completion_{proj.project_id}"
            )
        self.max_completion = max_completion

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
        self.total_epc: float = 0.0  # 总拖期惩罚成本

    def run(self) -> Dict[str, Any]:
        """执行MTPC主流程"""

        # 初始化资源分配（虚拟项目0拥有全部全局资源）
        self._initialize_alloc()

        # 生成活动序列 LIST_1（按基准开始时间、权重、项目ID排序）
        sorted_projects = self._generate_project_list()

        # 遍历项目进行资源分配
        for proj in sorted_projects:
            self._allocate_resources_for_project(proj)

        # 计算总EPC并保存结果
        self.total_epc = self._calculate_total_epc()
        # self._save_results()

        return {
            "resource_arcs": list(self.A_R),
            "total_epc": self.total_epc,
            "allocations": self.alloc
        }

    def _initialize_alloc(self) -> None:
        """初始化资源分配：虚拟项目拥有全部全局资源"""
        # 初始化虚拟项目1资源分配为全局资源容量
        self.alloc = {
            "0": {res: cap for res, cap in self.program.global_resources.items()}
        }
        # 初始化其他项目资源分配为0
        for proj in self.program.projects.values():
            self.alloc[proj.project_id] = {res: 0 for res in self.program.global_resources}

    def _generate_project_list(self) -> List[Project]:
        """生成项目序列 LIST_1（排序规则：基准开始时间↑ → 权重↓ → 项目ID↑）"""

        return sorted(
            self.program.projects.values(),
            key=lambda p: (p.start_time, -getattr(p, 'priority', 0), p.project_id)
        )


    def _allocate_resources_for_project(self, proj: Project) -> None:
        """为项目分配资源（核心逻辑）"""
        required  = proj.global_resources_request

        # 步骤1：检查资源是否充足
        avail = self._calculate_available_resources(proj)
        if all(avail[res] >= required.get(res, 0) for res in self.program.global_resources):
            # 步骤3：直接分配资源
            self._direct_allocation(proj, avail)
        else:
            # 步骤2：寻找最优资源弧
            H_j = self._find_optimal_Hj(proj)
            self.A_R.update(H_j)
            self._allocate_with_priority(proj, H_j)

    def _calculate_available_resources(self, proj: Project) -> Dict[str, int]:
        """计算可用资源"""
        avail = {res: 0 for res in self.program.global_resources}
        for pred_id in proj.predecessors:
            if pred_id in self.alloc:
                avail = {res: avail[res] + self.alloc[pred_id][res] for res in avail}
        return avail

    def _find_optimal_Hj(self, proj: Project) -> Set[Tuple[str, str]]:
        """寻找最小EPC的候选集"""
        candidates = [
            (h.project_id, proj.project_id)
            for h in self.program.projects.values()
            if h.project_id not in proj.predecessors and
               h.start_time + h.total_duration <= proj.start_time
        ]
        min_epc = float('inf')
        best_Hj = set()
        for candidate in candidates:
            H_j = {candidate}
            epc = self._calculate_single_epc(proj, H_j)
            if epc < min_epc:
                min_epc = epc
                best_Hj = H_j
        return best_Hj

    def _calculate_single_epc(self, proj: Project, H_j: Set[Tuple[str, str]]) -> float:
        """计算单个候选集的EPC"""
        epc = 0.0
        all_arcs = self.A_R.union(H_j)

        # 获取所有紧前活动（包括资源弧）
        predecessors = set(proj.predecessors).union(
            {arc[0] for arc in all_arcs if arc[1] == proj.project_id}
        )

        for pred_id in predecessors:
            pred = self.program.projects[pred_id]
            # 计算LPL(i,j)简化为项目i的工期
            lpl = pred.total_duration
            threshold = proj.start_time - pred.start_time - lpl

            # 计算对数正态分布概率
            mu = np.log(pred.total_duration)  # 假设均值等于基准工期
            sigma = 0.2  # 标准差假设为0.2
            if threshold > 0:
                pr = 1 - lognorm.cdf(threshold, s=sigma, scale=np.exp(mu))
                epc += pr  # w_j=1
        return epc

    def _direct_allocation(self, proj: Project, avail: Dict[str, int]) -> None:
        """直接分配资源（无需添加额外弧）"""
        for res in self.program.global_resources:
            self.alloc[proj.project_id][res] = avail[res]
            self.alloc["0"][res] -= avail[res]

    def _allocate_with_priority(self, proj: Project, H_j: Set[Tuple[str, str]]) -> None:
        """根据优先准则分配资源（简化实现）"""
        # 按准则排序（准则）
        for res in self.program.global_resources:
            sorted_arcs = sorted(
                H_j,
                key=lambda arc: (
                    len(self.program.projects[arc[0]].successors),
                    -self.program.projects[arc[0]].weight,  # 准则2：权重降序
                    -(self.program.projects[arc[0]].start_time +
                      self.program.projects[arc[0]].total_duration),
                    -self.alloc[arc[0]][res],
                    0 if arc[0] in proj.predecessors else 1
                )
            )

        for res in self.program.global_resources:
            demand = proj.global_resources_request.get(res, 0)
            for arc in sorted_arcs:
                h_id = arc[0]
                alloc = min(self.alloc[h_id][res], demand)
                if alloc > 0:
                    self.alloc[proj.project_id][res] += alloc
                    self.alloc[h_id][res] -= alloc
                    demand -= alloc
                if demand <= 0:
                    break

    def _calculate_total_epc(self) -> float:
        """计算全局总EPC"""
        total = 0.0
        for proj in self.program.projects.values():
            total += self._calculate_single_epc(proj, self.A_R)
        return total


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
