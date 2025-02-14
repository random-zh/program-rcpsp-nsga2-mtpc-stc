# model/algorithm.py
import itertools
import logging
import math
import random
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
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
    """

    def __init__(self, program: Program):
        self.program = program
        # 修改资源弧格式为 (来源项目ID, 目标项目ID, 资源类型, 分配资源量)
        self.A_R: Set[Tuple[str, str, str, int]] = set()
        self.alloc: Dict[str, Dict[str, int]] = {}  # 资源分配量 {项目ID: {资源类型: 数量}}
        self.total_epc: float = 0.0  # 总拖期惩罚成本
        self.sigma = config["stc"]["sigma"]  # 对数正态分布的标准差参数

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

        # 更新Program对象的信息
        self.program.resource_arcs = list(self.A_R)
        self.program.total_epc = self.total_epc

        # 更新每个项目的EPC
        for proj_id, proj in self.program.projects.items():
            proj.project_epc = self._calculate_single_epc(proj, {(arc[0], arc[1]) for arc in self.A_R})

        # 保存结果
        # result_path = Path("results") / "mtpc_program.json"
        # self.program.save_to_json(result_path)

        return {
            "resource_arcs": list(self.A_R),
            "total_epc": self.total_epc,
            "allocations": self.alloc
        }

    def _initialize_alloc(self) -> None:
        """初始化资源分配：虚拟起始项目拥有全部全局资源，最后的虚拟项目分配为资源最大量"""
        # 初始化所有项目的资源分配为0
        for proj in self.program.projects.values():
            self.alloc[proj.project_id] = {res: 0 for res in self.program.global_resources}

        # 为项目1分配全部全局资源容量
        project_1_id = "1_virtual"  # 或实际项目1的ID
        if project_1_id in self.program.projects:
            self.alloc[project_1_id] = {res: cap for res, cap in self.program.global_resources.items()}

        # 设置最后的虚拟项目资源需求量为全局资源容量
        project_13_id = "13_virtual"
        if project_13_id in self.program.projects:
            self.program.projects[project_13_id].global_resources_request = self.program.global_resources


    def _generate_project_list(self) -> List[Project]:
        """生成项目序列 LIST_1（排序规则：基准开始时间↑ → 权重↓ → 项目ID↑）"""

        return sorted(
            self.program.projects.values(),
            key=lambda p: (p.start_time, -getattr(p, 'weight', 1), p.project_id)
        )


    def _allocate_resources_for_project(self, proj: Project) -> None:
        """为项目分配资源（核心逻辑）"""
        needs_resource_arcs = False

        # 步骤1：判断资源是否满足需求
        for res, demand in proj.global_resources_request.items():
            # 计算可用资源量
            avail_resources = self._calculate_available_resources_for_type(proj, res)
            if avail_resources < demand:
                needs_resource_arcs = True
                break

        if not needs_resource_arcs:
            # 直接从紧前项目分配资源
            self._direct_allocation(proj)
        else:
            # 寻找最优资源弧并分配
            self._allocate_with_additional_arcs(proj)

    def _calculate_available_resources_for_type(self, proj: 'Project', res_type: str) -> int:
        """计算特定类型资源的可用量"""
        return sum(
            self.alloc[pred_id][res_type]
            for pred_id in proj.predecessors
            if pred_id in self.alloc
        )

    def _direct_allocation(self, proj: 'Project') -> None:
        """直接从紧前项目分配资源，并添加资源弧"""
        sorted_preds = self._sort_predecessors(proj)

        for res, demand in proj.global_resources_request.items():
            remaining = demand
            for pred_id in sorted_preds:
                available = self.alloc[pred_id][res]
                to_allocate = min(available, remaining)

                if to_allocate > 0:
                    # 更新资源分配
                    self.alloc[proj.project_id][res] += to_allocate
                    self.alloc[pred_id][res] -= to_allocate

                    # 添加资源弧
                    self.A_R.add((pred_id, proj.project_id, res, to_allocate))

                    remaining -= to_allocate
                    if remaining <= 0:
                        break

    def _sort_predecessors(self, proj: 'Project') -> List[str]:
        """按优先级规则对紧前项目排序"""
        return sorted(
            proj.predecessors,
            key=lambda pid: (
                len(self.program.projects[pid].successors),
                -(self.program.projects[pid].start_time +
                  self.program.projects[pid].total_duration),
                -sum(self.alloc[pid].values()),
                pid
            )
        )

    def _allocate_with_additional_arcs(self, proj: 'Project') -> None:
        """寻找最优资源弧并分配资源"""
        # 寻找最优候选集
        H_j = self._find_optimal_Hj(proj)

        # 根据优先准则进行分配
        task_list = self._generate_task_list(H_j)

        # 为每种资源类型分配资源
        for res in self.program.global_resources:
            remaining = proj.global_resources_request.get(res, 0)
            for h_id, *_ in task_list:
                available = self.alloc[h_id][res]
                to_allocate = min(available, remaining)

                if to_allocate > 0:
                    # 更新资源分配
                    self.alloc[proj.project_id][res] += to_allocate
                    self.alloc[h_id][res] -= to_allocate

                    # 添加资源弧
                    self.A_R.add((h_id, proj.project_id, res, to_allocate))

                    remaining -= to_allocate
                    if remaining <= 0:
                        break

    def _find_optimal_Hj(self, proj: 'Project') -> Set[Tuple[str, str]]:
        """找最小EPC的候选集（包含能提供资源的紧前项目）"""
        min_epc = float('inf')
        best_Hj = set()
        min_size = float('inf')
        # 1. 获取能提供资源的紧前项目集合
        resource_providing_preds = set()
        for pred_id in proj.predecessors:
            for res_type in self.program.global_resources:
                if self.alloc[pred_id][res_type] > 0:
                    resource_providing_preds.add(pred_id)
                    break
        # 2. 计算各资源类型的需求
        total_demands = {
            res: proj.global_resources_request.get(res, 0)
            for res in self.program.global_resources
        }

        # 3. 先计算从紧前项目获得的资源及剩余需求
        remaining_demands = dict(total_demands)

        for pred_id in resource_providing_preds:
            for res_type, needed in remaining_demands.items():
                available = self.alloc[pred_id][res_type]
                if available > 0:
                    remaining_demands[res_type] = max(0, needed - available)

        # 4. 寻找可以提供额外资源的项目
        potential_providers = [
            h for h in self.program.projects.values()
            if (h.project_id not in proj.predecessors and  # 不是紧前项目
                h.start_time + h.total_duration <= proj.start_time and  # 满足时序约束
                any(self.alloc[h.project_id][res] > 0 for res in remaining_demands)  # 有可用资源
                )
        ]

        # 5. 生成并评估候选集合
        def can_satisfy_demands(providers: List['Project']) -> bool:
            """检查给定的项目集合是否能满足资源需求"""
            available = defaultdict(int)
            for h in providers:
                for res_type in total_demands:
                    available[res_type] += self.alloc[h.project_id][res_type]
            return all(available[res] >= total_demands[res]
                       for res in total_demands)

        # 5.1 首先评估只包含紧前项目的候选集（不可能紧靠紧前满足需求）
        if resource_providing_preds:
            # 如果达到了就有问题了
            if can_satisfy_demands([self.program.projects[pid] for pid in resource_providing_preds]):
                raise Exception("Resource demands cannot be satisfied by predecessors")

        # 5.2 评估包含额外项目的候选集
        for k in range(1, len(potential_providers) + 1):
            for combo in itertools.combinations(potential_providers, k):
                # 构建包含紧前项目和当前候选项目的集合
                H_j = {(pred_id, proj.project_id)
                       for pred_id in resource_providing_preds}
                for provider in combo:
                    H_j.add((provider.project_id, proj.project_id))
                # 目前Hj = {紧前项目+k个数量的组合项目}
                # 检查是否能满足资源需求
                # 紧前项目+k个数量的组合项目
                providers = [self.program.projects[pid]
                             for pid in resource_providing_preds]
                providers.extend(combo)

                if can_satisfy_demands(providers):

                    # 临时添加H_j中的资源弧到self.A_R
                    temp_arcs = set()
                    for src, dst in H_j:
                        # 为每个资源类型添加资源弧
                        for res_type in self.program.global_resources:
                            temp_arc = (src, dst, res_type, 0)  # 使用0作为临时分配量
                            if temp_arc not in self.A_R:
                                self.A_R.add(temp_arc)
                                temp_arcs.add(temp_arc)

                    epc = self._calculate_single_epc(proj, H_j)
                    size = len(H_j)

                    # 移除临时添加的资源弧
                    self.A_R.difference_update(temp_arcs)

                    if size < min_size or (size == min_size and epc < min_epc):
                        min_epc = epc
                        min_size = size
                        best_Hj = H_j

        return best_Hj

    def _calculate_single_epc(self, proj: 'Project', H_j: Set[Tuple[str, str]]) -> float:
        """计算单个候选集的EPC"""
        epc = 0.0
        # 获取现有资源弧的项目对
        existing_arcs = {(arc[0], arc[1]) for arc in self.A_R}
        # Hj包含了紧前可以提供资源的项目以及目前额外的提供资源的项目
        # all_arcs 代表目前所有的资源弧（包括紧前提供的）
        all_arcs = existing_arcs.union(H_j)

        # 获取j项目的所有前驱（包括资源弧）
        predecessors = self._get_all_predecessors(proj, all_arcs)

        # 计算EPC
        for pred_id in predecessors:
            # 紧前项目pred
            pred = self.program.projects[pred_id]
            lpl = self._calculate_lpl(pred_id, proj.project_id, all_arcs)
            time_gap = proj.start_time - pred.start_time - lpl

            if time_gap >= 0 and pred.total_duration > 0:
                mu = np.log(pred.total_duration)
                pr = 1 - lognorm.cdf(time_gap, s=self.sigma, scale=np.exp(mu))
                epc += pr  # w_j = 1

        return epc

    def _get_all_predecessors(self, proj: 'Project', project_arcs: Set[Tuple[str, str]]) -> Set[str]:
        """获取所有前驱项目"""
        all_preds = set()
        to_process = list(proj.predecessors)

        # 首次添加资源依赖前驱
        for src, dst in project_arcs:
            if dst == proj.project_id:
                to_process.append(src)

        while to_process:
            current = to_process.pop()
            if current not in all_preds:
                all_preds.add(current)
                if current in self.program.projects:
                    to_process.extend(self.program.projects[current].predecessors)
                # 添加资源依赖前驱
                for src, dst in project_arcs:
                    if dst == current:
                        to_process.append(src)

        return all_preds

    def _calculate_lpl(self, start_id: str, end_id: str, project_arcs: Set[Tuple[str, str]]) -> int:
        """
        计算项目间最长路径长度（路径上中间项目工期之和）

        Args:
            start_id: 起始项目ID
            end_id: 终止项目ID
            project_arcs: 资源依赖关系集合

        Returns:
            int: 最长路径上中间项目（不包括起始和终止项目）工期之和
        """

        # 构建项目网络图
        network = defaultdict(dict)

        # 添加资源弧
        for src, dst in project_arcs:
            network[src][dst] = 0  # 资源弧的权重为0。计算工期的时候，可以不考虑资源弧

        # 添加项目依赖边
        for pid, proj in self.program.projects.items():
            for succ_id in proj.successors:
                network[pid][succ_id] = 0  # 边的权重初始化为0

        # 记录每个项目的工期
        durations = {
            pid: proj.total_duration
            for pid, proj in self.program.projects.items()
        }

        def dfs(current: str, target: str, visited: Set[str]) -> Tuple[int, List[str]]:
            """
            深度优先搜索，返回最长路径长度和路径

            Returns:
                Tuple[int, List[str]]: (路径长度, 路径上的项目ID列表)
            """
            if current == target:
                return 0, [current]

            if current in visited:
                return float('-inf'), []

            visited.add(current)
            max_length = float('-inf')
            best_path = []

            # 遍历所有后继节点
            for next_node in network[current]:
                if next_node not in visited:
                    sub_length, sub_path = dfs(next_node, target, visited)
                    if sub_length != float('-inf'):
                        # 如果是起始项目，不加入其工期
                        if next_node != target:  # 只加入中间项目的工期
                            current_length = sub_length + durations.get(next_node, 0)
                        else:
                            current_length = sub_length

                        if current_length > max_length:
                            max_length = current_length
                            best_path = [current] + sub_path

            visited.remove(current)
            return max_length, best_path

        # 计算最长路径
        max_duration, path = dfs(start_id, end_id, set())

        if max_duration == float('-inf'):
            return 0

        # 输出详细的路径信息用于调试
        path_info = [
            f"{pid}({self.program.projects[pid].total_duration})"
            for pid in path
            if pid in self.program.projects
        ]

        middle_projects_duration = sum(
            self.program.projects[pid].total_duration
            for pid in path[1:-1]  # 只取中间项目
            if pid in self.program.projects
        )

        logging.debug(f"Path from {start_id} to {end_id}: {' -> '.join(path_info)}, "
                      f"middle projects duration: {middle_projects_duration}")

        return max_duration

    def _generate_task_list(self, H_j: Set[Tuple[str, str]]) -> List[Tuple[str, ...]]:
        """生成排序后的任务列表"""
        task_list = []
        for h_id, _ in H_j:
            h_proj = self.program.projects[h_id]
            task_list.append((
                h_id,
                len(h_proj.successors),
                h_proj.start_time + h_proj.total_duration,
                sum(self.alloc[h_id].values())
            ))

        return sorted(task_list, key=lambda x: (
            -x[1],  # 后继数量降序
            x[2],   # 完成时间升序
            -x[3],  # 可用资源量降序
            x[0]    # 项目ID升序
        ))

    def _calculate_total_epc(self) -> float:
        """计算全局总EPC"""
        total = 0.0
        # 从self.A_R中提取所有资源弧对
        existing_arcs = {(arc[0], arc[1]) for arc in self.A_R}

        logging.debug(f"MTPC Existing resource arcs: {existing_arcs}")

        for proj in self.program.projects.values():
            # 使用完整的资源弧集合计算EPC
            total += self._calculate_single_epc(proj, existing_arcs)
            logging.debug(f"MTPC EPC for project {proj.project_id}: {self._calculate_single_epc(proj, existing_arcs)}")
        return total


# class ArtiguesAlgorithm:
#     """
#     Artigues算法（项目群层面）：生成可行性资源流网络（不优化鲁棒性）
#     """
#
#     def __init__(self, program: Program):
#         self.program = program
#         self.A_R = set()  # 资源弧集合 (from_id, to_id, res_type, amount)
#         self.alloc = {}  # 资源分配 {项目ID: {资源类型: 数量}}
#         self.remaining_resource = {}  # 记录每个活动剩余的资源需求
#         self.total_epc = 0.0
#         self.sigma = 0.2  # 对数正态分布的标准差参数
#
#     def run(self) -> Dict[str, Any]:
#         """执行算法主流程"""
#         # 初始化资源分配
#         self._initialize_allocation()
#
#         # 获取所有时间点（按升序）
#         time_points = self._get_time_points()
#
#         # 按时间顺序处理每个时间点
#         for t in time_points:
#             # 获取在时间t开始的活动
#             starting_activities = self._get_activities_starting_at(t)
#
#             # 为每个开始的活动分配资源
#             for j in starting_activities:
#                 # 处理每种资源类型
#                 for k in self.program.global_resources:
#                     self._allocate_resource(j, k, t)
#
#         # 计算总的EPC
#         self._calculate_total_epc()
#
#         return {
#             "resource_arcs": list(self.A_R),
#             "total_epc": self.total_epc,
#             "allocations": self.alloc
#         }
#
#     def _initialize_allocation(self):
#         """初始化资源分配"""
#         # 初始化所有项目的资源分配为0
#         for proj in self.program.projects.values():
#             self.alloc[proj.project_id] = {
#                 res: 0 for res in self.program.global_resources
#             }
#
#         # 初始化每个活动的资源需求
#         self.remaining_resource = {}
#         for proj in self.program.projects.values():
#             for act_id, act in proj.activities.items():
#                 for k in self.program.global_resources:
#                     self.remaining_resource[(act_id, k)] = act.resource_request.get(k, 0)
#
#     def _calculate_total_epc(self):
#         """计算总的EPC值"""
#         self.total_epc = 0.0
#         for proj in self.program.projects.values():
#             self.total_epc += self._calculate_project_epc(proj)
#
#     def _calculate_project_epc(self, proj: Project) -> float:
#         """计算单个项目的EPC"""
#         epc = 0.0
#
#         # 获取所有前驱（包括资源前驱）
#         predecessors = self._get_all_predecessors(proj)
#
#         for pred_id in predecessors:
#             if pred_id not in self.program.projects:
#                 continue
#
#             pred = self.program.projects[pred_id]
#             lpl = self._calculate_lpl(pred_id, proj.project_id)
#             time_gap = proj.start_time - pred.start_time - lpl
#
#             if time_gap > 0 and pred.total_duration > 0:
#                 mu = np.log(pred.total_duration)
#                 pr = 1 - lognorm.cdf(time_gap, s=self.sigma, scale=np.exp(mu))
#                 epc += pr
#
#         return epc
#
#     def _get_all_predecessors(self, proj: Project) -> Set[str]:
#         """获取项目的所有前驱（包括资源前驱）"""
#         predecessors = set(proj.predecessors)
#
#         # 添加资源前驱
#         for from_id, to_id, _, _ in self.A_R:
#             if to_id == proj.project_id:
#                 predecessors.add(from_id)
#
#         return predecessors
#
#     def _calculate_lpl(self, start_id: str, end_id: str) -> int:
#         """计算最长路径长度"""
#         # 构建网络图
#         network = defaultdict(dict)
#
#         # 添加项目依赖边
#         for pid, proj in self.program.projects.items():
#             for succ_id in proj.successors:
#                 network[pid][succ_id] = proj.total_duration
#
#         # 添加资源弧（权重为0）
#         for src, dst, _, _ in self.A_R:
#             if dst not in network[src]:  # 避免覆盖项目依赖边
#                 network[src][dst] = 0
#
#         # 使用DFS计算最长路径
#         def dfs(current: str, target: str, visited: Set[str]) -> int:
#             if current == target:
#                 return 0
#             if current in visited:
#                 return float('-inf')
#
#             visited.add(current)
#             max_length = float('-inf')
#
#             for next_node, weight in network[current].items():
#                 length = dfs(next_node, target, visited)
#                 if length != float('-inf'):
#                     max_length = max(max_length, length + weight)
#
#             visited.remove(current)
#             return max_length
#
#         result = dfs(start_id, end_id, set())
#         return result if result != float('-inf') else 0
#
#     def _get_time_points(self) -> List[int]:
#         """获取所有活动的开始时间点（升序）"""
#         time_points = set()
#         for proj in self.program.projects.values():
#             for act in proj.activities.values():
#                 if act.start_time is not None:
#                     time_points.add(act.start_time)
#         return sorted(list(time_points))
#
#     def _get_activities_starting_at(self, t: int) -> List[str]:
#         """获取在时间t开始的活动列表"""
#         activities = []
#         for proj in self.program.projects.values():
#             for act_id, act in proj.activities.items():
#                 if act.start_time == t:
#                     activities.append(act_id)
#         return activities
#
#     def _get_completed_activities_before(self, t: int) -> List[str]:
#         """获取在时间t之前完成的活动列表"""
#         completed = []
#         for proj in self.program.projects.values():
#             for act_id, act in proj.activities.items():
#                 if act.start_time + act.duration <= t:
#                     completed.append(act_id)
#         return completed
#
#     def _allocate_resource(self, j: str, k: str, t: int):
#         """为活动j分配类型k的资源"""
#         req_k = self.remaining_resource.get((j, k), 0)
#         if req_k <= 0:
#             return
#
#         completed_acts = self._get_completed_activities_before(t)
#         m = 0
#
#         while req_k > 0 and m < len(completed_acts):
#             provider = completed_acts[m]
#             if (provider, k) in self.remaining_resource:
#                 # 计算可分配的资源量
#                 flow_amount = min(req_k, self.remaining_resource[(provider, k)])
#                 if flow_amount > 0:
#                     # 更新资源弧和分配
#                     self.A_R.add((provider, j, k, flow_amount))
#
#                     # 更新资源分配
#                     proj_id = next(proj.project_id
#                                    for proj in self.program.projects.values()
#                                    if j in proj.activities)
#                     self.alloc[proj_id][k] += flow_amount
#
#                     # 更新剩余需求
#                     req_k -= flow_amount
#                     self.remaining_resource[(provider, k)] -= flow_amount
#             m += 1


class STCAlgorithm:
    """
    项目群层面的STC算法：在资源分配基础上插入项目间缓冲以提升鲁棒性

    核心步骤：
    1. 固定第一阶段采用MEPC优化算法构建的资源流网络，计算各项目的EPCj
    2. 根据EPC降序排列项目获得LIST3
    3. 在EPC最大的项目前插入缓冲，更新后续项目时间
    4. 检查完工期约束和EPC改进
    5. 重复优化直到无法继续改进
    """

    def __init__(self, program: Program, resource_network: Dict, max_completion_time: int):
        self.program = deepcopy(program)  # 深拷贝防止修改原始数据
        self.real_program = program
        self.resource_network = resource_network  # 来自MEPC的资源流网络
        self.max_completion_time = max_completion_time  # 项目群最大允许完工期限
        self.project_buffers = {}  # 改为 {proj_id: buffer_size}，只记录项目及其缓冲大小
        self.original_epc = 0.0  # 原始总EPC值
        self.final_epc = 0.0  # 最终总EPC值
        self.sigma = config["stc"]["sigma"]  # 对数正态分布标准差参数

    def run(self) -> Dict[str, Any]:
        """执行STC算法主流程"""
        # 步骤1：计算项目的单独EPC
        projects_epc = self._calculate_all_projects_epc()
        # 计算初始总EPC
        self.original_epc = self._calculate_total_epc()

        # 按单个项目EPC降序排列项目
        LIST3 = sorted(
            [(proj_id, epc) for proj_id, epc in projects_epc.items()],
            key=lambda x: (x[1], -self.program.projects[x[0]].weight),  # 考虑项目权重
            reverse=True
        )

        current_total_epc = self.original_epc

        while LIST3:
            # 步骤2：选择EPC最大的项目
            current_proj_id, proj_epc = LIST3[0]
            if proj_epc == 0:
                break

            # 在项目前插入单位缓冲
            buffer_inserted = self._insert_buffer(current_proj_id)
            if not buffer_inserted:
                LIST3.pop(0)
                continue

            # 步骤3：计算新计划的总EPC
            new_total_epc = self._calculate_total_epc()

            # 步骤4：检查完工期约束和总EPC改进
            if self._check_completion_time() and new_total_epc < current_total_epc:
                # 更新成功，保存当前状态
                current_total_epc = new_total_epc

                # 重新计算每个项目的EPC并更新LIST3
                projects_epc = self._calculate_all_projects_epc()
                LIST3 = sorted(
                    [(proj_id, epc) for proj_id, epc in projects_epc.items()],
                    key=lambda x: (x[1], -self.program.projects[x[0]].weight),
                    reverse=True
                )
            else:
                # 步骤5：移除缓冲
                self._remove_buffer(current_proj_id)
                LIST3.pop(0)  # 移除当前项目

        self.final_epc = current_total_epc

        # 在返回结果前更新real_program的信息
        self.real_program.resource_arcs = self.resource_network["resource_arcs"]
        self.real_program.project_buffers = self.project_buffers
        self.real_program.total_epc = self.final_epc
        self.real_program.buffered_completion_time = max(
            proj.start_time + proj.total_duration
            for proj in self.program.projects.values()
        )

        # 更新每个原始项目的信息
        for proj_id, proj in self.program.projects.items():
            real_proj = self.real_program.projects[proj_id]
            real_proj.buffered_start_time = proj.start_time
            real_proj.project_epc = self._calculate_single_epc(
                proj,
                {(arc[0], arc[1]) for arc in self.resource_network["resource_arcs"]}
            )
            real_proj.buffer_size = self.project_buffers.get(proj_id, 0)

        return {
            "program": self.real_program.to_dict(),
            "project_buffers": self.project_buffers,
            "original_epc": self.original_epc,
            "final_epc": self.final_epc,
            "improved_percentage": ((self.original_epc - self.final_epc) / self.original_epc * 100
                                   if self.original_epc > 0 else 0)
        }

    def _calculate_all_projects_epc(self) -> Dict[str, float]:
        """计算所有项目的EPC值"""
        total_epc = {}
        # 从resource_network中提取所有资源弧对
        existing_arcs = {(arc[0], arc[1]) for arc in self.resource_network["resource_arcs"]}

        logging.debug(f"STC Existing resource arcs: {existing_arcs}")

        for proj_id, proj in self.program.projects.items():
            total_epc[proj_id] = self._calculate_single_epc(proj, existing_arcs)
            logging.debug(f"STC EPC for project {proj_id}: {total_epc[proj_id]}")
        return total_epc

    def _calculate_single_epc(self, proj: Project, existing_arcs: Set[Tuple[str, str]]) -> float:
        """计算单个项目的EPC值"""
        epc = 0.0

        # 获取所有前驱项目（包括资源弧）
        predecessors = self._get_all_predecessors(proj, existing_arcs)

        for pred_id in predecessors:
            pred = self.program.projects.get(pred_id)
            if not pred:
                raise ValueError(f"Predecessor project {pred_id} not found")

            lpl = self._calculate_lpl(pred_id, proj.project_id, existing_arcs)
            time_gap = proj.start_time - pred.start_time - lpl

            if time_gap > 0 and pred.total_duration > 0:
                mu = np.log(pred.total_duration)
                pr = 1 - lognorm.cdf(time_gap, s=self.sigma, scale=np.exp(mu))
                epc += pr * proj.weight  # 考虑项目权重
                logging.debug(f"Sigle EPC for project {pred_id} from {proj.project_id}: {pr}")

        return epc

    def _get_all_predecessors(self, proj: Project, project_arcs: Set[Tuple[str, str]]) -> Set[str]:
        """获取所有前驱项目（包括资源前驱）"""
        all_preds = set()
        to_process = list(proj.predecessors)

        # 添加资源依赖前驱
        for src, dst in project_arcs:
            if dst == proj.project_id:
                to_process.append(src)

        while to_process:
            current = to_process.pop()
            if current not in all_preds:
                all_preds.add(current)
                if current in self.program.projects:
                    to_process.extend(self.program.projects[current].predecessors)
                    # 添加资源依赖前驱
                    for src, dst in project_arcs:
                        if dst == current:
                            to_process.append(src)

        return all_preds

    def _calculate_lpl(self, start_id: str, end_id: str, project_arcs: Set[Tuple[str, str]]) -> int:
        """
        计算项目间最长路径长度（路径上中间项目工期之和）

        Args:
            start_id: 起始项目ID
            end_id: 终止项目ID
            project_arcs: 资源依赖关系集合

        Returns:
            int: 最长路径上中间项目（不包括起始和终止项目）工期之和
        """
        # 构建项目网络图
        network = defaultdict(dict)

        # 添加资源弧
        for src, dst in project_arcs:
            network[src][dst] = 0  # 资源弧的权重为0。计算工期的时候，可以不考虑资源弧

        # 添加项目依赖边
        for pid, proj in self.program.projects.items():
            for succ_id in proj.successors:
                network[pid][succ_id] = 0  # 边的权重初始化为0

        # 记录每个项目的工期
        durations = {
            pid: proj.total_duration
            for pid, proj in self.program.projects.items()
        }

        def dfs(current: str, target: str, visited: Set[str]) -> Tuple[int, List[str]]:
            """
            深度优先搜索，返回最长路径长度和路径

            Returns:
                Tuple[int, List[str]]: (路径长度, 路径上的项目ID列表)
            """
            if current == target:
                return 0, [current]

            if current in visited:
                return float('-inf'), []

            visited.add(current)
            max_length = float('-inf')
            best_path = []

            # 遍历所有后继节点
            for next_node in network[current]:
                if next_node not in visited:
                    sub_length, sub_path = dfs(next_node, target, visited)
                    if sub_length != float('-inf'):
                        # 如果是起始项目，不加入其工期
                        if next_node != target:  # 只加入中间项目的工期
                            current_length = sub_length + durations.get(next_node, 0)
                        else:
                            current_length = sub_length

                        if current_length > max_length:
                            max_length = current_length
                            best_path = [current] + sub_path

            visited.remove(current)
            return max_length, best_path

        # 计算最长路径
        max_duration, path = dfs(start_id, end_id, set())

        if max_duration == float('-inf'):
            return 0

        # 输出详细的路径信息用于调试
        path_info = [
            f"{pid}({self.program.projects[pid].total_duration})"
            for pid in path
            if pid in self.program.projects
        ]

        middle_projects_duration = sum(
            self.program.projects[pid].total_duration
            for pid in path[1:-1]  # 只取中间项目
            if pid in self.program.projects
        )

        logging.debug(f"Path from {start_id} to {end_id}: {' -> '.join(path_info)}, "
                      f"middle projects duration: {middle_projects_duration}")

        return max_duration

    def _insert_buffer(self, proj_id: str) -> bool:
        """在项目前插入单位缓冲"""
        project = self.program.projects.get(proj_id)
        if not project:
            return False

        # 推迟项目开始时间
        delay = 1
        project.start_time += delay

        # 更新所有后继项目的开始时间
        self._update_successor_projects_time(proj_id, delay)

        # 记录项目缓冲
        self.project_buffers[proj_id] = self.project_buffers.get(proj_id, 0) + delay

        return True

    def _remove_buffer(self, proj_id: str):
        """移除项目的一个单位缓冲"""
        project = self.program.projects.get(proj_id)
        if not project:
            return

        # 检查项目是否有缓冲
        if proj_id in self.project_buffers and self.project_buffers[proj_id] > 0:
            # 提前项目开始时间一个单位
            project.start_time -= 1
            # 更新后继项目时间
            self._update_successor_projects_time(proj_id, -1)
            # 减少缓冲记录
            self.project_buffers[proj_id] -= 1

            # 如果缓冲已全部移除,则删除记录
            if self.project_buffers[proj_id] == 0:
                self.project_buffers.pop(proj_id)

    def _update_successor_projects_time(self, proj_id: str, delay: int):
        """更新后继项目的开始时间"""
        visited = set()
        queue = deque([(proj_id, delay)])

        # 构建完整的后继项目集合
        successor_map = defaultdict(set)

        # 初始化技术后继关系
        for project in self.program.projects.values():
            successor_map[project.project_id].update(project.successors)

        # 添加资源后继关系
        for src, dst, _, _ in self.resource_network["resource_arcs"]:
            successor_map[src].add(dst)

        while queue:
            current_id, current_delay = queue.popleft()
            if current_id in visited:
                continue

            visited.add(current_id)

            # 获取所有后继（包括技术后继和资源后继）
            successors = successor_map[current_id]

            # 更新所有后继项目
            for succ_id in successors:
                if succ_id in self.program.projects:  # 确保项目存在
                    self.program.projects[succ_id].start_time += current_delay
                    queue.append((succ_id, current_delay))

    def _check_completion_time(self) -> bool:
        """检查是否满足最大完工期限约束"""
        max_completion = max(
            proj.start_time + proj.total_duration
            for proj in self.program.projects.values()
        )
        return max_completion <= self.max_completion_time

    def _calculate_total_epc(self) -> float:
        """计算当前调度方案的总EPC"""
        return sum(self._calculate_all_projects_epc().values())