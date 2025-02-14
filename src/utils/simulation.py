# simulation.py
import logging
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
from tqdm import trange

from models.problem import Program, Project, Activity


class ProgramSimulator:
    """项目群仿真器：用于读取和分析项目群调度结果"""

    def __init__(self):
        self.program: Optional[Program] = None

    @staticmethod
    def _create_activity(activity_data: Dict) -> Activity:
        """从字典数据创建Activity对象"""
        activity = Activity(
            activity_id=activity_data["activity_id"],
            duration=activity_data["duration"],
            resource_request=activity_data["resource_request"],
            successors=activity_data["successors"]
        )
        activity.predecessors = activity_data["predecessors"]
        activity.start_time = activity_data.get("start_time")
        return activity

    @staticmethod
    def _create_project(project_data: Dict) -> Project:
        """从字典数据创建Project对象"""
        # 创建基本项目对象
        project = Project(
            project_id=project_data["project_id"],
            local_resources=project_data["local_resources"],
            successors=project_data["successors"],
            predecessors=project_data["predecessors"]
        )

        # 添加基准调度信息
        project.start_time = project_data["start_time"]
        project.weight = project_data["weight"]
        project.total_duration = project_data.get("total_duration")

        # 资源需求不为none时，添加资源需求
        if project_data.get("global_resources_request") is not None:
            project.global_resources_request = project_data.get("global_resources_request").get("global 1")

        # 添加缓冲相关信息
        project.buffered_start_time = project_data.get("buffered_start_time")
        project.project_epc = project_data.get("project_epc")
        project.buffer_size = project_data.get("buffer_size")


        # 创建并添加活动
        for act_id, act_data in project_data["activities"].items():
            activity = ProgramSimulator._create_activity(act_data)
            project.add_activity(activity)

        return project

    @classmethod
    def load_program_from_json(cls, filepath: Path) -> Program:
        """从JSON文件加载完整的Program对象"""
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))

        # 拼接文件路径
        filepatha = project_root / filepath
        filepathc = "data/c_final_program.json"
        filepathb = "D:\File\c_final_program.json"
        filepathd = filepath
        with open(filepathd, 'r') as f:
            data = json.load(f)
        # 创建Program对象
        program = Program(
            program_id=data["program_id"],
            global_resources=data["global_resources"]
        )

        # 添加项目群基本信息
        program.total_duration = data.get("total_duration")
        program.robustness = data.get("robustness")
        program.resource_usage = data.get("resource_usage")

        # 添加资源流和缓冲信息
        program.resource_arcs = data.get("resource_arcs", [])
        program.project_buffers = data.get("project_buffers", {})
        program.total_epc = data.get("total_epc")
        program.buffered_completion_time = data.get("buffered_completion_time")

        # 创建并添加项目
        for proj_id, proj_data in data["projects"].items():
            project = cls._create_project(proj_data)
            program.add_project(project)

        return program

    def load_result(self, filepath: Path):
        """加载调度结果"""
        self.program = self.load_program_from_json(filepath)


@dataclass
class SimulationStats:
    """仿真统计结果"""
    cpsc: float = 0.0  # 不稳定惩罚成本
    pct: float = 0.0   # 平均完成时间
    tpct: float = 0.0  # 按时完工率

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "CPSC": self.cpsc,
            "PCT": self.pct,
            "TPCT": self.tpct
        }


class SimulationRunner:
    """项目群蒙特卡洛仿真执行器"""

    def __init__(self, program: Program):
        self.program = program
        self.n_simulations = 1000  # 仿真次数
        self.sigma_levels = [0.3, 0.6, 0.9]  # 标准差水平

        # 存储仿真结果
        self.project_stats: Dict[str, Dict[float, SimulationStats]] = {}
        self.mtpc_stats: Dict[float, SimulationStats] = {}
        self.stc_stats: Dict[float, SimulationStats] = {}

        # 存储项目仿真的详细结果
        # project_id -> sigma -> simulation_index -> duration
        self.project_durations: Dict[str, Dict[float, List[int]]] = defaultdict(lambda: defaultdict(list))

    def run(self) -> None:
        """执行完整的仿真实验"""
        # 对每个标准差水平进行仿真
        for sigma in self.sigma_levels:
            logging.info(f"Running simulations with sigma={sigma}")

            # 1. 项目层面仿真
            self._simulate_projects(sigma)

            # 2. MTPC项目群仿真（无缓冲）
            self._simulate_mtpc(sigma)

            # 3. STC项目群仿真（有缓冲）
            self._simulate_stc(sigma)

        # 保存结果
        self._save_results()

    def _simulate_projects(self, sigma: float) -> None:
        """对每个项目进行仿真"""
        for proj_id, project in self.program.projects.items():
            if proj_id not in ["1_virtual", "13_virtual"]:  # 跳过虚拟项目
                stats = self._simulate_single_project(project, sigma)
                if proj_id not in self.project_stats:
                    self.project_stats[proj_id] = {}
                self.project_stats[proj_id][sigma] = stats

    def _simulate_single_project(self, project: Project, sigma: float) -> SimulationStats:
        """单个项目的仿真"""
        stats = SimulationStats()
        completion_times = []
        cpsc_values = []
        on_time_count = 0

        # 获取项目的共享资源需求
        shared_resource_demand = project.global_resources_request

        # 更新项目活动的第一种资源的限量为共享资源需求
        first_resource_type = next(iter(project.local_resources.keys()))
        project.local_resources[first_resource_type] = shared_resource_demand


        # baseline_completion = project.start_time + project.total_duration

        for sim_index in range(self.n_simulations):
            # 执行单次仿真
            actual_start_times, actual_completion = self._simulate_project_execution(
                project, sigma
            )

            # 保存项目工期
            duration = actual_completion
            self.project_durations[project.project_id][sigma].append(duration)

            # 计算CPSC
            cpsc = self._calculate_cpsc(project, actual_start_times)
            cpsc_values.append(cpsc)

            # 记录完成时间
            completion_times.append(actual_completion)

            # 检查是否按时完成
            if actual_completion <= project.total_duration:
                on_time_count += 1

        # 计算统计指标
        stats.cpsc = np.mean(cpsc_values)
        stats.pct = np.mean(completion_times)
        stats.tpct = on_time_count / self.n_simulations

        return stats

    def _simulate_mtpc(self, sigma: float) -> None:
        """MTPC情况下的项目群仿真"""
        stats = SimulationStats()
        completion_times = []
        cpsc_values = []
        on_time_count = 0

        baseline_completion = max(
            proj.start_time + proj.total_duration
            for proj in self.program.projects.values()
        )

        for sim_index in range(self.n_simulations):
            # 执行项目群仿真
            project_starts, actual_completion = self._simulate_program_execution(
                self.program, sigma, sim_index, use_buffers=False
            )

            # 计算CPSC
            cpsc = self._calculate_program_cpsc(self.program, project_starts)
            cpsc_values.append(cpsc)

            # 记录完成时间
            completion_times.append(actual_completion)

            # 检查是否按时完成
            if actual_completion <= baseline_completion:
                on_time_count += 1

        # 计算统计指标
        stats.cpsc = np.mean(cpsc_values)
        stats.pct = np.mean(completion_times)
        stats.tpct = on_time_count / self.n_simulations

        self.mtpc_stats[sigma] = stats

    def _simulate_stc(self, sigma: float) -> None:
        """STC情况下的项目群仿真（考虑缓冲）"""
        stats = SimulationStats()
        completion_times = []
        cpsc_values = []
        on_time_count = 0

        baseline_completion = self.program.buffered_completion_time

        for sim_index in range(self.n_simulations):
            # 执行项目群仿真
            project_starts, actual_completion = self._simulate_program_execution(
                self.program, sigma,sim_index, use_buffers=True,
            )

            # 计算CPSC
            cpsc = self._calculate_program_cpsc(self.program, project_starts)
            cpsc_values.append(cpsc)

            # 记录完成时间
            completion_times.append(actual_completion)

            # 检查是否按时完成
            if actual_completion <= baseline_completion:
                on_time_count += 1

        # 计算统计指标
        stats.cpsc = np.mean(cpsc_values)
        stats.pct = np.mean(completion_times)
        stats.tpct = on_time_count / self.n_simulations

        self.stc_stats[sigma] = stats

    def _simulate_project_execution(
            self, project: Project, sigma: float
    ) -> Tuple[Dict[int, int], int]:
        """模拟单个项目的执行过程"""
        actual_start_times = {}  # 活动实际开始时间

        # 获取所有活动的基线开始时间
        baseline_starts = {
            act_id: act.start_time
            for act_id, act in project.activities.items()
        }

        # 按基线开始时间排序活动
        sorted_activities = sorted(
            project.activities.items(),
            key=lambda x: (x[1].start_time, x[0])
        )

        for act_id, activity in sorted_activities:
            # 确定最早可开始时间（考虑前驱活动）
            earliest_start = baseline_starts[act_id]  # 不早于基准开始时间
            for pred_id in activity.predecessors:
                if pred_id in actual_start_times:
                    pred_activity = project.activities[pred_id]
                    # 生成前驱活动的实际工期
                    pred_duration = self._generate_duration(pred_activity.duration, sigma)
                    pred_end = actual_start_times[pred_id] + pred_duration
                    earliest_start = max(earliest_start, pred_end)

            # 设置活动实际开始时间
            actual_start_times[act_id] = earliest_start

        # 计算项目实际完成时间， 实际开始时间（已经进行了工期仿真）+活动工期
        completion_time = 0
        for act_id, start_time in actual_start_times.items():
            activity = project.activities[act_id]
            completion_time = max(completion_time, start_time + activity.duration)

        return actual_start_times, completion_time

    def _simulate_program_execution(
            self, program: Program, sigma: float, sim_index: int, use_buffers: bool = False
    ) -> Tuple[Dict[str, int], int]:
        """模拟项目群的执行过程"""
        project_start_times = {}  # 项目实际开始时间
        project_durations = {}  # 项目实际工期

        # 获取基准开始时间
        for proj_id, project in program.projects.items():
            if use_buffers and project.buffered_start_time is not None:
                baseline_start = project.buffered_start_time
            else:
                baseline_start = project.start_time

            project_start_times[proj_id] = baseline_start

        # 按基准开始时间排序项目
        sorted_projects = sorted(
            program.projects.items(),
            key=lambda x: (project_start_times[x[0]], x[0])
        )

        # 模拟每个项目的执行
        for proj_id, project in sorted_projects:
            if proj_id in ["1_virtual", "13_virtual"]:
                project_durations[proj_id] = 0
                continue

            # 获取所有前驱项目（技术前驱 + 资源前驱）
            all_predecessors = set(project.predecessors)  # 技术前驱

            # 添加资源前驱
            for src, dst, _, _ in self.program.resource_arcs:
                if dst == proj_id:
                    all_predecessors.add(src)

            # 确定最早可开始时间
            earliest_start = project_start_times[proj_id]
            for pred_id in all_predecessors:
                if pred_id in project_start_times:
                    pred_end = (project_start_times[pred_id] +
                              project_durations.get(pred_id, 0))
                    earliest_start = max(earliest_start, pred_end)

            # 使用之前保存的项目仿真结果
            project_durations[proj_id] = self.project_durations[proj_id][sigma][sim_index]

            # 更新项目开始时间
            project_start_times[proj_id] = earliest_start

        # 计算项目群完成时间
        program_completion = 0
        for proj_id in program.projects:
            if proj_id not in ["1_virtual", "13_virtual"]:
                end_time = project_start_times[proj_id] + project_durations[proj_id]
                program_completion = max(program_completion, end_time)

        return project_start_times, program_completion

    def _generate_duration(self, baseline_duration: int, sigma: float) -> int:
        """生成服从对数正态分布的实际工期"""
        if baseline_duration == 0:
            return 0

        mu = np.log(baseline_duration)  # 对数正态分布的位置参数
        duration = np.random.lognormal(mu, sigma)
        return int(round(duration))

    def _calculate_cpsc(
            self, project: Project, actual_starts: Dict[int, int]
    ) -> float:
        """计算单个项目的CPSC"""
        cpsc = 0
        for act_id, actual_start in actual_starts.items():
            activity = project.activities[act_id]
            planned_start = activity.start_time
            # 所有活动权重为1
            cpsc += abs(actual_start - planned_start)
        return cpsc

    def _calculate_program_cpsc(
            self, program: Program, actual_starts: Dict[str, int]
    ) -> float:
        """计算项目群的CPSC"""
        cpsc = 0
        for proj_id, actual_start in actual_starts.items():
            if proj_id not in ["1_virtual", "13_virtual"]:
                project = program.projects[proj_id]
                planned_start = (project.buffered_start_time
                                 if project.buffered_start_time is not None
                                 else project.start_time)
                # 项目权重为1
                cpsc += abs(actual_start - planned_start)
        return cpsc

    def _save_results(self) -> None:
        """保存仿真结果到CSV文件"""
        # 准备项目层面的数据
        project_rows = []
        for proj_id, sigma_stats in self.project_stats.items():
            for sigma, stats in sigma_stats.items():
                row = {
                    "Level": "Project",
                    "ID": proj_id,
                    "Sigma": sigma,
                    **stats.to_dict()
                }
                project_rows.append(row)

        # 准备MTPC数据
        mtpc_rows = [
            {
                "Level": "MTPC",
                "ID": "Program",
                "Sigma": sigma,
                **stats.to_dict()
            }
            for sigma, stats in self.mtpc_stats.items()
        ]

        # 准备STC数据
        stc_rows = [
            {
                "Level": "STC",
                "ID": "Program",
                "Sigma": sigma,
                **stats.to_dict()
            }
            for sigma, stats in self.stc_stats.items()
        ]

        # 合并所有数据并保存
        all_rows = project_rows + mtpc_rows + stc_rows
        df = pd.DataFrame(all_rows)
        df.to_csv("simulation_results.csv", index=False)
        logging.info("Simulation results saved to simulation_results.csv")



# 运行仿真的辅助函数
def run_simulations(program_path: str) -> None:
    """运行完整的仿真实验"""
    # 使用ProgramSimulator加载项目群数据
    simulator = ProgramSimulator()
    simulator.load_result(Path(program_path))
    program = simulator.program

    runner = SimulationRunner(program)
    runner.run()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_simulations("data/c_final_program.json")