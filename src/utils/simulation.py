# simulation.py
import logging
import os
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Dict, List, Any, Optional
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
        filepathb = 'F:\\学习资料\\final_program.json'
        with open(filepathb, 'r') as f:
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
class ActivityResult(Activity):
    """单个活动的模拟结果"""
    activity_id: int
    sc_values: List[float]


@dataclass
class ProjectResult:
    """单个项目的模拟结果"""
    project_id: str
    activity_results: Dict[int, ActivityResult]
    sc_values: List[float]


@dataclass
class ProgramResult:
    """项目群的模拟结果"""
    program_sc_values: List[float]
    project_results: Dict[str, ProjectResult]


class SimulationRunner:
    """
    仿真执行器：对项目群进行蒙特卡洛仿真
    """

    def __init__(self, program: Program, n_simulations: int = 1000):
        self.program = program
        self.n_simulations = n_simulations
        self.sigma_levels = [0.3, 0.6, 0.9]
        self.results = defaultdict(list)

    def run(self) -> None:
        """执行仿真主流程"""
        for sigma in self.sigma_levels:
            logging.info(f"Running simulations with sigma={sigma}")

            # 对于每个sigma水平进行n_simulations次仿真
            project_metrics = defaultdict(list)  # 项目层面的指标
            program_metrics = defaultdict(list)  # 项目群层面的指标

            for sim in trange(self.n_simulations, desc=f"Sigma {sigma}"):
                # 执行一次完整的仿真
                sim_results = self._run_single_simulation(sigma)

                # 收集项目层面的指标
                for proj_id, metrics in sim_results["project_metrics"].items():
                    for metric_name, value in metrics.items():
                        project_metrics[f"{proj_id}_{metric_name}"].append(value)

                # 收集项目群层面的指标
                for metric_name, value in sim_results["program_metrics"].items():
                    program_metrics[metric_name].append(value)

            # 计算并保存平均指标
            self._save_simulation_results(sigma, project_metrics, program_metrics)

    def _run_single_simulation(self, sigma: float) -> Dict[str, Any]:
        """
        执行单次仿真
        Returns:
            Dict 包含项目和项目群层面的指标
        """
        # 创建深拷贝用于仿真
        sim_program = deepcopy(self.program)

        # 针对每个项目进行仿真
        project_metrics = {}
        actual_start_times = {}  # 记录实际开始时间
        actual_finish_times = {}  # 记录实际完成时间

        # 按开始时间排序项目
        sorted_projects = sorted(
            sim_program.projects.values(),
            key=lambda x: x.start_time
        )

        for proj in sorted_projects:
            # 更新项目的共享资源需求为第一个活动的资源限制
            self._update_project_resource_request(proj)

            # 执行项目仿真
            metrics = self._simulate_project(proj, sigma)
            project_metrics[proj.project_id] = metrics

            # 记录实际时间
            actual_start_times[proj.project_id] = metrics["actual_start"]
            actual_finish_times[proj.project_id] = metrics["actual_finish"]

        # 计算项目群层面的指标
        program_metrics = self._calculate_program_metrics(
            sim_program,
            actual_start_times,
            actual_finish_times
        )

        return {
            "project_metrics": project_metrics,
            "program_metrics": program_metrics
        }

    def _simulate_project(self, project: Project, sigma: float) -> Dict[str, float]:
        """
        模拟单个项目的执行

        Args:
            project: 项目对象
            sigma: 对数正态分布的标准差

        Returns:
            Dict: 包含项目层面的各项指标
        """
        actual_times = {}  # 活动实际开始时间
        planned_start = project.start_time
        planned_finish = project.start_time + project.total_duration

        # 获取拓扑排序的活动列表
        activities = self._get_sorted_activities(project)

        # 模拟每个活动的执行
        for act in activities:
            # 确定实际开始时间
            earliest_start = max([
                actual_times[pred_id]["finish"]
                for pred_id in act.predecessors
                if pred_id in actual_times
            ], default=project.start_time)

            actual_start = max(earliest_start, act.start_time)

            # 生成实际工期
            actual_duration = self._generate_actual_duration(act.duration, sigma)
            actual_finish = actual_start + actual_duration

            actual_times[act.activity_id] = {
                "start": actual_start,
                "finish": actual_finish
            }

        # 计算项目的实际开始和完成时间
        project_actual_start = min(times["start"] for times in actual_times.values())
        project_actual_finish = max(times["finish"] for times in actual_times.values())

        # 计算各项指标
        cpsc = sum(
            abs(actual_times[act_id]["start"] - project.activities[act_id].start_time)
            for act_id in actual_times
        )

        is_on_time = project_actual_finish <= planned_finish

        return {
            "actual_start": project_actual_start,
            "actual_finish": project_actual_finish,
            "cpsc": cpsc,
            "pct": project_actual_finish - planned_start,
            "tpct": 1 if is_on_time else 0
        }

    def _calculate_program_metrics(
            self,
            program: Program,
            actual_starts: Dict[str, int],
            actual_finishes: Dict[str, int]
    ) -> Dict[str, float]:
        """计算项目群层面的指标"""
        # 计算CPSC
        program_cpsc = sum(
            abs(actual_starts[proj_id] - proj.start_time)
            for proj_id, proj in program.projects.items()
        )

        # 计算完成时间
        program_actual_finish = max(actual_finishes.values())
        program_planned_finish = max(
            proj.start_time + proj.total_duration
            for proj in program.projects.values()
        )

        # 按时完工判断
        is_on_time = program_actual_finish <= program_planned_finish

        return {
            "cpsc": program_cpsc,
            "pct": program_actual_finish,
            "tpct": 1 if is_on_time else 0
        }

    def _update_project_resource_request(self, project: Project) -> None:
        """更新项目的共享资源需求为第一个活动的资源限制"""
        first_activity = min(
            project.activities.values(),
            key=lambda x: x.start_time
        )
        project.global_resources_request = first_activity.resource_request

    def _get_sorted_activities(self, project: Project) -> List[Activity]:
        """获取拓扑排序的活动列表"""
        sorted_acts = []
        visited = set()
        temp = set()

        def visit(act_id):
            if act_id in temp:
                raise ValueError("Cycle detected in project activities")
            if act_id in visited:
                return

            temp.add(act_id)
            act = project.activities[act_id]

            for pred_id in act.predecessors:
                visit(pred_id)

            temp.remove(act_id)
            visited.add(act_id)
            sorted_acts.append(act)

        for act in project.activities.values():
            if act.activity_id not in visited:
                visit(act.activity_id)

        return sorted_acts

    @staticmethod
    def _generate_actual_duration(planned_duration: int, sigma: float) -> int:
        """生成实际工期（对数正态分布）"""
        mu = np.log(planned_duration)
        actual = np.random.lognormal(mu, sigma)
        return max(1, int(round(actual)))

    def _save_simulation_results(
            self,
            sigma: float,
            project_metrics: Dict[str, List[float]],
            program_metrics: Dict[str, List[float]]
    ) -> None:
        """保存仿真结果"""
        # 计算平均值
        results = {
            "sigma": sigma,
            "simulation_count": self.n_simulations
        }

        # 项目层面指标
        for metric_key, values in project_metrics.items():
            proj_id, metric_name = metric_key.split('_')
            results[f"project_{proj_id}_{metric_name}_mean"] = np.mean(values)
            results[f"project_{proj_id}_{metric_name}_std"] = np.std(values)

        # 项目群层面指标
        for metric_name, values in program_metrics.items():
            results[f"program_{metric_name}_mean"] = np.mean(values)
            results[f"program_{metric_name}_std"] = np.std(values)

        # 保存到CSV
        results_df = pd.DataFrame([results])
        results_df.to_csv(f"simulation_results_sigma_{sigma}.csv", index=False)


# 运行仿真的辅助函数
def run_simulations(program_path: str) -> None:
    """运行完整的仿真实验"""
    # 使用ProgramSimulator加载项目群数据
    simulator = ProgramSimulator()
    simulator.load_result(Path(program_path))
    program = simulator.program

    # MTPC基准情况
    runner = SimulationRunner(program)
    runner.scenario = "mtpc"
    runner.run()

    # STC缓冲情况
    stc_program = deepcopy(program)
    for proj in stc_program.projects.values():
        if proj.buffer_size:
            proj.start_time = proj.buffered_start_time

    stc_runner = SimulationRunner(stc_program)
    stc_runner.scenario = "stc"
    stc_runner.run()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_simulations("data/final_program.json")