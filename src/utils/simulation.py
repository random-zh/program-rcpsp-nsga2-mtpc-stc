# simulation.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ActivityResult:
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
    模拟执行器：面向对象封装模拟实验、结果分析与可视化
    功能：
    1. 基于对数正态分布生成活动实际开始时间
    2. 计算项目和项目群的SC（Schedule Compliance）
    3. 统计分析和可视化
    """

    def __init__(self, program: Any, output_dir: str = "simulation_results"):
        """
        :param program: 项目群对象（需包含项目和活动定义）
        :param output_dir: 结果输出目录
        """
        self.original_program = program
        self.output_dir = output_dir
        self.results: Dict[float, ProgramResult] = {}  # {sigma: ProgramResult}
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, sigmas: List[float], n_simulations: int = 1000) -> None:
        """执行多组模拟实验"""
        for sigma in sigmas:
            print(f"Running {n_simulations} simulations with sigma={sigma}...")
            program_result = self._run_single_sigma(sigma, n_simulations)
            self.results[sigma] = program_result
            self._save_single_result(sigma, program_result)

        # 全局分析
        self.analyze()
        self.plot_sc_distribution()

    def _run_single_sigma(self, sigma: float, n_sim: int) -> ProgramResult:
        """执行单个sigma值的模拟"""
        program_sc = []
        project_results = {}

        for proj in self.original_program.projects.values():
            project_results[proj.id] = ProjectResult(
                project_id=proj.id,
                activity_results={},
                sc_values=[]
            )

        for _ in range(n_sim):
            sim_program = deepcopy(self.original_program)
            self._simulate_activities(sim_program, sigma)

            # 计算项目级SC
            project_sc_values = {}
            for proj in sim_program.projects.values():
                sc = self._calculate_project_sc(proj)
                project_sc_values[proj.id] = sc
                project_results[proj.id].sc_values.append(sc)

            # 计算项目群级SC
            program_sc.append(np.mean(list(project_sc_values.values())))

        return ProgramResult(
            program_sc_values=program_sc,
            project_results=project_results
        )

    def _simulate_activities(self, program: Any, sigma: float) -> None:
        """为所有活动生成实际开始时间"""
        for proj in program.projects.values():
            for act in proj.activities.values():
                delay = np.random.lognormal(mean=0, sigma=sigma)
                act.actual_start = act.planned_start + delay
                act.sc = act.actual_start - act.planned_start

    def _calculate_project_sc(self, project: Any) -> float:
        """计算单个项目的SC（活动SC的平均值）"""
        return np.mean([act.sc for act in project.activities.values()])

    def analyze(self) -> Dict[str, Any]:
        """分析所有模拟结果（均值与标准差）"""
        analysis = {}
        for sigma, program_result in self.results.items():
            project_means = {
                proj_id: np.mean(proj.sc_values)
                for proj_id, proj in program_result.project_results.items()
            }
            project_stds = {
                proj_id: np.std(proj.sc_values)
                for proj_id, proj in program_result.project_results.items()
            }
            analysis[sigma] = {
                "projects": {
                    "means": project_means,
                    "stds": project_stds
                },
                "program": {
                    "mean": np.mean(program_result.program_sc_values),
                    "std": np.std(program_result.program_sc_values)
                }
            }

        # 保存分析结果
        with open(os.path.join(self.output_dir, "analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)

        return analysis

    def plot_sc_distribution(self) -> None:
        """绘制SC分布直方图（所有sigma）"""
        plt.figure(figsize=(10, 6))
        for sigma, program_result in self.results.items():
            plt.hist(
                program_result.program_sc_values,
                bins=30,
                alpha=0.5,
                label=f"σ={sigma}"
            )
        plt.xlabel("Schedule Compliance (SC)")
        plt.ylabel("Frequency")
        plt.title("SC Distribution Across Sigma Levels")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "sc_distribution.png"))
        plt.close()

    def _save_single_result(self, sigma: float, result: ProgramResult) -> None:
        """保存单个sigma值的详细结果"""
        sigma_dir = os.path.join(self.output_dir, f"sigma_{sigma}")
        os.makedirs(sigma_dir, exist_ok=True)

        # 保存项目群级结果
        with open(os.path.join(sigma_dir, "program_sc.json"), "w") as f:
            json.dump(result.program_sc_values, f)

        # 保存项目级结果
        for proj_id, proj_result in result.project_results.items():
            proj_data = {
                "sc_values": proj_result.sc_values,
                "activity_sc": {
                    act_id: act.sc_values
                    for act_id, act in proj_result.activity_results.items()
                }
            }
            with open(os.path.join(sigma_dir, f"project_{proj_id}.json"), "w") as f:
                json.dump(proj_data, f, indent=2)


# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    # 创建示例项目群（需根据实际项目结构定义）
    from ..models.problem import Program, Project, Activity

    # 示例项目群
    projects = {
        "P1": Project(
            project_id="P1",
            activities={
                1: Activity(activity_id=1, planned_start=0),
                2: Activity(activity_id=2, planned_start=5)
            }
        ),
        "P2": Project(
            project_id="P2",
            activities={
                3: Activity(activity_id=3, planned_start=0),
                4: Activity(activity_id=4, planned_start=6)
            }
        )
    }
    program = Program(program_id="Demo", projects=projects)

    # 运行模拟
    runner = SimulationRunner(program, output_dir="res/simulation")
    runner.run(sigmas=[0.3, 0.6, 0.9], n_simulations=1000)