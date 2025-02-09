# main.py
import os
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from utils.config_loader import config
from utils.RCMPSPreader import RCMPSPreader
from models.problem import Program, Project, Activity
from models.algorithm import NSGA2Algorithm
from utils.painter import ProgramVisualizer


# --------------------------
# 配置全局路径与日志
# --------------------------
def setup_directories(base_dir: str = "res") -> str:
    """创建带时间戳的结果目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(res_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(res_dir, "execution.log")),
            logging.StreamHandler()
        ]
    )
    return res_dir


# --------------------------
# 核心业务逻辑
# --------------------------
class ProjectOptimizer:
    def __init__(self, instance_root: str = "data/rcmpsp/KolischInstanzen"):
        self.instance_root = Path(instance_root)
        self.visualizer = ProgramVisualizer()

    def find_sm_files(self) -> List[Path]:
        """查找所有.sm文件"""
        sm_files = []
        for dataset in ["j30", "j60"]:
            dataset_path = self.instance_root / dataset
            if not dataset_path.exists():
                logging.warning(f"Dataset directory not found: {dataset_path}")
                continue

            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".sm"):
                        sm_files.append(Path(root) / file)
        return sm_files

    def process_sm_file(self, sm_file: Path, output_dir: Path) -> Dict:
        """处理单个项目文件"""
        # 读取并修改项目配置
        project = self._create_project_from_sm(sm_file)
        logging.info(f"Processing {sm_file.stem} with {len(project.activities)} activities")

        # 运行NSGA-II优化
        nsga_data = self.run_nsga_for_project(project, output_dir)

        # 收集结果数据
        return {
            "project_id": sm_file.stem,
            "activities": len(project.activities),
            "resources": project.local_resources,
            "optimization": nsga_data
        }

    def _create_project_from_sm(self, sm_file: Path) -> Project:
        """从SM文件创建项目对象，修改第一个资源为共享"""
        reader = RCMPSPreader()
        dummy_program = Program(program_id="dummy", global_resources={})

        # 读取原始项目
        project = reader.read_sm_file(str(sm_file), "temp", [])

        # 修改第一个资源为无限共享资源
        if project.local_resources:
            first_res = next(iter(project.local_resources.keys()))
            project.local_resources[first_res] = 999999  # 模拟无限大

        return project

    def run_nsga_for_project(self, project: Project, output_dir: Path) -> Dict:
        """执行NSGA-II优化并记录数据"""
        os.makedirs(output_dir, exist_ok=True)

        # 初始化算法
        nsga = NSGA2Algorithm(
            project=project,
            program=Program(program_id="single", global_resources={})
        )

        # 运行优化
        nsga.evolve()

        # 保存迭代数据
        self.save_iteration_data(nsga.history_knee_points, output_dir)

        # 保存帕累托前沿
        self.save_pareto_front(nsga.population, nsga.best_knee, output_dir)

        return {
            "makespan": nsga.best_knee.schedule.total_duration,
            "robustness": nsga.best_knee.schedule.robustness,
            "iterations": len(nsga.history_knee_points)
        }

    def save_iteration_data(self, history: List[Dict], output_dir: Path):
        """保存迭代过程数据到CSV"""
        csv_path = output_dir / "iteration_data.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Makespan", "Robustness"])

            for entry in history:
                writer.writerow([
                    entry["generation"],
                    entry["makespan"],
                    entry["robustness"]
                ])

    def save_pareto_front(self, population: List, knee_point: any, output_dir: Path):
        """保存帕累托前沿数据"""
        # 保存原始数据
        pareto_data = [{
            "makespan": ind.objectives[0],
            "robustness": -ind.objectives[1]
        } for ind in population]

        with open(output_dir / "pareto_front.json", 'w') as f:
            json.dump(pareto_data, f, indent=2)

        # 生成可视化图表
        self.visualizer.plot_pareto_front(
            population=population,
            knee_point=knee_point,
            save_path=str(output_dir / "pareto_front.png")
        )


# --------------------------
# 主流程
# --------------------------
def run_res_for_project():
    """主执行方法"""
    res_dir = setup_directories()
    optimizer = ProjectOptimizer()

    # 查找所有项目文件
    sm_files = optimizer.find_sm_files()
    logging.info(f"Found {len(sm_files)} SM files")

    final_report = []

    for sm_file in sm_files:
        # 为每个项目创建子目录
        project_dir = Path(res_dir) / sm_file.stem
        os.makedirs(project_dir, exist_ok=True)

        # 处理项目
        result = optimizer.process_sm_file(sm_file, project_dir)

        if result:
            final_report.append(result)
            logging.info(f"Completed {sm_file.stem}")

    # 保存总报告
    with open(Path(res_dir) / "final_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    logging.info(f"All tasks completed. Results saved to: {res_dir}")


if __name__ == "__main__":
    run_res_for_project()