# main.py
import os
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

import concurrent.futures
from threading import Lock

from utils.config_loader import config
from utils.RCMPSPreader import RCMPSPreader
from models.problem import Program, Project, Activity
from models.algorithm import NSGA2Algorithm, Individual
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
        """查找所有.sm文件，从j30随机选40个，从j60随机选20个"""
        sm_files = []
        for dataset in ["j30", "j60"]:
            dataset_path = self.instance_root / dataset
            if not dataset_path.exists():
                logging.warning(f"Dataset directory not found: {dataset_path}")
                continue

            dataset_files = []
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".sm"):
                        dataset_files.append(Path(root) / file)

            if dataset == "j30":
                # 随机选择40个文件
                num_to_select = 50
            else:
                # 随机选择20个文件
                num_to_select = 30

            # 检查文件数量
            if num_to_select > len(dataset_files):
                logging.warning(f"Dataset {dataset} has only {len(dataset_files)} files. Selecting all.")
                sm_files.extend(dataset_files)
            else:
                # 随机选择文件
                selected_files = random.sample(dataset_files, num_to_select)
                sm_files.extend(selected_files)

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

    def run_nsga_for_project(self, project: Project, output_dir: Path) -> str:
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
        self.save_iteration_data(nsga.history_best_points, output_dir)

        # 保存帕累托前沿
        self.save_pareto_front(nsga.fronts, nsga.best_knee, output_dir)

        # 得到最终调度
        finnal_schedule = nsga.best_knee.schedule.to_dict()

        return finnal_schedule

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

    def save_pareto_front(self, fronts: List[List[Individual]], knee_point: Individual, output_dir: Path):
        """保存帕累托前沿数据（包含front层级信息）"""
        # 保存原始数据（包含front信息）
        pareto_data = []
        for front_idx, front in enumerate(fronts):
            for ind in front:
                pareto_data.append({
                    "makespan": ind.objectives[0],
                    "robustness": -ind.objectives[1],
                    "front": front_idx,
                    "is_knee": ind == knee_point
                })

        with open(output_dir / "pareto_front.json", 'w') as f:
            json.dump(pareto_data, f, indent=2)

        # 生成可视化图表
        self.visualizer.plot_pareto_front(
            fronts=fronts,
            knee_point=knee_point,
            save_path=str(output_dir / "pareto_front.png")
        )


# --------------------------
# 主流程
# --------------------------
def run_res_for_project():
    """主执行方法（多线程优化版）"""
    res_dir = setup_directories()
    optimizer = ProjectOptimizer()

    # 查找所有项目文件
    sm_files = optimizer.find_sm_files()
    logging.info(f"Found {len(sm_files)} SM files")

    # 共享数据结构和锁
    final_report = []
    report_lock = Lock()
    progress_lock = Lock()

    # 进度条初始化（线程安全）
    with tqdm(total=len(sm_files), desc="Processing Files") as pbar:
        def update_progress():
            with progress_lock:
                pbar.update(1)

        # 线程池执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()*2) as executor:
            futures = []
            for sm_file in sm_files:
                # 为每个任务创建独立目录
                project_dir = Path(res_dir) / sm_file.stem
                os.makedirs(project_dir, exist_ok=True)

                # 提交任务到线程池
                future = executor.submit(
                    process_single_file,
                    optimizer,
                    sm_file,
                    project_dir,
                    report_lock,
                    update_progress
                )
                futures.append(future)

            # 收集结果并处理异常
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        with report_lock:
                            final_report.append(result)
                except Exception as e:
                    logging.error(f"Task failed: {str(e)}", exc_info=True)

    # 保存总报告
    with open(Path(res_dir) / "final_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)

    logging.info(f"All tasks completed. Results saved to: {res_dir}")

def process_single_file(optimizer, sm_file, project_dir, report_lock, callback):
    """单个文件的处理逻辑（线程安全）"""
    try:
        result = optimizer.process_sm_file(sm_file, project_dir)
        if not result:
            raise ValueError(f"Empty result for {sm_file.stem}")

        # 保存独立结果文件
        json_file_path = project_dir / f"{sm_file.stem}.json"
        with report_lock:  # 确保文件写入顺序
            with open(json_file_path, 'w') as f:
                json.dump(result, f, indent=4)

        # 记录日志
        logging.info(f"Completed {sm_file.stem}")
        callback()  # 更新进度条
        return result
    except Exception as e:
        logging.error(f"Error processing {sm_file.stem}: {str(e)}")
        raise

if __name__ == "__main__":
    run_res_for_project()
