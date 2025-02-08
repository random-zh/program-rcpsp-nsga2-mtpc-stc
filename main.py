# main.py
import os
import json
import logging
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from utils.config_loader import config
from models.problem import Program, Project, Activity
from models.algorithm import MTPCAlgorithm, ArtiguesAlgorithm, STCAlgorithm
from utils.simulation import SimulationRunner


# --------------------------
# 配置全局路径与日志
# --------------------------
from utils.RCMPSPreader import RCMPSPreader


def setup_directories(base_dir="res"):
    """创建结果目录并配置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(res_dir, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        filename=os.path.join(res_dir, "execution.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return res_dir


# --------------------------
# 可视化模块
# --------------------------
def plot_resource_allocation(program: Program, save_path: str):
    """绘制全局资源分配图"""
    resources = list(program.global_resources.keys())
    time_max = max(p.start_time + p.total_duration for p in program.projects.values())

    fig, axes = plt.subplots(len(resources), 1, figsize=(10, 6))
    for i, res in enumerate(resources):
        usage = [0] * (time_max + 1)
        for proj in program.projects.values():
            for t in range(proj.start_time, proj.start_time + proj.total_duration):
                if t <= time_max:
                    usage[t] += proj.shared_resources_request.get(res, 0)
        axes[i].step(range(time_max + 1), usage, where="post", label=res)
        axes[i].set_title(f"Resource {res} Usage")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Units")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_gantt(program: Program, save_path: str):
    """绘制项目群甘特图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (proj_id, proj) in enumerate(program.projects.items()):
        ax.barh(proj_id, proj.total_duration, left=proj.start_time, alpha=0.6)
    ax.set_xlabel("Time")
    ax.set_ylabel("Projects")
    ax.set_title("Program Gantt Chart")
    plt.savefig(save_path)
    plt.close()


def plot_sc_distribution(sc_data: dict, save_path: str):
    """绘制SC分布直方图"""
    plt.figure(figsize=(10, 6))
    for sigma, data in sc_data.items():
        plt.hist(data["program_sc"], bins=30, alpha=0.5, label=f"σ={sigma}")
    plt.xlabel("Schedule Compliance (SC)")
    plt.ylabel("Frequency")
    plt.title("SC Distribution Across Sigma Levels")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# --------------------------
# 数据存储模块
# --------------------------
def save_results(data: dict, path: str):
    """保存数据为JSON"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------
# 主流程
# --------------------------
def main():
    # 读取项目数据
    reader = RCMPSPreader()
    program = reader.read_program_xml(config["project"]["input_file"])
    # 初始化路径和日志
    res_dir = setup_directories()
    logging.info("Initialized result directory: %s", res_dir)

    # --------------------------
    # 创建示例项目群
    # --------------------------
    projects = {
        "P1": Project(
            project_id="P1",
            total_duration=10,
            shared_resources_request={"R1": 2, "R2": 1},
            predecessors=[],
            start_time=0
        ),
        "P2": Project(
            project_id="P2",
            total_duration=8,
            shared_resources_request={"R1": 3, "R2": 2},
            predecessors=["P1"],
            start_time=10
        )
    }
    program = Program(
        program_id="DEMO",
        global_resources={"R1": 5, "R2": 3},
        projects=projects
    )
    logging.info("Created demo program with %d projects", len(projects))

    # --------------------------
    # 运行MTPC算法
    # --------------------------
    mtpc_dir = os.path.join(res_dir, "MTPC")
    os.makedirs(mtpc_dir, exist_ok=True)

    mtpc = MTPCAlgorithm(program)
    mtpc_result = mtpc.run()
    logging.info("MTPC algorithm completed")

    # 保存结果和图片
    save_results(mtpc_result, os.path.join(mtpc_dir, "allocations.json"))
    plot_resource_allocation(mtpc_result["best_schedule"], os.path.join(mtpc_dir, "resource_usage.png"))
    plot_gantt(mtpc_result["best_schedule"], os.path.join(mtpc_dir, "gantt.png"))

    # --------------------------
    # 运行STC算法
    # --------------------------
    stc_dir = os.path.join(res_dir, "STC")
    os.makedirs(stc_dir, exist_ok=True)

    stc = STCAlgorithm(mtpc_result["best_schedule"])
    stc_result = stc.run()
    logging.info("STC algorithm completed")

    save_results(stc_result, os.path.join(stc_dir, "buffers.json"))
    plot_gantt(stc_result["best_schedule"], os.path.join(stc_dir, "gantt_with_buffers.png"))

    # --------------------------
    # 运行仿真实验
    # --------------------------
    sim_dir = os.path.join(res_dir, "Simulation")
    os.makedirs(sim_dir, exist_ok=True)

    sigmas = [0.3, 0.6, 0.9]
    sc_data = {}
    for sigma in sigmas:
        # 使用STC优化后的计划进行仿真
        sim_program = deepcopy(stc_result["best_schedule"])
        sc_results = SimulationRunner.run(sim_program, sigma)
        sc_data[sigma] = sc_results

        # 保存每次仿真的详细结果
        sigma_dir = os.path.join(sim_dir, f"sigma_{sigma}")
        os.makedirs(sigma_dir, exist_ok=True)
        save_results(sc_results, os.path.join(sigma_dir, "sc_results.json"))

    # 绘制SC分布总图
    plot_sc_distribution(sc_data, os.path.join(sim_dir, "sc_distribution.png"))
    logging.info("Simulation completed for all sigma levels")

    # --------------------------
    # 整合最终报告
    # --------------------------
    report = {
        "parameters": {
            "sigmas": sigmas,
            "resource_capacity": program.global_resources
        },
        "result_dirs": {
            "MTPC": mtpc_dir,
            "STC": stc_dir,
            "Simulation": sim_dir
        }
    }
    save_results(report, os.path.join(res_dir, "report.json"))
    logging.info("Main process completed. Results saved to: %s", res_dir)


if __name__ == "__main__":
    main()