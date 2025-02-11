# main-program.py
import json
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from utils.Projectreader import ProjectReader
from utils.painter import ProgramVisualizer
from models.algorithm import GurobiAlgorithm, MTPCAlgorithm, STCAlgorithm


def setup_logging(res_dir: Path) -> None:
    """配置日志系统"""
    # 通过 handlers 来指定处理器
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(res_dir / "execution.log"),  # 文件处理器
            logging.StreamHandler()  # 控制台处理器
        ]
    )


def save_schedule(schedule: dict, path: Path) -> None:
    """保存调度计划"""
    with open(path, 'w') as f:
        json.dump(schedule, f, indent=2)
    logging.info(f"Saved schedule to {path}")


def main():
    # 初始化结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = Path("res") / f"program_run_{timestamp}"
    res_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(res_dir)

    # =================================================================
    # 阶段1: 读取项目群信息
    # =================================================================
    logging.info("[Phase 1] Reading program data")
    reader = ProjectReader()
    program = reader.read_projects_from_dir("data/projects/")

    # 保存项目群基本信息
    program_info = {
        "num_projects": len(program.projects),
        "global_resources": program.global_resources,
        "project_ids": list(program.projects.keys())
    }
    with open(res_dir / "program_info.json", 'w') as f:
        json.dump(program_info, f, indent=2)

    # =================================================================
    # 阶段2: 精确求解初始调度
    # =================================================================
    logging.info("[Phase 2] Running exact scheduling")
    gurobi_dir = res_dir / "gurobi"
    gurobi_dir.mkdir()

    # 求解并保存结果
    gurobi = GurobiAlgorithm(program)
    gurobi_result = gurobi.solve()
    save_schedule(gurobi_result, gurobi_dir / "baseline_schedule.json")

    # 可视化
    ProgramVisualizer.plot_gantt(program, gurobi_dir / "gantt.png")
    ProgramVisualizer.plot_resource_allocation(program, gurobi_dir / "resource.png")

    # =================================================================
    # 阶段3: MTPC资源流分配
    # =================================================================
    logging.info("[Phase 3] Allocating resource flows with MTPC")
    mtpc_dir = res_dir / "mtpc"
    mtpc_dir.mkdir()

    mtpc = MTPCAlgorithm(program)
    mtpc_result = mtpc.run()

    # 保存资源流数据
    with open(mtpc_dir / "resource_arcs.json", 'w') as f:
        json.dump({
            "resource_arcs": list(mtpc_result["resource_arcs"]),
            "total_tpc": mtpc_result["total_tpc"]
        }, f, indent=2)

    # 可视化资源流
    ProgramVisualizer.plot_resource_network(
        program=program,
        resource_arcs=mtpc_result["resource_arcs"],
        save_path=mtpc_dir / "resource_network.png"
    )

    # =================================================================
    # 阶段4: STC缓冲分配
    # =================================================================
    logging.info("[Phase 4] Adding buffers with STC")
    stc_dir = res_dir / "stc"
    stc_dir.mkdir()

    # 设置工期限制（1.2倍基准工期）
    baseline_makespan = gurobi_result["makespan"]
    stc = STCAlgorithm(program, time_limit=1.2 * baseline_makespan)
    stc_result = stc.run()

    # 保存缓冲数据
    with open(stc_dir / "buffers.json", 'w') as f:
        json.dump({
            "buffered_activities": stc_result["buffers_added"],
            "final_makespan": stc_result["best_schedule"].total_duration,
            "robustness_gain": stc_result["final_robustness"] - program.robustness
        }, f, indent=2)

    # 可视化缓冲效果
    ProgramVisualizer.plot_gantt_comparison(
        original=program,
        buffered=stc_result["best_schedule"],
        save_path=stc_dir / "gantt_comparison.png"
    )

    logging.info(f"All phases completed! Results saved to: {res_dir}")


if __name__ == "__main__":
    main()