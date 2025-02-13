# main-program.py
import json
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from utils.Projectreader import ProjectReader
from utils.painter import ProgramVisualizer
from models.algorithm import GurobiAlgorithm, MTPCAlgorithm, STCAlgorithm


# def setup_logging(res_dir: Path) -> None:
#     """配置日志系统"""
#     # 通过 handlers 来指定处理器
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(res_dir / "execution.log"),  # 文件处理器
#             logging.StreamHandler()  # 控制台处理器
#         ]
#     )
def setup_logging(res_dir: Path) -> None:
    """配置日志系统（白色控制台输出）"""

    # 创建白色文本的ANSI转义码
    class WhiteFormatter(logging.Formatter):
        FORMAT = "\033[97m%(asctime)s - %(levelname)s - %(message)s\033[0m"  # 97=亮白色

        def format(self, record):
            formatter = logging.Formatter(self.FORMAT)
            return formatter.format(record)

    # 文件处理器（保持原色）
    file_handler = logging.FileHandler(res_dir / "execution.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # 控制台处理器（白色输出）
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(WhiteFormatter())

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[file_handler, stream_handler]
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

    # 将项目群各项目开始时间信息保存到对象
    for project in program.projects.values():
        project.start_time = gurobi_result["schedule"][project.project_id]

    # 保存项目群基本信息
    program_info = {
        "num_projects": len(program.projects),
        "program_info": program.to_dict()
    }
    with open(res_dir / "program_info.json", 'w') as f:
        json.dump(program_info, f, indent=2)

    # 可视化
    ProgramVisualizer.plot_gantt(program, gurobi_dir / "gantt.png")
    ProgramVisualizer.plot_resource_allocation(program, gurobi_dir / "resource.png")

    # 绘制初始网络图
    ProgramVisualizer.plot_network(
        program,
        save_path=res_dir / "original_network.png",
        title="Original Program Network"
    )

    # =================================================================
    # 阶段3: MTPC资源流分配
    # =================================================================
    logging.info("[Phase 3] Allocating resource flows with MTPC")
    mtpc_dir = res_dir / "mtpc"
    mtpc_dir.mkdir()

    mtpc = MTPCAlgorithm(program)
    mtpc_result = mtpc.run()
    # rras = ArtiguesAlgorithm(program)
    # rras_result = rras.run()

    # 保存资源流数据
    with open(mtpc_dir / "resource_arcs.json", 'w') as f:
        json.dump({
            "resource_arcs": list(mtpc_result["resource_arcs"]),
            "total_tpc": mtpc_result["total_epc"],
            "allocations": mtpc_result["allocations"]
        }, f, indent=2)

    # # 保存资源流数据
    # with open(mtpc_dir / "rras_resource_arcs.json", 'w') as f:
    #     json.dump({
    #         "resource_arcs": list(rras_result["resource_arcs"]),
    #         "total_tpc": rras_result["total_epc"],
    #         "allocations": rras_result["allocations"]
    #     }, f, indent=2)

    # 可视化资源流
    ProgramVisualizer.plot_resource_network(
        program,
        mtpc_result["resource_arcs"],
        str(mtpc_dir / "resource_network.png")
    )

    # =================================================================
    # 阶段4: STC缓冲分配
    # =================================================================
    logging.info("[Phase 4] Adding buffers with STC")
    stc_dir = res_dir / "stc"
    stc_dir.mkdir()

    # 计算最大允许完工期限（1.2倍基准工期）
    max_duration = max(
        sum(act.duration for act in proj.activities.values())
        for proj in program.projects.values()
    )
    max_completion_time = int(max_duration * 1.3)

    # 运行STC算法（使用MTPC的资源流结果）
    stc = STCAlgorithm(program, mtpc_result, max_completion_time)
    stc_result = stc.run()

    # 保存STC结果
    with open(stc_dir / "stc_result.json", 'w') as f:
        json.dump({
            "buffers": stc_result["buffers"],
            "original_epc": stc_result["original_epc"],
            "final_epc": stc_result["final_epc"],
            "improved_percentage": stc_result["improved_percentage"],
            "program_info": stc_result["program"]
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