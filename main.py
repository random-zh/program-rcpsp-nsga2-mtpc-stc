# main.py
import os
import json
import logging
from datetime import datetime

from utils.config_loader import config
from utils.RCMPSPreader import RCMPSPreader
from models.algorithm import (
    NSGA2Algorithm,
    GurobiAlgorithm,
    MTPCAlgorithm,
    ArtiguesAlgorithm,
    STCAlgorithm
)
from utils.simulation import SimulationRunner
from utils.painter import ProgramVisualizer

# --------------------------
# 配置全局路径与日志
# --------------------------

def setup_directories(base_dir: str = "res") -> str:
    """创建结果目录并配置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(res_dir, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        filename=os.path.join(res_dir, "execution.log"),
        level=logging.getLevelName(config["project"]["log_level"]),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(res_dir, "execution.log")),
            logging.StreamHandler()
        ]
    )
    return res_dir


# --------------------------
# 数据存储模块
# --------------------------
def save_results(data: dict, path: str) -> None:
    """保存数据为JSON"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    logging.info("Saved results to: %s", path)


# --------------------------
# 主流程
# --------------------------
def main():
    try:
        # 初始化路径和日志
        res_dir = setup_directories()
        visualizer = ProgramVisualizer()
        logging.info("Initialized result directory: %s", res_dir)

        # --------------------------
        # 阶段1：数据读取与预处理
        # --------------------------
        logging.info("[Phase 1] Loading program data")
        reader = RCMPSPreader()
        program = reader.read_program_xml(config["project"]["input_file"])
        logging.info("Loaded program with %d projects", len(program.projects))

        # --------------------------
        # 阶段2：单项目优化 (NSGA-II)
        # --------------------------
        logging.info("[Phase 2] Single-project optimization with NSGA-II")
        for proj_id, project in program.projects.items():
            if not project.activities:
                continue  # 跳过虚项目

            logging.info("Optimizing project: %s", proj_id)
            nsga = NSGA2Algorithm(project, program)
            nsga.evolve()

            # 保存帕累托前沿可视化
            visualizer.plot_pareto_front(
                population=nsga.population,
                knee_point=nsga.best_knee,
                save_path=os.path.join(res_dir, f"pareto_{proj_id}.png")
            )

            # 更新项目数据
            project.total_duration = nsga.best_knee.schedule.total_duration
            project.shared_resources_request = nsga.best_knee.project.shared_resources_request
            logging.info("Project %s optimized. Makespan: %d", proj_id, project.total_duration)

        # --------------------------
        # 阶段3：项目群级调度 (Gurobi)
        # --------------------------
        logging.info("[Phase 3] Program-level scheduling with Gurobi")
        gurobi = GurobiAlgorithm(program)
        baseline_schedule = gurobi.solve()
        save_results(baseline_schedule, os.path.join(res_dir, "baseline_schedule.json"))

        # 可视化基准调度
        visualizer.plot_gantt(
            program=program,
            save_path=os.path.join(res_dir, "baseline_gantt.png")
        )

        # --------------------------
        # 阶段4：资源分配 (MTPC vs Artigues)
        # --------------------------
        logging.info("[Phase 4] Resource allocation")

        # MTPC 算法
        mtpc_dir = os.path.join(res_dir, "MTPC")
        os.makedirs(mtpc_dir, exist_ok=True)

        mtpc = MTPCAlgorithm(program)
        mtpc_result = mtpc.run()
        save_results(mtpc_result, os.path.join(mtpc_dir, "mtpc_result.json"))
        visualizer.plot_resource_allocation(
            program=program,
            save_path=os.path.join(mtpc_dir, "mtpc_resource_allocation.png")
        )

        # Artigues 算法（对比）
        artigues_dir = os.path.join(res_dir, "Artigues")
        os.makedirs(artigues_dir, exist_ok=True)

        artigues = ArtiguesAlgorithm(program)
        artigues_result = artigues.run()
        save_results(artigues_result, os.path.join(artigues_dir, "artigues_result.json"))
        visualizer.plot_resource_allocation(
            program=program,
            save_path=os.path.join(artigues_dir, "artigues_resource_allocation.png")
        )

        # --------------------------
        # 阶段5：鲁棒性优化 (STC)
        # --------------------------
        logging.info("[Phase 5] Robustness optimization with STC")
        stc_dir = os.path.join(res_dir, "STC")
        os.makedirs(stc_dir, exist_ok=True)

        stc = STCAlgorithm(program)
        stc_result = stc.run(max_iter=100)
        save_results(stc_result, os.path.join(stc_dir, "stc_result.json"))

        # 可视化带缓冲的调度
        visualizer.plot_gantt(
            program=program,
            save_path=os.path.join(stc_dir, "stc_gantt.png")
        )

        # --------------------------
        # 阶段6：仿真分析
        # --------------------------
        logging.info("[Phase 6] Simulation analysis")
        sim_dir = os.path.join(res_dir, "Simulation")
        os.makedirs(sim_dir, exist_ok=True)

        runner = SimulationRunner(
            program=program,
            output_dir=sim_dir
        )
        sigmas = [0.2, 0.5, 0.8]  # 扰动参数
        runner.run(sigmas=sigmas, n_simulations=1000)

        # --------------------------
        # 生成最终报告
        # --------------------------
        report = {
            "program_id": program.program_id,
            "global_resources": program.global_resources,
            "num_projects": len(program.projects),
            "simulation_sigmas": sigmas,
            "result_dirs": {
                "baseline": res_dir,
                "MTPC": mtpc_dir,
                "Artigues": artigues_dir,
                "STC": stc_dir,
                "Simulation": sim_dir
            }
        }
        save_results(report, os.path.join(res_dir, "final_report.json"))
        logging.info("All processes completed successfully!")

    except Exception as e:
        logging.error("Main process failed: %s", str(e), exc_info=True)
        raise


if __name__ == "__main__":
    main()