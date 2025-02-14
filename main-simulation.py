from datetime import datetime
import logging
from pathlib import Path
import os

from utils.simulation import ProgramSimulator, SimulationRunner

def setup_logging(res_dir: Path) -> None:
    """配置日志系统（白色控制台输出）"""
    # 创建白色文本的ANSI转义码
    class WhiteFormatter(logging.Formatter):
        FORMAT = "\033[97m%(asctime)s - %(levelname)s - %(message)s\033[0m"  # 97=亮白色

        def format(self, record):
            formatter = logging.Formatter(self.FORMAT)
            return formatter.format(record)

    # 文件处理器（保持原色）
    file_handler = logging.FileHandler(res_dir / "simulation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # 控制台处理器（白色输出）
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(WhiteFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler]
    )

def run_simulation():
    """执行仿真主流程"""
    # 初始化结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = Path("res") / f"simulation_run_{timestamp}"
    res_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(res_dir)

    logging.info("Starting simulation experiment...")

    # 加载项目群数据
    logging.info("Loading program data...")
    simulator = ProgramSimulator()
    simulator.load_result(Path("data/c_final_program.json"))
    program = simulator.program

    # 执行仿真
    logging.info("Running simulation...")
    runner = SimulationRunner(program)
    runner.run()

    # 将结果文件移动到结果目录
    logging.info("Moving results to result directory...")
    result_file = Path("simulation_results.csv")
    if result_file.exists():
        result_file.rename(res_dir / result_file.name)

    logging.info(f"Simulation completed! Results saved to: {res_dir}")

if __name__ == "__main__":
    run_simulation()