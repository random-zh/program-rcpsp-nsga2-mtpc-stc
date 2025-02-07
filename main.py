from utils.RCMPSPreader import RCMPSPreader

reader = RCMPSPreader()

program = reader.read_program_xml("data/program.xml")

# main.py

from utils.config_loader import config
from utils.RCMPSPreader import RCMPSPreader
from models.algorithm import NSGA2Algorithm


def main():
    # 读取项目数据
    reader = RCMPSPreader()
    program = reader.read_program_xml(config["project"]["input_file"])

    # 初始化算法
    algorithm = NSGA2Algorithm(program)

    # 运行优化
    algorithm.evolve()

    # 保存结果
    algorithm.save_results(config["project"]["output_dir"])


if __name__ == "__main__":
    main()
    print("Optimization finished.")