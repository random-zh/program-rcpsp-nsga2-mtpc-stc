# src/utils/ProjectReader.py
import json
import random
from pathlib import Path
from typing import Dict, List
from models.problem import Program, Project, Activity
from utils.config_loader import config


class ProjectReader:
    def __init__(self):
        self.global_resources = self._parse_global_resources()
        self.projects = []
        self.program_dic = config["program"]["progarm_dic"]
        self.filelist = config["file"]["filelist"]

    def _parse_global_resources(self) -> Dict[str, int]:
        """从配置解析全局资源"""
        return {f"global {i + 1}": cap
                for i, cap in enumerate(config["program"]["total_global_resource"])
                if cap > 0}

    def read_projects_from_dir(self, data_dir: str) -> Program:
        """从目录读取特定项目JSON文件"""
        program = Program(
            program_id="multi_project_program",
            global_resources=self.global_resources
        )

        # for json_file in Path(data_dir).glob("**/j*.json"):
        #     project = self._parse_single_project(json_file)
        #     program.add_project(project)

        # 添加虚拟项目
        self._add_virtual_projects(program)

        # 读取实际项目
        real_projects = self._load_real_projects(data_dir)

        random.shuffle(real_projects)  # 随机打乱顺序

        # 构建项目间依赖关系
        self._build_project_dependencies(program, real_projects)

        return program

    def _add_virtual_projects(self, program: Program) -> None:
        """添加虚拟项目（项目1和13）"""
        # 虚拟起始项目
        start_project = Project(
            project_id="1_virtual",
            local_resources={},
            successors=[],
            predecessors=[]
        )
        start_project.total_duration = 0
        program.add_project(start_project)

        # 虚拟终止项目
        end_project = Project(
            project_id="13_virtual",
            local_resources={},
            successors=[],
            predecessors=[]
        )
        end_project.total_duration = 0
        program.add_project(end_project)

    def _load_real_projects(self, data_dir: str) -> List[Project]:
        """加载实际项目文件"""
        real_projects = []
        data_path = Path(data_dir)

        # 只读取filelist中的文件
        for filename in self.filelist:
            json_path = data_path / filename / f"{filename}.json"
            if not json_path.exists():
                raise FileNotFoundError(f"项目文件 {filename} 不存在")

            project = self._parse_single_project(json_path)

            real_projects.append(project)

        return real_projects

    def _parse_single_project(self, json_path: Path) -> Project:
        """解析单个项目JSON文件"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 创建Project对象
        project = Project(
            project_id=data["project_id"],
            local_resources=data["resources"],
            successors=[]  # 项目间关系需额外处理
        )
        # 配置项目属性
        global_resources_value = data["optimization"]["global_resources_request"]
        # 将其包装为字典
        project.global_resources_request = {
            "global 1": global_resources_value
        }
        project.total_duration = data["optimization"]["total_duration"]
        project.robustness = data["optimization"]["robustness"]

        # 解析调度结果
        optimization_data = data["optimization"]
        priority_order = {int(k): v for k, v in optimization_data["priority_order"].items()}
        start_times = {int(k): v for k, v in optimization_data["start_times"].items()}

        # 添加活动
        activities = data["optimization"]["project"]["activities"]
        for act_id, act_data in activities.items():
            activity = Activity(
                activity_id=int(act_id),
                duration=act_data["duration"],
                resource_request=act_data["resource_request"],
                successors=act_data["successors"]
            )
            # 设置扩展属性
            activity.priority = priority_order.get(int(act_id))
            activity.start_time = start_times.get(int(act_id))
            activity.predecessors = act_data["predecessors"]  # 直接设置前置
            project.add_activity(activity)

        return project

    def _build_project_dependencies(self, program: Program, real_projects: List[Project]) -> None:
        """构建项目间依赖关系"""
        # ====================
        # 1. 参数验证
        # ====================
        if len(real_projects) != 11:
            raise ValueError(f"实际项目数量应为11个，当前为{len(real_projects)}个")

        # ====================
        # 2. 创建序号映射表
        # ====================
        # 生成program_dic中2-12的序号（共11个）
        program_ids = list(range(2, 13))

        # 创建映射关系：程序序号 -> 实际项目对象
        # 示例：{2: project_obj, 3: project_obj,...12: project_obj}
        id_project_map = {
            pid: real_projects[i]
            for i, pid in enumerate(program_ids)
        }

        # ====================
        # 3. 设置项目属性
        # ====================
        for seq_id, project in id_project_map.items():
            # 获取原始项目ID（如"j301_1"）
            original_id = project.project_id

            # 生成新项目ID（如"2_j301_1"）
            new_project_id = f"{seq_id}_{original_id}"
            project.project_id = new_project_id

            # 从program_dic获取依赖关系
            dependencies = self.program_dic.get(seq_id)
            if not dependencies:
                raise KeyError(f"program_dic中缺少序号{seq_id}的配置")

            # 转换前驱/后继序号为实际项目ID
            predecessors = [
                f"{pred}_virtual" if pred in [1, 13]
                else id_project_map[pred].project_id
                for pred in dependencies['predecessors']
            ]

            successors = [
                f"{succ}_virtual" if succ in [1, 13]
                else id_project_map[succ].project_id
                for succ in dependencies['successors']
            ]

            # 更新项目依赖关系
            project.predecessors = predecessors
            project.successors = successors

            # 将项目添加到项目群
            program.add_project(project)

        # ====================
        # 4. 设置虚拟项目依赖
        # ====================
        # 处理虚拟项目1（起始节点）
        virtual_start = program.projects.get("1_virtual")
        virtual_start.successors = [
            id_project_map[2].project_id,
            id_project_map[3].project_id
        ]

        # 处理虚拟项目13（终止节点）
        virtual_end = program.projects.get("13_virtual")
        virtual_end.predecessors = [
            id_project_map[10].project_id,
            id_project_map[11].project_id,
            id_project_map[12].project_id
        ]
