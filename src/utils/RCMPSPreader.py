import os
import xml.etree.ElementTree as ET
from models.problem import Program, Project, Activity


class RCMPSPreader:
    """ 读取项目群XML文件和项目SM文件，生成对象结构 """

    def __init__(self, base_path: str = "data/rcmpsp"):
        self.base_path = base_path  # rcmpsp 目录路径
        self.program = None  # 最终生成的项目群对象

    def read_program_xml(self, xml_file: str) -> Program:
        """解析program.xml文件，生成Program对象"""
        xml_path = xml_file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 提取项目群基本信息
        mp = root.find("mp")
        program_id = mp.find("name").text
        resources = [int(r.text) for r in mp.find("resources").findall("resource")]

        # 仅保留资源值>0的全局资源，并记录其编号（1-based）
        global_resources = {}
        shared_resource_indices = set()  # 存储非零资源的编号（如1,2,3,4）

        for i, res in enumerate(resources):
            if res > 0:
                res_key = f"global {i + 1}"  # 资源命名格式为"global 1", "global 2"等
                global_resources[res_key] = res
                shared_resource_indices.add(i + 1)  # 记录资源编号

        # 初始化Program对象
        self.program = Program(
            program_id=program_id,
            global_resources=global_resources
        )

        # 解析所有项目
        for proj_elem in mp.find("project-list").findall("project"):
            sm_file = proj_elem.find("filename").text.strip()
            project_id = proj_elem.find("id").text.strip()
            successors = proj_elem.find("successors").text.strip().split(",") if proj_elem.find(
                "successors").text else []

            # 解析.sm文件生成Project对象
            if sm_file != "null":
                project = self._read_sm_file(os.path.join(self.base_path, sm_file), project_id, successors)
            # 虚活动
            else:
                project = Project(
                    project_id=project_id,
                    successors=successors,
                    local_resources={}
                )
            project.project_id = project_id
            project.successors = [int(item) for item in successors]
            self.program.add_project(project)

        return self.program

    def _read_sm_file(self, sm_path: str, project_id, successors) -> Project:
        """解析单个.sm文件，生成Project对象"""
        with open(sm_path, 'r') as f:
            lines = f.readlines()

        # 提取本地资源限额
        resource_avail = None
        for i, line in enumerate(lines):
            if "RESOURCEAVAILABILITIES" in line:
                resource_avail = [int(x) for x in lines[i + 2].strip().split()]
                break
        local_resources = {f"R{j + 1}": resource_avail[j] for j in range(len(resource_avail))}

        # 初始化Project对象前，检查全局资源并覆盖本地资源
        # 覆盖共享资源的本地限额
        if self.program and self.program.shared_resource_indices:
            for res_name in list(local_resources.keys()):
                # 解析资源编号（假设本地资源命名格式为"R1", "R2"等）
                if res_name.startswith("R"):
                    try:
                        res_number = int(res_name[1:])  # 提取编号（如"R1"→1）
                    except ValueError:
                        continue  # 忽略无效资源名
                    if res_number in self.program.shared_resource_indices:
                        # 构造全局资源名（如"global 1"）
                        global_res_name = f"global {res_number}"
                        # 覆盖本地资源限额
                        local_resources[res_name] = self.program.global_resources[global_res_name]

        # 初始化Project对象
        project = Project(
            project_id=project_id,
            successors=successors,
            local_resources=local_resources
        )

        # 解析活动和依赖关系
        in_precedence = False
        in_requests = False
        activities = {}
        for line in lines:
            line = line.strip()
            if "PRECEDENCE RELATIONS" in line:
                in_precedence = True
                continue
            if "REQUESTS/DURATIONS" in line:
                in_precedence = False
                in_requests = True
                continue
            if "RESOURCEAVAILABILITIES" in line:
                break

            # 解析活动依赖
            if in_precedence and line and not line.startswith("jobnr") and not line.startswith("*"):
                parts = line.split()
                job_id = int(parts[0])
                successors = [int(s) for s in parts[3:]] if len(parts) > 3 else []
                activities[job_id] = {"successors": successors}

            # 解析活动持续时间和资源需求
            if in_requests and line and not line.startswith("jobnr") and not line.startswith("-") and not line.startswith("*"):
                parts = line.split()
                job_id = int(parts[0])
                duration = int(parts[2])
                resources = {f"R{j + 1}": int(parts[3 + j]) for j in range(4)}
                activities[job_id].update({
                    "duration": duration,
                    "resource_request": resources
                })

        # 创建Activity对象并添加到Project
        for job_id, data in activities.items():
            activity = Activity(
                activity_id=job_id,
                duration=data["duration"],
                resource_request=data["resource_request"],
                successors=data["successors"],
            )
            project.add_activity(activity)

        project.build_predecessors_from_successors() # 由于Activity对象的successors属性是动态添加的，因此需要手动更新predecessors
        return project