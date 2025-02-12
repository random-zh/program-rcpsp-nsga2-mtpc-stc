# problem.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set


class Activity(object):
    """
    活动类：表示项目中的单个任务，支持时序依赖和资源需求管理。

    属性:
        activity_id (int): 活动唯一标识
        duration (int): 活动持续时间
        resource_request (Dict[str, int]): 资源需求字典（如 {"machine": 2}）
        predecessors (List[int]): 紧前活动ID列表
        successors (List[int]): 紧后活动ID列表
        priority (int): 调度优先级（越小越优先）
    """

    def __init__(
            self,
            activity_id: int,
            duration: int,
            resource_request: Dict[str, int],
            successors: List[int]
    ):
        # === 参数类型验证 ===
        if not isinstance(activity_id, int):
            raise TypeError("活动ID必须为整数")
        if duration < 0:
            raise ValueError("持续时间不能为负数")
        if not isinstance(resource_request, dict):
            raise TypeError("资源需求必须为字典类型")
        if not isinstance(successors, list):
            raise TypeError("紧后活动必须为列表类型")

        # === 核心属性 ===
        self.activity_id = activity_id
        self.duration = duration
        self.resource_request = resource_request
        self.successors = successors

        # === 时间参数初始化为None（表示未计算） ===
        self.priority: Optional[int] = None
        self.start_time: Optional[int] = None
        self.es: Optional[int] = None
        # self.ef: Optional[int] = None
        # self.ls: Optional[int] = None
        # self.lf: Optional[int] = None
        # self.tf: Optional[int] = None
        # self.ff: Optional[int] = None
        # self.ciw: Optional[float] = None

        # === 紧前活动需通过方法动态添加 ===
        self.predecessors: List[int] = []

    def to_dict(self) -> dict:
        return {
            "activity_id": self.activity_id,
            "duration": self.duration,
            "resource_request": self.resource_request,
            "predecessors": self.predecessors,
            "successors": self.successors
        }


@dataclass  # 使用dataclass装饰器，自动添加__init__方法
class Project:
    """项目类：管理项目中的活动、资源及调度依赖关系"""

    # === 核心属性 ===
    project_id: str  # 项目id

    local_resources: Dict[str, int]  # 本地资源限额（如 {"worker": 5}）
    successors: List[int]  # 项目间的依赖关系（后继项目ID列表）
    # === 紧前活动需通过方法动态添加 ===
    predecessors: List[int] = field(default_factory=list)

    # === 动态属性（初始化后由方法填充） ===
    total_duration: Optional[int] = None  # 项目总工期（调度后更新）
    robustness: Optional[float] = None  # 项目鲁棒性（调度后更新）
    global_resources_request: Dict[str, int] = field(default_factory=dict)  # 共享资源需求（如 {"machine": 2}）
    start_time: Optional[int] = None  # 新增：项目的基准开始时间
    weight: int = 1  # 新增：项目的权重

    activities: Dict[int, Activity] = None  # 活动字典 {activity_id: Activity}

    def __post_init__(self):
        """初始化后验证数据合法性"""
        if not isinstance(self.project_id, (str, int)):
            raise TypeError("项目ID必须为字符串或整数")
        if self.activities is None:
            self.activities = {}

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "local_resources": self.local_resources,
            "successors": self.successors,
            "predecessors": self.predecessors,
            "activities": {aid: act.to_dict() for aid, act in self.activities.items()},
            "start_time": self.start_time,
            "weight": self.weight
        }

    # === 方法定义 ===

    def build_predecessors_from_successors(self) -> None:
        """
        根据所有活动的successors信息，自动填充predecessors。
        确保双向依赖关系的完整性。
        """
        # 遍历所有活动
        for current_act_id, current_act in self.activities.items():
            # 遍历当前活动的所有successors
            for succ_act_id in current_act.successors:
                # 检查successor活动是否存在
                if succ_act_id not in self.activities:
                    raise ValueError(f"活动 {succ_act_id} 不存在于项目中")
                # 获取successor活动对象
                succ_act = self.activities[succ_act_id]
                # 将当前活动添加到successor的predecessors中（避免重复）
                if current_act_id not in succ_act.predecessors:
                    succ_act.predecessors.append(current_act_id)

    def add_activity(self, activity: Activity) -> None:
        """添加活动，并确保ID唯一性"""
        if not isinstance(activity, Activity):
            raise TypeError("必须传入Activity对象")
        if activity.activity_id in self.activities:
            raise ValueError(f"活动ID {activity.activity_id} 已存在！")
        self.activities[activity.activity_id] = activity


@dataclass
class Program:
    """项目群类：管理多项目共享资源、全局调度优化"""

    # === 核心属性 ===
    program_id: str  # 项目群ID
    global_resources: Dict[str, int]  # 全局共享资源限量

    # === 动态属性（初始化后填充） ===
    projects: Dict[str, Project] = None  # 项目字典 {project_id: Project}
    total_duration: Optional[int] = None  # 项目群总工期
    robustness: Optional[float] = None  # 全局鲁棒性
    resource_usage: Dict[str, List[int]] = None  # 新增：全局资源时间轴占用

    # 新增资源弧
    unavoidable_arcs: Set[Tuple[int, int]] = field(default_factory=set)  # A_U: 不可避免资源弧
    extra_arcs: Set[Tuple[int, int]] = field(default_factory=set)        # A_E: 额外资源弧

    def __post_init__(self):
        """ 初始化后验证数据合法性 """
        if not isinstance(self.program_id, str):
            raise TypeError("项目群name必须为str")
        if self.projects is None:
            self.projects = {}

    def to_dict(self) -> dict:
        """将当前 Program 类实例转换为字典"""
        return {
            "program_id": self.program_id,
            "global_resources": self.global_resources,
            "projects": {
                project_id: project.to_dict() for project_id, project in self.projects.items()
            } if self.projects is not None else None,
            "total_duration": self.total_duration,
            "robustness": self.robustness,
            "resource_usage": self.resource_usage
        }

    # === 方法定义 ===
    def add_project(self, project: Project) -> None:
        """添加项目"""
        if not isinstance(project, Project):
            raise TypeError("必须传入Project对象")
        if project.project_id in self.projects:
            raise ValueError(f"项目ID {project.project_id} 已存在！")
        self.projects[project.project_id] = project
