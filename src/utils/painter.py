# utils/painter.py
from models.algorithm import Individual
import matplotlib.pyplot as plt
from typing import List, Dict
from models.problem import Program, Project, Activity


class KneeVisualizer:
    @staticmethod
    def plot_knee_progress(history_knee_points: List[dict]):
        """绘制Knee点的迭代过程图"""
        gens = [p["generation"] for p in history_knee_points]
        makespans = [p["makespan"] for p in history_knee_points]
        robustness = [p["robustness"] for p in history_knee_points]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # 工期变化图
        ax1.plot(gens, makespans, 'b-', marker='o')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Makespan', color='b')
        ax1.tick_params('y', colors='b')

        # 鲁棒性变化图
        ax2.plot(gens, robustness, 'r-', marker='s')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Robustness', color='r')
        ax2.tick_params('y', colors='r')

        plt.tight_layout()
        plt.savefig('knee_progress.png')
        plt.close()

    @staticmethod
    def plot_pareto_front(final_population: List[Individual], knee_point: Individual):
        """绘制最终帕累托前沿与Knee点"""
        makespans = [ind.objectives[0] for ind in final_population]
        robustness = [-ind.objectives[1] for ind in final_population]

        plt.figure(figsize=(8, 5))
        plt.scatter(makespans, robustness, c='gray', label='Pareto Front')
        plt.scatter(knee_point.objectives[0], -knee_point.objectives[1],
                    c='red', s=100, marker='*', label='Knee Point')
        plt.xlabel('Makespan')
        plt.ylabel('Robustness')
        plt.legend()
        plt.grid(True)
        plt.savefig('final_pareto_front.png')
        plt.close()


class ProgramVisualizer:
    """项目群可视化工具类"""

    @staticmethod
    def plot_resource_allocation(program: Program, save_path: str) -> None:
        """绘制全局资源分配热力图"""
        plt.figure(figsize=(12, 6))

        # 获取资源列表和时间范围
        resources = list(program.global_resources.keys())
        time_points = max(
            (p.start_time + p.total_duration for p in program.projects.values()),
            default=0
        ) + 1

        # 创建数据矩阵
        usage_data = []
        for res in resources:
            res_usage = [0] * time_points
            for proj in program.projects.values():
                start = proj.start_time
                end = start + proj.total_duration
                demand = proj.shared_resources_request.get(res, 0)
                for t in range(start, min(end, time_points)):
                    res_usage[t] += demand
            usage_data.append(res_usage)

        # 绘制热力图
        plt.imshow(usage_data, cmap='YlOrRd', aspect='auto',
                   extent=[0, time_points, 0, len(resources)])
        plt.yticks(range(len(resources)), resources)
        plt.colorbar(label='Resource Units')
        plt.title("Global Resource Allocation Heatmap")
        plt.xlabel("Time Period")
        plt.ylabel("Resource Type")
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_gantt(program: Program, save_path: str) -> None:
        """绘制项目群甘特图（带资源标注）"""
        fig, ax = plt.subplots(figsize=(15, 8))

        projects = sorted(
            program.projects.values(),
            key=lambda p: p.start_time
        )

        # 绘制项目条
        for idx, proj in enumerate(projects):
            ax.barh(
                y=idx,
                width=proj.total_duration,
                left=proj.start_time,
                height=0.6,
                label=f"{proj.project_id} (R: {proj.shared_resources_request})",
                alpha=0.7
            )

        # 标注资源需求
        for idx, proj in enumerate(projects):
            ax.text(
                x=proj.start_time + 0.1,
                y=idx - 0.2,
                s=f"{proj.shared_resources_request}",
                fontsize=8,
                va='top'
            )

        ax.set_yticks(range(len(projects)))
        ax.set_yticklabels([p.project_id for p in projects])
        ax.set_xlabel("Time")
        ax.set_title("Program Gantt Chart with Resource Annotations")
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_pareto_front(fronts: List[List[Individual]], knee_point: Individual, save_path: str) -> None:
        """绘制带非支配层级的帕累托前沿"""
        plt.figure(figsize=(10, 6))

        # 绘制各层前沿
        for front_idx, front in enumerate(fronts):
            makespans = [ind.objectives[0] for ind in front]
            robustness = [-ind.objectives[1] for ind in front]

            # 第一层用红色突出显示
            color = 'red' if front_idx == 0 else 'grey'
            alpha = 0.8 if front_idx == 0 else 0.4
            label = 'Pareto Front (Rank 0)' if front_idx == 0 else f'Front {front_idx}'

            plt.scatter(makespans, robustness,
                        c=color, alpha=alpha,
                        edgecolors='k', linewidths=0.5,
                        label=label)

        # 标注knee point
        plt.scatter([knee_point.objectives[0]],
                    [-knee_point.objectives[1]],
                    c='gold', s=200, marker='*',
                    edgecolor='k', linewidth=1,
                    label='Knee Point')

        plt.xlabel('Makespan', fontsize=12)
        plt.ylabel('Robustness', fontsize=12)
        plt.title('Pareto Front with Non-dominated Layers', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # 保存高清图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()