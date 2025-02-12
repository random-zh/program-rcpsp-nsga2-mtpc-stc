# utils/painter.py
import logging
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')  # 显式设置后端
from models.algorithm import Individual
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Tuple
from models.problem import Program, Project, Activity
import networkx as nx


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
                demand = proj.global_resources_request.get(res, 0)
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
                label=f"{proj.project_id} (R: {proj.global_resources_request})",
                alpha=0.7
            )

        # 标注资源需求
        for idx, proj in enumerate(projects):
            ax.text(
                x=proj.start_time + 0.1,
                y=idx - 0.2,
                s=f"{proj.global_resources_request}",
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

    @staticmethod
    def plot_resource_network(program: Program, resource_arcs: set, save_path: str):
        """绘制资源流网络图"""
        plt.figure(figsize=(12, 8))

        # 绘制项目节点
        projects = sorted(program.projects.values(), key=lambda p: p.start_time)
        for i, proj in enumerate(projects):
            plt.scatter(proj.start_time, i, s=200, label=f"Proj {proj.project_id}")

        # 绘制资源弧
        for arc in resource_arcs:
            src = next(p for p in projects if p.project_id == arc[0])
            dest = next(p for p in projects if p.project_id == arc[1])
            plt.plot([src.start_time + src.total_duration, dest.start_time],
                     [projects.index(src), projects.index(dest)],
                     'gray', alpha=0.5)

        plt.xlabel("Time")
        plt.ylabel("Project Sequence")
        plt.title("Resource Flow Network")
        plt.legend(loc='upper right')
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_gantt_comparison(original: Program, buffered: Program, save_path: str):
        """绘制缓冲前后甘特图对比"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 原始计划
        for proj in original.projects.values():
            ax1.barh(proj.project_id, proj.total_duration,
                     left=proj.start_time, alpha=0.6)
        ax1.set_title("Original Schedule")

        # 缓冲后计划
        for proj in buffered.projects.values():
            ax2.barh(proj.project_id, proj.total_duration,
                     left=proj.start_time, alpha=0.6)
            # 标记缓冲
            if proj.project_id in buffered.buffers_added:
                ax2.text(proj.start_time, proj.project_id, "Buffer",
                         color='red', va='center')
        ax2.set_title("Schedule with Buffers")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_network(
            program: 'Program',
            save_path: str,
            title: str = "Project Network",
            resource_arcs: Set[Tuple[str, str]] = None,
            node_size: int = 2000,
            figsize: Tuple[int, int] = (12, 8)
    ):
        """绘制单代号网络图"""
        plt.figure(figsize=figsize)
        G = nx.DiGraph()

        # 添加节点
        for proj in program.projects.values():
            G.add_node(
                proj.project_id,
                label=f"{proj.project_id}\nStart: {proj.start_time}\nDur: {proj.total_duration}"
            )

        # 添加依赖关系边
        edge_colors = []
        for proj in program.projects.values():
            # 添加逻辑依赖边（黑色）
            for succ_id in proj.successors:
                if succ_id in program.projects:
                    G.add_edge(proj.project_id, succ_id, color='black', style='solid')
                    edge_colors.append('black')

        # 布局算法
        pos = nx.spring_layout(G, seed=42)

        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=node_size,
            alpha=0.8
        )

        # 绘制标签
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            verticalalignment='center'
        )

        # 绘制边
        edge_styles = [G[u][v]['style'] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            style=edge_styles,
            arrows=True
        )

        # 添加图例
        plt.plot([], [], color='black', linestyle='solid', label='Logical Dependency')
        if resource_arcs:
            plt.plot([], [], color='red', linestyle='dashed', label='Resource Arc')
        plt.legend(loc='upper right')

        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()