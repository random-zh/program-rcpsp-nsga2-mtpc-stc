# utils/painter.py
from typing import List

import matplotlib.pyplot as plt

from models.algorithm import Individual


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