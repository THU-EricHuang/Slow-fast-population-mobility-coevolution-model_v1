import numpy as np
import torch
import os
import matplotlib.pyplot as plt

def check_convergence(ni, previous_ni, Aij, previous_Aij):
    # 计算人口分布的变化 (L2 范数)
    pop_change = torch.norm(ni - previous_ni, p=2).item()

    # 计算 weight 矩阵的余弦相似度
    weight_similarity = torch.nn.functional.cosine_similarity(Aij.flatten(), previous_Aij.flatten(), dim=0).item()

    return pop_change, weight_similarity


def plot_convergence(pop_changes, weight_similarities, dt, output_path):
    plt.figure(figsize=(12, 6))

    # 绘制人口分布的 L2 范数变化曲线
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(0, len(pop_changes), dt), pop_changes)
    plt.xlabel('Time')
    plt.ylabel('Population Change (L2 Norm)')
    plt.title('Population Distribution Convergence')

    # 绘制 weight 矩阵的余弦相似度变化曲线
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, len(weight_similarities) * dt, dt), weight_similarities)
    plt.xlabel('Time')
    plt.ylabel('Weight Matrix Cosine Similarity')
    plt.title('Weight Matrix Convergence')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'convergence_plots.png'))
    # plt.show()