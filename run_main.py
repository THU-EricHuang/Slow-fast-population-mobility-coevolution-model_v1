import pandas as pd
import numpy as np
import glob
import torch
import os
import tqdm
import matplotlib.pyplot as plt
from Final_twoDistributionPlotter_ALL_clean import LogLogPDFPlotter
import time
from bayes_opt import BayesianOptimization
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from scipy import stats
from utils.check_and_plot_convergence import check_convergence, plot_convergence
import math

class NetworkDynamics:
    def __init__(self, distance_matrix, similarity_matrix, alpha, beta):
        self.distance_matrix = distance_matrix
        self.similarity_matrix = similarity_matrix
        self.alpha = alpha
        self.beta = beta

    def edge_evolving(self, ni, λ):
        distance = self.distance_matrix / 1000  # Unit: km
        similarity = self.similarity_matrix

        distance = torch.where(distance == 0, torch.tensor(1.2), distance)
        similarity = torch.where(similarity == 0, torch.tensor(1e-1), similarity)

        distance = distance ** (-self.alpha)
        similarity = similarity ** self.beta

        distance = distance / torch.max(distance)
        similarity = similarity / torch.max(similarity)
        ni_lambda = ni ** λ
        ni_lambda = ni_lambda / torch.max(ni_lambda)

        Pij = distance * similarity * (ni_lambda.T)
        Pij = Pij / Pij.sum(dim=1, keepdim=True)

        delta_Aij = ni * Pij

        return delta_Aij


class NodeDynamics:
    def __init__(self, dt=0.1):
        self.dt = dt

    def update_population(self, ni, inflow, outflow, initial_total_population):
        outflow = torch.where(ni == 0, torch.tensor(0.0), outflow)
        delta_x = inflow - outflow
        delta_x = torch.where(torch.isnan(delta_x) | torch.isinf(delta_x), torch.tensor(0.0), delta_x)

        ni = ni + delta_x
        ni = torch.where(ni < 0, torch.tensor(0.0), ni)

        total_population = ni.sum()
        ni = ni * (initial_total_population / total_population)

        return ni


def load_data(datapath, node_file, distance_file, similarity_file,
              ss_weight_file, ss_pop_file,
              init_mode="uniform",
              init_nodes=None,
              init_properations=None):
    """
    Parameters
    ----------
    init_mode : {"uniform", "concentrated", "custom"}
        - "uniform"
        - "concentrated"
        - "custom"
    init_nodes : list[int] | None
    init_props : list[float] | None
    """
    start_time = time.time()
    node_df = pd.read_csv(os.path.join(datapath, node_file))
    N         = node_df['resident_count'].sum()
    num_nodes = node_df.shape[0]

    node_df['init_pop'] = 0.0

    if init_mode == "uniform":
        node_df['init_pop'] = N / num_nodes

    elif init_mode == "concentrated":
        if not init_nodes:
            raise ValueError("init_nodes 不能为空 (concentrated).")
        node_df.loc[node_df['new_id'] == init_nodes[0], 'init_pop'] = N

    elif init_mode == "custom":
        if not (init_nodes and init_properations and len(init_nodes) == len(init_properations)):
            raise ValueError("custom 模式需同时提供等长的 init_nodes 与 init_props.")
        props = np.array(init_properations, dtype=float)
        props = props / props.sum()
        for nid, p in zip(init_nodes, props):
            node_df.loc[node_df['new_id'] == nid, 'init_pop'] = N * p
    else:
        raise ValueError("init_mode 只能取 'uniform' | 'concentrated' | 'custom'.")

    ni  = torch.tensor(node_df['init_pop'].values, dtype=torch.float32).view(-1, 1)
    Aij = ni / num_nodes
    Aij = Aij.repeat(1, num_nodes)

    distance_df = pd.read_csv(os.path.join(datapath, distance_file))
    similarity_df = pd.read_csv(os.path.join(datapath, similarity_file))

    ss_pop_df = pd.read_csv(os.path.join(datapath, ss_pop_file))
    ss_weight_df = pd.read_csv(os.path.join(datapath, ss_weight_file))

    all_ids = node_df['new_id'].unique()
    id_to_index = {id_val: index for index, id_val in enumerate(all_ids)}

    distance_df['source_idx'] = distance_df['source_id'].map(id_to_index)
    distance_df['target_idx'] = distance_df['target_id'].map(id_to_index)
    similarity_df['source_idx'] = similarity_df['source_id'].map(id_to_index)
    similarity_df['target_idx'] = similarity_df['target_id'].map(id_to_index)
    ss_weight_df['start_new_idx'] = ss_weight_df['start_new_id'].map(id_to_index)
    ss_weight_df = ss_weight_df.dropna(subset=['start_new_idx'])  # 估计是有没匹配上的
    ss_weight_df['start_new_idx'] = ss_weight_df['start_new_idx'].astype(int)
    ss_weight_df['end_new_idx'] = ss_weight_df['end_new_id'].map(id_to_index)
    ss_weight_df = ss_weight_df.dropna(subset=['end_new_idx'])
    ss_weight_df['end_new_idx'] = ss_weight_df['end_new_idx'].astype(int)

    distance_matrix = torch.zeros((num_nodes, num_nodes))
    similarity_matrix = torch.zeros((num_nodes, num_nodes))

    distance_matrix[distance_df['source_idx'], distance_df['target_idx']] = torch.tensor(distance_df['distance'].values, dtype=torch.float)
    similarity_matrix[similarity_df['source_idx'], similarity_df['target_idx']] = torch.tensor(
        similarity_df['similarity'].values, dtype=torch.float)

    distance_matrix = (distance_matrix + distance_matrix.t())
    similarity_matrix = (similarity_matrix + similarity_matrix.t())

    distance_matrix.fill_diagonal_(1000.0)
    similarity_matrix.fill_diagonal_(1.0)

    if Aij.shape != distance_matrix.shape or Aij.shape != similarity_matrix.shape:
        print(
            f"Error: Dimensions of Aij {Aij.shape}, distance_matrix {distance_matrix.shape}, and similarity_matrix {similarity_matrix.shape} do not match!")
    else:
        print("All matrices have matching dimensions.")

    total_time = time.time() - start_time
    print(f"Total execution time of load_data: {total_time:.2f} seconds")

    return node_df, Aij, distance_matrix, similarity_matrix, ni, distance_df, similarity_df, ss_pop_df, ss_weight_df


def visualize_results(output_path, param, ss_pop_df, ss_weight_df, min_exp_w, max_exp_w, min_exp_n, max_exp_n,city):
    """变成for循环，把output_nodes和output_edges开头的csv文件都绘图"""
    ss_weight_df = ss_weight_df[ss_weight_df['start_new_id'] != ss_weight_df['end_new_id']] # 自边不算流量

    node_files = sorted(glob.glob(os.path.join(output_path, 'output_nodes*.csv')))
    edge_files = sorted(glob.glob(os.path.join(output_path, 'output_edges*.csv')))

    for nf in node_files:
        nodes_df = pd.read_csv(nf)
        nodes_df = nodes_df[nodes_df['population'] >= 1].copy()
        base_name = os.path.splitext(os.path.basename(nf))[0]
        node_plotter = LogLogPDFPlotter(nodes_df, base_name)
        node_plotter.plot_log_log_pdf(
            path=output_path,
            param_name=param,
            column_name="population",
            plot_type="node",
            plot_ss=True,
            ss_df=ss_pop_df,
            ss_column_name="steady_pop",
            use_bounds=True,
            min_exp=min_exp_n,
            max_exp=max_exp_n,
            city=city
        )

    for ef in edge_files:
        edges_df = pd.read_csv(ef)
        edges_df = edges_df[edges_df['weight'] > 0].copy()
        edges_df = edges_df[edges_df['source'] != edges_df['target']]
        base_name = os.path.splitext(os.path.basename(ef))[0]
        edge_plotter = LogLogPDFPlotter(edges_df, base_name)
        edge_plotter.plot_log_log_pdf(
            path=output_path,
            param_name=param,
            column_name="weight",
            plot_type="weight",
            plot_ss=True,
            ss_df=ss_weight_df,
            ss_column_name="weight_30min",
            use_bounds=True,
            min_exp=min_exp_w,
            max_exp=max_exp_w
        )


def check_steady_state_similarity(node_df, edge_output, ss_pop_df, ss_weight_df, param):
    start_time = time.time()

    pop_vector = torch.tensor(node_df['population'].values, dtype=torch.float32)
    steady_pop_vector = torch.tensor(ss_pop_df['steady_pop'].values, dtype=torch.float32)

    num_nodes = node_df.shape[0]
    weight_matrix = torch.zeros((num_nodes, num_nodes))
    steady_weight_matrix = torch.zeros((num_nodes, num_nodes))

    weight_matrix[edge_output['source'], edge_output['target']] = torch.tensor(edge_output['weight'].values, dtype=torch.float)

    row_indices = ss_weight_df['start_new_idx'].values
    col_indices = ss_weight_df['end_new_idx'].values
    weights = torch.tensor(ss_weight_df['weight_30min'].values, dtype=torch.float)

    steady_weight_matrix[row_indices, col_indices] = weights


    if pop_vector.shape != steady_pop_vector.shape:
        print(f"Error: Population vector dimensions do not match for {param}!")
        return
    if weight_matrix.shape != steady_weight_matrix.shape:
        print(f"Error: Weight matrix dimensions do not match for {param}!")
        return

    pop_similarity = torch.nn.functional.cosine_similarity(pop_vector, steady_pop_vector, dim=0).item()
    weight_similarity = torch.nn.functional.cosine_similarity(weight_matrix.flatten(), steady_weight_matrix.flatten(), dim=0).item()

    print(f'Parameters: {param}')
    print(f'Population similarity: {pop_similarity:.4f}')
    print(f'Weight matrix similarity: {weight_similarity:.4f}')

    total_time = time.time() - start_time
    print(f"Total execution time of check_steady_state_similarity: {total_time:.2f} seconds")


def calculate_density_weight(df, column_name, min_exp, max_exp):
    bins = np.logspace(min_exp, max_exp, int((max_exp - min_exp) / 0.2) + 1)
    counts, _ = np.histogram(df[column_name], bins=bins, density=True)
    counts += 1e-10
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return bin_centers, counts

def calculate_density_node (df, column_name, min_exp, max_exp):
    bins = np.logspace(min_exp, max_exp, int((max_exp - min_exp) / 0.1) + 1)
    counts, _ = np.histogram(df[column_name], bins=bins, density=True)
    counts += 1e-10
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return bin_centers, counts

def calculate_divergences(edge_output_df, ss_weight_df, min_exp_w, max_exp_w, node_output_df, ss_pop_df, min_exp_n, max_exp_n):
    edge_centers, edge_density = calculate_density_weight(edge_output_df, 'weight', min_exp_w, max_exp_w)
    ss_edge_centers, ss_edge_density = calculate_density_weight(ss_weight_df, 'weight_30min',min_exp_w, max_exp_w)

    edge_kl_divergence = stats.entropy(edge_density, ss_edge_density)
    edge_js_divergence = jensenshannon(edge_density, ss_edge_density, base=2)

    edge_ks_statistic, edge_ks_p_value = ks_2samp(edge_output_df['weight'], ss_weight_df['weight_30min'].dropna()
)

    node_centers, node_density = calculate_density_node(node_output_df, 'population', min_exp_n, max_exp_n)
    ss_node_centers, ss_node_density = calculate_density_node(ss_pop_df, 'steady_pop', min_exp_n, max_exp_n)

    node_kl_divergence = stats.entropy(node_density, ss_node_density)
    node_js_divergence = jensenshannon(node_density, ss_node_density, base=2)
    node_ks_statistic, node_ks_p_value = ks_2samp(node_output_df['population'], ss_pop_df['steady_pop'])

    print(f"edge_kl_divergence: {edge_kl_divergence}, "
          f"edge_js_divergence: {edge_js_divergence}, "
          f"edge_ks_statistic: {edge_ks_statistic}, "
          f"edge_ks_p_value: {edge_ks_p_value}, "
          f"node_kl_divergence: {node_kl_divergence}, "
          f"node_js_divergence: {node_js_divergence}, "
          f"node_ks_statistic: {node_ks_statistic}, "
          f"node_ks_p_value: {node_ks_p_value}")

    return (
        edge_kl_divergence, edge_js_divergence, edge_ks_statistic, edge_ks_p_value,
        node_kl_divergence, node_js_divergence, node_ks_statistic, node_ks_p_value
    )


def main(node_df, Aij, distance_matrix, similarity_matrix, ni, ss_pop_df, ss_weight_df, resultpath, min_exp_w,
         max_exp_w, min_exp_n, max_exp_n, city, alpha, beta, λ, T, dt):

    num_nodes = node_df.shape[0]

    net_dyn = NetworkDynamics(distance_matrix, similarity_matrix, alpha=alpha, beta=beta)
    node_dyn = NodeDynamics(dt=dt)

    output_path = os.path.join(resultpath, f'./results_alpha_{alpha}_beta_{beta}_lambda_{λ}/')
    os.makedirs(output_path, exist_ok=True)

    initial_total_population = ni.sum().item()
    previous_ni = ni.clone()
    previous_Aij = Aij.clone()

    pop_changes = []
    weight_similarities = []

    start_time = time.time()
    for iteration, t in enumerate(np.arange(0, T, dt)):
        print(iteration)

        inflow = Aij.sum(axis=0).view(-1, 1)
        outflow = Aij.sum(axis=1).view(-1, 1)

        ni = node_dyn.update_population(ni, inflow, outflow, initial_total_population)
        Aij = net_dyn.edge_evolving(ni, λ)

        if int(iteration / dt) % 5000 == 0:
            node_df['population'] = ni.numpy().flatten()
            node_df.to_csv(os.path.join(output_path, f'output_nodes{iteration + 1}.csv'), index=False)

            edges = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if Aij[i, j] >= 0:
                        edges.append([i, j, Aij[i, j].item()])
            edge_output = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
            edge_output.to_csv(os.path.join(output_path, f'output_edges{iteration + 1}.csv'), index=False)

        # 判断收敛
        pop_change, weight_similarity = check_convergence(ni, previous_ni, Aij, previous_Aij)
        pop_changes.append(pop_change)
        weight_similarities.append(weight_similarity)
        previous_ni = ni.clone()
        previous_Aij = Aij.clone()

        if t >= 1:
            if pop_change < 5 and weight_similarity > 0.95:
                print(f'Convergence achieved at iteration {t}')
                break

    total_time = time.time() - start_time
    print(f"Total execution time of maindiedai: {total_time:.2f} seconds")

    node_df['population'] = ni.numpy().flatten()
    node_df.to_csv(os.path.join(output_path, 'output_nodes.csv'), index=False)

    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if Aij[i, j] >= 0:
                edges.append([i, j, Aij[i, j].item()])

    edge_output = pd.DataFrame(edges, columns=['source', 'target', 'weight'])
    edge_output.to_csv(os.path.join(output_path, 'output_edges.csv'), index=False)

    param = f'α_{alpha}_β_{beta}_lambda_{λ}'
    plot_convergence(pop_changes, weight_similarities, dt, output_path)
    check_steady_state_similarity(node_df, edge_output, ss_pop_df, ss_weight_df, param)
    visualize_results(output_path, param, ss_pop_df, ss_weight_df,min_exp_w, max_exp_w, min_exp_n, max_exp_n,city)

    plt.figure(figsize=(12, 7))

    node_output = node_df
    divergence_results = calculate_divergences(edge_output, ss_weight_df, min_exp_w, max_exp_w, node_output, ss_pop_df,
                                               min_exp_n, max_exp_n)

    print(f"Parameters: alpha={alpha}, beta={beta}, λ={λ}")
    print(f"Edge KL Divergence: {divergence_results[0]}")
    print(f"Edge JS Divergence: {divergence_results[1]}")
    print(f"Edge KS Statistic: {divergence_results[2]}, p-value: {divergence_results[3]}")
    print(f"Node KL Divergence: {divergence_results[4]}")
    print(f"Node JS Divergence: {divergence_results[5]}")
    print(f"Node KS Statistic: {divergence_results[6]}, p-value: {divergence_results[7]}")

    divergence_data = {
        'alpha': alpha,
        'beta': beta,
        'λ': λ,
        'Edge KL Divergence': divergence_results[0],
        'Edge JS Divergence': divergence_results[1],
        'Edge KS Statistic': divergence_results[2],
        'Edge KS p-value': divergence_results[3],
        'Node KL Divergence': divergence_results[4],
        'Node JS Divergence': divergence_results[5],
        'Node KS Statistic': divergence_results[6],
        'Node KS p-value': divergence_results[7]
    }

    divergence_df = pd.DataFrame([divergence_data])

    divergence_csv_path = os.path.join(resultpath, 'divergence_results.csv')

    if os.path.exists(divergence_csv_path):
        divergence_df.to_csv(divergence_csv_path, mode='a', header=False, index=False)
    else:
        divergence_df.to_csv(divergence_csv_path, mode='w', header=True, index=False)

if __name__ == '__main__':
    city='BJ'
    datapath = f'Data&Result_{city}/Data'
    resultpath = f'Data&Result_{city}/Result'
    node_df, Aij, distance_matrix, similarity_matrix, ni, distance_df, similarity_df, ss_pop_df, ss_weight_df = load_data(
        datapath, f'2024{city}pop_2km.csv', 'distance.csv', 'similarity_double_norm.csv',
        '2024aveweight_steadystate33.csv', '2024pop_steadystate33.csv')

    min_val_w = ss_weight_df.min().min()
    max_val_w = ss_weight_df.max().max()
    min_exp_w = math.floor(math.log10(min_val_w)) if min_val_w > 0 else 0
    max_exp_w = math.ceil(math.log10(max_val_w)) if max_val_w > 0 else 0

    min_val_n = ss_pop_df.min().min()
    max_val_n = ss_pop_df.max().max()
    min_exp_n = math.floor(math.log10(min_val_n)) if min_val_n > 0 else 0
    max_exp_n = math.ceil(math.log10(max_val_n)) if max_val_n > 0 else 0

    alpha_list = np.arange(0, 3.0, 0.1).tolist()
    beta_list = np.arange(1.0, 4.0, 0.1).tolist()
    λ_list = np.arange(0.1, 0.9, 0.1).tolist()

    parameters = []
    for alpha in alpha_list:
        for beta in beta_list:
            for λ in λ_list:
                parameters.append((alpha, beta, λ))

    # 遍历参数组合
    for p in tqdm.tqdm(parameters):
        main(node_df, Aij, distance_matrix, similarity_matrix, ni, ss_pop_df, ss_weight_df, resultpath, min_exp_w,
             max_exp_w, min_exp_n, max_exp_n, city,
             alpha=p[0], beta=p[1], λ=p[2], T=8000, dt=1)