# -*- coding: utf-8 -*-
"""Through this code,
1) the human flow between cbgs is resampled to that between 2km grids;
2) the population remaining in the original grid is written to the diagonal positions of the edge
 matrix, making the edge matrix complete."""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy import sparse
from pathlib import Path


class SteadyPopulationFlow:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def process_avg_weight(self, combined_df):
        grouped_df = combined_df.groupby(['start_new_id', 'end_new_id'])['move_num'].sum().reset_index()
        grouped_df['avg_weight30min'] = grouped_df['move_num'] / 48
        return grouped_df[['start_new_id', 'end_new_id', 'avg_weight30min']]

    def calculate_cumulated_flows(self, avg_weight_df):
        inflow = avg_weight_df.groupby('end_new_id')['avg_weight30min'].sum().reset_index()
        inflow.columns = ['new_id', 'cumulated_inflow']
        outflow = avg_weight_df.groupby('start_new_id')['avg_weight30min'].sum().reset_index()
        outflow.columns = ['new_id', 'cumulated_outflow']
        result_df = pd.merge(inflow, outflow, on='new_id', how='outer').fillna(0)
        return result_df

    def calculate_steady_population(self, population_df, cumulated_flows_df):
        merged_df = pd.merge(population_df, cumulated_flows_df, on='new_id', how='left').fillna(0)
        merged_df['steady_pop'] = merged_df['resident_count'] + merged_df['cumulated_inflow'] - merged_df[
            'cumulated_outflow']
        return merged_df

    def process_weight_steadystate(self, avg_weight_df):
        avg_weight_df = avg_weight_df.rename(columns={'avg_weight30min': 'weight_30min'})
        avg_weight_df['weight_1d'] = avg_weight_df['weight_30min'] * 48
        return avg_weight_df[['start_new_id', 'end_new_id', 'weight_30min', 'weight_1d']]


def cbg2grid_od(cbg_path, grid_path, od_df):
    cbg = gpd.read_file(cbg_path)[['GEOID', "geometry"]]
    grid = gpd.read_file(grid_path)[["new_id", "geometry"]]

    od_df["start_new_id"] = od_df["start_new_id"].astype(str).str.zfill(12)
    od_df["end_new_id"] = od_df["end_new_id"].astype(str).str.zfill(12)

    cbg_ids_set = set(cbg["GEOID"])
    od_df = od_df.loc[od_df["start_new_id"].isin(cbg_ids_set) & od_df["end_new_id"].isin(cbg_ids_set)].copy()

    centroid = cbg.unary_union.centroid
    utm_zone = int((centroid.x + 180) // 6) + 1
    epsg_utm = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone

    cbg_proj = cbg.to_crs(epsg=epsg_utm)
    grid_proj = grid.to_crs(epsg=epsg_utm)

    inter = gpd.overlay(cbg_proj, grid_proj, how="intersection")
    inter["area"] = inter.area
    area_sum = inter.groupby("GEOID")["area"].transform("sum")
    inter["prop"] = inter["area"] / area_sum

    cbg_ids = cbg_proj["GEOID"].unique()
    grid_ids = grid_proj["new_id"].unique()
    cbg_idx = {g: i for i, g in enumerate(cbg_ids)}
    grid_idx = {g: i for i, g in enumerate(grid_ids)}

    rows_P = inter["GEOID"].map(cbg_idx).to_numpy(int)
    cols_P = inter["new_id"].map(grid_idx).to_numpy(int)
    data_P = inter["prop"].to_numpy(float)

    P = sparse.coo_matrix((data_P, (rows_P, cols_P)), shape=(len(cbg_ids), len(grid_ids))).tocsr()

    rows_W = od_df["start_new_id"].map(cbg_idx).to_numpy(int)
    cols_W = od_df["end_new_id"].map(cbg_idx).to_numpy(int)
    data_W = od_df["weight_30min"].to_numpy(float)

    W = sparse.coo_matrix((data_W, (rows_W, cols_W)), shape=(len(cbg_ids), len(cbg_ids))).tocsr()

    G = P.T @ W @ P
    G.sum_duplicates()
    G.setdiag(0)
    G.eliminate_zeros()

    G_coo = G.tocoo()
    out_df = (
        pd.DataFrame({
            "start_new_id": [grid_ids[i] for i in G_coo.row],
            "end_new_id": [grid_ids[j] for j in G_coo.col],
            "weight_30min": G_coo.data
        })
        .groupby(["start_new_id", "end_new_id"], as_index=False)
        .agg({"weight_30min": "sum"})
    )

    diagonal_rows = pd.DataFrame({
        'start_new_id': grid_ids,
        'end_new_id': grid_ids,
        'weight_30min': 0.0
    })

    out_df_with_diag = pd.concat([out_df, diagonal_rows], ignore_index=True)

    out_df_with_diag = out_df_with_diag.groupby(["start_new_id", "end_new_id"], as_index=False).agg(
        {"weight_30min": "sum"})

    return out_df_with_diag


def supplement_self_weight(steady_pop_df, weight_df):
    steady_pop_df['new_id'] = steady_pop_df['new_id'].astype(str)
    weight_df['start_new_id'] = weight_df['start_new_id'].astype(str)
    weight_df['end_new_id'] = weight_df['end_new_id'].astype(str)

    merged_df = pd.merge(weight_df, steady_pop_df, left_on='start_new_id', right_on='new_id', how='left')

    grouped = merged_df[merged_df['start_new_id'] != merged_df['end_new_id']] \
        .groupby('start_new_id')['weight_30min'].sum().reset_index() \
        .rename(columns={'weight_30min': 'd'})

    merged_df = pd.merge(merged_df, grouped, on='start_new_id', how='left')
    merged_df['d'] = merged_df['d'].fillna(0)

    merged_df.loc[merged_df['start_new_id'] == merged_df['end_new_id'], 'weight_30min'] = \
        merged_df['steady_pop'] - merged_df['d']

    merged_df.loc[merged_df['weight_30min'] < 0, 'weight_30min'] = 0

    return merged_df[['start_new_id', 'end_new_id', 'weight_30min']]


if __name__ == "__main__":
    city = "US_SantaClaraCounty"
    year = 2019
    root = Path(f"Data&Result_{city}") / "Data"

    analysis = SteadyPopulationFlow(f'{root}/')
    combined_df = pd.read_csv(f'{root}/{year}{city}od_1d_NC.csv')
    avg_weight_df = analysis.process_avg_weight(combined_df)
    cumulated_flows_df = analysis.calculate_cumulated_flows(avg_weight_df)
    population_df = pd.read_csv(os.path.join(analysis.data_folder, f'{year}{city}pop_NC.csv'))

    steady_population_df = analysis.calculate_steady_population(population_df, cumulated_flows_df)

    weight_steadystate_df = analysis.process_weight_steadystate(avg_weight_df)

    cbg_path = root / f"{city}_cbg.geojson"
    grid_path = root / f"{city}_grid.geojson"

    grid_weight_df = cbg2grid_od(
        cbg_path, grid_path, weight_steadystate_df
    )

    pop_file_path = root / f'{year}pop_steadystate33.csv'
    if not pop_file_path.exists():
        print(f"could not find {pop_file_path}")

    grid_steady_pop_df = pd.read_csv(pop_file_path)

    final_weight_df = supplement_self_weight(grid_steady_pop_df, grid_weight_df)

    final_weight_df.to_csv(f'{root}/{year}aveweight_steadystate33.csv', index=False)

    print("All steps completed successfully.")