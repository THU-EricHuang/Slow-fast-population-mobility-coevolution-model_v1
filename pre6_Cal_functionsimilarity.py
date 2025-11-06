

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from shapely.validation import make_valid

def create_grid_polygon(lat, lon, cell_size=0.018, lon_cell_size=0.022):  # Unit：degree, depend on locations
    half_lat_size = cell_size / 2
    half_lon_size = lon_cell_size / 2
    return Polygon([
        (lon - half_lon_size, lat - half_lat_size),
        (lon - half_lon_size, lat + half_lat_size),
        (lon + half_lon_size, lat + half_lat_size),
        (lon + half_lon_size, lat - half_lat_size)
    ])

def calculate_similarity(grid_csv, poi_folder, grid_output_csv, edge_output_csv, grid_shp=None):
    if grid_shp:
        gdf = gpd.read_file(grid_shp)
        if "new_id" not in gdf.columns:
            gdf["new_id"] = gdf["CVEGEO"]
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: make_valid(geom) if not geom.is_valid else geom
        )
    else:
        centroids = pd.read_csv(grid_csv)
        gdf = gpd.GeoDataFrame(
            centroids,
            geometry=gpd.points_from_xy(centroids.longitude, centroids.latitude),
            crs="EPSG:4326",
        )
        gdf["geometry"] = gdf.apply(
            lambda r: create_grid_polygon(r.latitude, r.longitude), axis=1
        )

    poi_files = [
        os.path.join(poi_folder, f) for f in os.listdir(poi_folder) if f.endswith(".csv")
    ]
    for file in tqdm(poi_files, desc="Processing POI files"):
        poi = pd.read_csv(file, low_memory=False)
        poi_gdf = gpd.GeoDataFrame(
            poi,
            geometry=gpd.points_from_xy(poi.plonwgs84, poi.platwgs84),
            crs="EPSG:4326",
        )
        if poi_gdf.crs != gdf.crs:
            poi_gdf = poi_gdf.to_crs(gdf.crs)
        col = os.path.splitext(os.path.basename(file))[0]
        join = gpd.sjoin(
            poi_gdf[["geometry"]],
            gdf[["geometry"]],
            predicate="within",
            how="left",
        )
        cnt = join.groupby("index_right").size()
        gdf[col] = 0
        gdf.loc[cnt.index, col] = cnt.values

    poi_cols = [os.path.splitext(os.path.basename(f))[0] for f in poi_files]
    gdf['total_poi'] = gdf[poi_cols].sum(axis=1)

    for c in poi_cols:
        col_sum = gdf[c].sum()
        if col_sum:
            gdf[c] = gdf[c] / col_sum

    row_sum = gdf[poi_cols].sum(axis=1).replace(0, 1)
    for c in poi_cols:
        gdf[c] = gdf[c] / row_sum

    gdf = gdf.fillna(0)
    gdf.to_file(grid_output_csv, driver='GeoJSON')

    feature_cols = poi_cols # ,'total_poi'
    gdf[feature_cols] = gdf[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    vectors = gdf[feature_cols].values.astype(float)
    similarity_matrix = cosine_similarity(vectors)

    edges = []
    for i in tqdm(range(len(similarity_matrix)), desc="Calculating similarities"):
        for j in range(i + 1, len(similarity_matrix)):
            edges.append([gdf.iloc[i]['new_id'], gdf.iloc[j]['new_id'], similarity_matrix[i][j]])

    edge_df = pd.DataFrame(edges, columns=['source_id', 'target_id', 'similarity'])
    edge_df.to_csv(edge_output_csv, index=False)

if __name__ == "__main__":
    city = 'US_SantaClaraCounty'
    year = '2019'
    path=f'Data&Result_{city}/Data'
    calculate_similarity(f'{path}/centroidsv2.csv', f'Data&Result_{city}\Data\POI_overture_{city}',
                         f'{path}/grid_with_pois.geojson', f'{path}/similarity_double_norm.csv',
                         f'{path}/{city}_grid.geojson')
