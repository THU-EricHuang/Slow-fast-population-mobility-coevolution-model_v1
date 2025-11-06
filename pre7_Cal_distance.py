import geopandas as gpd
import pandas as pd
from itertools import combinations
from shapely.geometry import Point
from tqdm import tqdm

def calculate_grid_distances(centroid_path, output_path):
    df = pd.read_csv(centroid_path)
    df['geometry'] = df.apply(lambda x: Point((x['longitude'], x['latitude'])), axis=1)

    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    gdf = gdf.to_crs(epsg=32633)  # Kept user-specified EPSG

    edges = []
    combs = list(combinations(range(len(gdf)), 2))

    for i, j in tqdm(combs, desc="Calculating distances"):
        distance = gdf.geometry.iloc[i].distance(gdf.geometry.iloc[j])
        edges.append([df['new_id'].iloc[i], df['new_id'].iloc[j], distance])

    edge_df = pd.DataFrame(edges, columns=['source_id', 'target_id', 'distance'])
    edge_df.to_csv(output_path, index=False)  # unit：meter


if __name__ == "__main__":
    city = 'US_SantaClaraCounty'
    path = f'Data&Result_{city}/Data'

    centroid_file = f'{path}/centroidsv2.csv'
    output_file = f'{path}/distance.csv'

    calculate_grid_distances(centroid_file, output_file)