# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS
import rasterio
from rasterstats import zonal_stats

city = "US_SantaClaraCounty"

# ---------------- 路径 ----------------
boundary_path = fr"Data&Result_{city}\Data\{city}_cbg.geojson"
raster_path   = fr"Data\usa_ppp_2020_UNadj.tif"
out_csv       = fr"Data&Result_{city}\Data\2019pop_steadystate33.csv"
out_geojson   = fr"Data&Result_{city}\Data\{city}_grid.geojson"
out_centroid = fr"Data&Result_{city}\Data\centroidsv2.csv"

# ------------------------------------------------------------------
# 1) 读取行政区并 dissolve → 单一面
# ------------------------------------------------------------------
cbg = gpd.read_file(boundary_path)

# 计算合适的 UTM 投影（按质心经度）
centroid = cbg.geometry.unary_union.centroid
utm_zone = int((centroid.x + 180) // 6) + 1
epsg     = 32600 + utm_zone if centroid.y >= 0 else 32700 + utm_zone
proj_crs = CRS.from_epsg(epsg)

cbg_proj = cbg.to_crs(proj_crs)
boundary_union = cbg_proj.unary_union              # dissolve 成单一多边形

# ------------------------------------------------------------------
# 2) 生成 2 km × 2 km 网格（保留完整矩形，不 clip）
# ------------------------------------------------------------------
minx, miny, maxx, maxy = cbg_proj.total_bounds
grid_size = 2000  # 米

cols = np.arange(minx, maxx, grid_size)
rows = np.arange(miny, maxy, grid_size)

cells = [
    box(x, y, x + grid_size, y + grid_size)
    for x in cols
    for y in rows
    if box(x, y, x + grid_size, y + grid_size).intersects(boundary_union)
]

grid = gpd.GeoDataFrame(geometry=cells, crs=proj_crs)

# ------------------------------------------------------------------
# 3) zonal stats 汇总人口栅格
# ------------------------------------------------------------------
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    nodata_val = src.nodata

grid_raster = grid.to_crs(raster_crs)    # 与栅格同坐标系

stats = zonal_stats(
    grid_raster.geometry,
    raster_path,
    stats=["sum"],
    nodata=nodata_val,
    all_touched=True,
)
grid["steady_pop"] = [s["sum"] or 0 for s in stats]

# ------------------------------------------------------------------
# 4) 生成 new_id，并导出
# ------------------------------------------------------------------
grid = grid.loc[grid["steady_pop"] >= 1].reset_index(drop=True)
grid["new_id"] = [f"{city}_{i+1:05d}" for i in grid.index]

grid.to_crs(4326).to_file(out_geojson, driver="GeoJSON")       # 可选：存回 WGS84
grid["resident_count"] = grid["steady_pop"]
grid[["new_id", "steady_pop","resident_count"]].to_csv(out_csv, index=False, encoding="utf-8-sig")

grid_ll = grid.to_crs(4326)                      # 转 WGS84
grid_ll["longitude"] = grid_ll.geometry.centroid.x
grid_ll["latitude"]  = grid_ll.geometry.centroid.y

grid_ll[["new_id", "longitude", "latitude"]].to_csv(out_centroid, index=False, encoding="utf-8-sig")