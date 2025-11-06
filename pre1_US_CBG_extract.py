import os
from pathlib import Path
import geopandas as gpd
import pandas as pd

shp_path = Path("Data&Result_US\Data\Admin\cb_2019_us_bg_500k.shp")
cbg = gpd.read_file(shp_path)

CITY_RULES = {
    "SantaClaraCounty":      dict(STATEFP="06", COUNTYFP=["085"]),  # Santa Clara County, CA
}

for city, rule in CITY_RULES.items():
    g = cbg[
        (cbg["STATEFP"] == rule["STATEFP"]) &
        (cbg["COUNTYFP"].isin(rule["COUNTYFP"]))
    ].copy()

    if g.empty:
        print(f"[WARN] {city}: No matching")
        continue

    # —— 3-2) 输出 GeoJSON（保持原字段）—— #
    out_dir = Path('Data&Result_US_'+ city) / "Data"
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = out_dir / f"US_{city}_cbg.geojson"
    g.to_file(geojson_path, driver="GeoJSON")

    # —— 3-3) 计算质心、生成 CSV —— #
    g = g.to_crs("EPSG:4326")
    g["longitude"] = g.geometry.centroid.x
    g["latitude"]  = g.geometry.centroid.y
    g.rename(columns={"GEOID": "new_id"}, inplace=True)
    centroid_df = g[["new_id", "longitude", "latitude"]]
    centroid_df.to_csv(out_dir / "centroidsv_cbg.csv", index=False)

    print(f"[DONE] {city}: {len(g):,} CBG → {geojson_path}")

