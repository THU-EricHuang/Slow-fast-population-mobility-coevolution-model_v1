import os, sys, subprocess, json, re, shutil
import geopandas as gpd
import pandas as pd

def process_overture_pois(city):
    root       = f"Data&Result_{city}/Data"
    os.makedirs(root, exist_ok=True)
    shp_path   = f"{root}/{city}_grid.geojson"
    out_geojson = f"{root}/{city}_poi.geojson"
    out_dir     = f"{root}/POI_overture_{city}"
    os.makedirs(out_dir, exist_ok=True)

    poly = gpd.read_file(shp_path).to_crs(4326).union_all()
    west, south, east, north = poly.bounds
    bbox_str = f"{west},{south},{east},{north}"

    tmp_parquet = f"{root}/_tmp_{city}_places.gpq"
    if not os.path.exists(tmp_parquet):
        exe = shutil.which("overturemaps")
        cmd = ([exe] if exe else [sys.executable, "-m", "overturemaps"]) + [
            "download", f"--bbox={bbox_str}", "-f", "geoparquet",
            "--type=place", "-o", tmp_parquet
        ]
        print("Downloading Overture Places …")
        subprocess.run(cmd, check=True)

    gdf = gpd.read_parquet(tmp_parquet)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf["plonwgs84"] = gdf.geometry.x
    gdf["platwgs84"] = gdf.geometry.y

    def primary(cat):
        if isinstance(cat, dict):
            return cat.get("primary")
        if isinstance(cat, str):
            try:
                return json.loads(cat).get("primary")
            except Exception:
                return None
        return None

    gdf["primary_cat"] = gdf["categories"].apply(primary)

    poly_mask = gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    residential_set = {
        "residential", "house", "apartment", "dormitory",
        "residential_building", "apartment_building", "housing_complex"
    }
    res_mask = gdf["primary_cat"].isin(residential_set)
    gdf.loc[poly_mask & res_mask, "geometry"] = (
        gdf.loc[poly_mask & res_mask, "geometry"].centroid
    )

    gdf["plonwgs84"] = gdf.geometry.x
    gdf["platwgs84"] = gdf.geometry.y

    gdf = gdf[gdf.geometry.type.isin(["Point", "MultiPoint"])]

    SET = lambda *xs: set(xs)
    MAP = {
        "transportation": SET(
            "bus_station", "train_station", "subway_station", "light_rail_station",
            "airport", "parking", "gas_station", "ferry_terminal",
            "taxi_stand", "car_rental", "bicycle_sharing_station"
        ),
        "food_beverage": SET(
            "restaurant", "fast_food", "cafe", "coffee_shop", "bar", "pub",
            "food_court", "bakery", "tea_room", "ice_cream_shop"
        ),
        "retail_service": SET(
            "convenience_store", "supermarket", "grocery_store", "department_store",
            "shopping_mall", "clothing_store", "beauty_salon", "hairdresser",
            "laundry_service"
        ),
        "health_medical": SET(
            "hospital", "clinic", "pharmacy", "doctor", "dentist",
            "veterinary_clinic", "urgent_care"
        ),
        "education": SET(
            "school", "college", "university", "library", "kindergarten",
            "research_institute", "training_center"
        ),
        "government": SET(
            "townhall", "courthouse", "police", "fire_station",
            "government_office", "embassy", "post_office"
        ),
        "finance": SET(
            "bank", "atm", "insurance_agency", "credit_union",
            "currency_exchange"
        ),
        "residential": residential_set,
        "recreation": SET(
            "park", "stadium", "sports_center", "gym", "museum", "theatre",
            "amusement_park", "zoo", "aquarium"
        ),
    }

    def to_major(cat):
        for major, bucket in MAP.items():
            if cat in bucket:
                return major
        return "other"

    gdf["major_cat"] = gdf["primary_cat"].apply(to_major)

    keep_cols = [c for c in gdf.columns if c != "categories"]
    gdf[keep_cols].to_file(out_geojson, driver="GeoJSON")

    safe = lambda s: re.sub(r"[\\/*?\"<>|:]+", "_", str(s))
    for cat, sub in gdf.groupby("major_cat"):
        cat_file = safe(cat)
        cols = ["plonwfs84", "platwgs84"] + [
            c for c in sub.columns if c not in ("geometry", "major_cat", "primary_cat")
        ]
        sub[cols].to_csv(
            f"{out_dir}/{cat_file}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"✓ {cat_file:15s} -> {len(sub):7d} rows")

    print("All done!  GeoJSON:", out_geojson, "|  CSV dir:", out_dir)


if __name__ == "__main__":
    city = "US_SantaClaraCounty"
    process_overture_pois(city)