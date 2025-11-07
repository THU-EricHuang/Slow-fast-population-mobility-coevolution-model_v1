"""人口的数据用不了，还是用worldpop之类的在台式机统计"""

import json, pandas as pd, glob, pathlib
import os.path

CITY_CBG = {
    "SantaClaraCounty": ('CA', 'Data&Result_US_SantaClaraCounty')
}
STATES = {"CA"}

_json = lambda s: json.loads(s) if isinstance(s, str) and s.startswith("{") else {}


def city_id_sets(root: pathlib.Path):
    out = {}
    for city, (_, folder) in CITY_CBG.items():
        p = os.path.join(folder,"Data","centroidsv_cbg.csv")
        out[city] = set(pd.read_csv(p, usecols=["new_id"], dtype=str)["new_id"])
    return out


def residents(chunk: pd.DataFrame) -> pd.DataFrame:
    js = chunk["NIGHTLIFE_DEVICE_HOME_AREAS"].apply(_json)
    cnt = [d.get(a, 0) for a, d in zip(chunk["AREA"], js)]  # Night at home
    return pd.DataFrame({"new_id": chunk["AREA"], "resident_count": cnt})


def weekday_od(chunk: pd.DataFrame) -> pd.DataFrame:
    js = chunk["WEEKDAY_DEVICE_HOME_AREAS"].apply(_json)
    rows = [(home, dest, v / 23)              # weekday mean value
            for dest, d in zip(chunk["AREA"], js)
            for home, v in d.items()]
    return pd.DataFrame(rows,
        columns=["start_new_id", "end_new_id", "move_num"])


def run(data_root: str):
    root  = pathlib.Path(data_root)
    csvs  = sorted(glob.glob(str(root /"od201901/Neighborhood_Patterns_US_One_Month_Sample-*2019-01-01.csv")))
    if not csvs: raise FileNotFoundError("no sample csv found")

    city_ids   = city_id_sets(root)
    pop_bucket = {c: {} for c in CITY_CBG}
    od_bucket  = {c: [] for c in CITY_CBG}

    for fp in csvs:
        for c in pd.read_csv(fp,
                             usecols=["AREA","REGION",
                                      "NIGHTLIFE_DEVICE_HOME_AREAS",
                                      "WEEKDAY_DEVICE_HOME_AREAS"],
                             dtype={"AREA":str,"REGION":str},
                             chunksize=200_000):
            c["AREA"] = c["AREA"].str.zfill(12)
            chunk = c[c["REGION"].isin(STATES)]
            if chunk.empty:
                continue

            for city, (state, folder) in CITY_CBG.items():
                sub = chunk[chunk["REGION"] == state]
                if sub.empty: continue

                ids = city_ids[city]
                sub = sub[sub["AREA"].isin(ids)]
                if sub.empty: continue

                # POP
                res = residents(sub)
                for nid, cnt in zip(res["new_id"], res["resident_count"]):
                    pop_bucket[city][nid] = pop_bucket[city].get(nid, 0) + cnt

                # OD
                od = weekday_od(sub)
                od = od[od["start_new_id"].isin(ids)]
                od_bucket[city].append(od)

    for city, (_, folder) in CITY_CBG.items():
        data_dir = os.path.join(folder, "Data")
        # data_dir.mkdir(parents=True, exist_ok=True)

        # pop
        pop_df = pd.Series(pop_bucket[city]).reset_index()
        pop_df.columns = ["new_id", "resident_count"]
        pop_df.to_csv(os.path.join(data_dir , f"2019US_{city}pop_NC.csv"), index=False)
        print(city, "population =", pop_df["resident_count"].sum())

        # od
        if od_bucket[city]:
            od_df = (pd.concat(od_bucket[city])
                       .groupby(["start_new_id","end_new_id"], as_index=False)
                       .sum())
            od_df.to_csv(os.path.join(data_dir , f"2019US_{city}od_1d_NC.csv"), index=False)
            print(city, "OD rows   =", len(od_df))

if __name__ == "__main__":
    run(r'raw data')

