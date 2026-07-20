"""
拿機構座標比對 Wikidata 權威園區清單，取代 Overpass is_in() 判斷方法 A
------------------------------------------------------------------
輸入：
  - geocoded.csv        （Phase 2a 產出，機構座標）
  - parks_wikidata.csv  （parks_gt.py 產出，全球科學/技術園區權威清單）
輸出：
  - park_matches.csv    （每筆機構座標，最近的權威園區、距離、是否判定在園區內）

比對邏輯：對每個機構座標，算它跟 Wikidata 清單裡每個園區座標的 Haversine
距離，取最近的一個。如果距離在 MATCH_RADIUS_M 之內，判定為「落在/緊鄰
這個園區」。這是距離門檻法（點對點），不是嚴格的多邊形邊界判斷，方法論上
比 is_in() 寬鬆一點，但換來的是：完全不依賴 OSM 多邊形標記品質（全球不
均勻，這是換掉 Overpass is_in() 的原因），而且整個比對是本地運算，沒有
任何外部 API 呼叫，沒有 rate limit 問題，3249 筆座標幾秒鐘就能跑完。

如果之後想做更嚴謹的版本：parks_wikidata.csv 裡 osm_relation_id 有值的
園區，可以針對「這幾個特定 relation」個別打 Overpass 查詢實際多邊形邊界
（不是查全世界的 is_in，是查已知、範圍很小的特定 relation，請求量完全
不是同一個量級），再對這些園區做真正的 point-in-polygon 判斷，兩種方法
可以在論文裡互相佐證、報告吻合率。這支腳本先做距離門檻版本，量體小、
好驗證，是比較務實的第一步。
"""

import math

import pandas as pd

MATCH_RADIUS_M = 2000  # 跟 Phase 2 方法 B 的密度搜尋半徑保持一致，方便比較


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 公尺
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_nearest_park(lat, lon, parks_df):
    """對單一座標，算出跟所有園區的距離，回傳最近的一筆跟距離。"""
    dists = parks_df.apply(
        lambda p: haversine(lat, lon, p["lat"], p["lon"]), axis=1
    )
    idx = dists.idxmin()
    return parks_df.loc[idx, "name"], parks_df.loc[idx, "wikidata_id"], dists[idx]


def match_all(df_geocoded, df_parks):
    results = []
    valid = df_geocoded.dropna(subset=["Latitude", "Longitude"])

    for i, row in enumerate(valid.itertuples(), 1):
        name, wikidata_id, dist = find_nearest_park(row.Latitude, row.Longitude, df_parks)
        results.append({
            "Raw_Affiliation": row.Raw_Affiliation,
            "Latitude": row.Latitude,
            "Longitude": row.Longitude,
            "nearest_park_name": name,
            "nearest_park_wikidata_id": wikidata_id,
            "distance_to_park_m": round(dist, 1),
            "in_science_park_gt": dist <= MATCH_RADIUS_M,
        })
        if i % 500 == 0:
            print(f"已比對 {i}/{len(valid)}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_geocoded = pd.read_csv("geocoded.csv")
    df_parks = pd.read_csv("parks_wikidata.csv")
    print(f"機構座標 {len(df_geocoded)} 筆，權威園區清單 {len(df_parks)} 筆")

    df_matches = match_all(df_geocoded, df_parks)
    df_matches.to_csv("park_matches.csv", index=False, encoding="utf-8-sig")

    hit_count = df_matches["in_science_park_gt"].sum()
    print(f"\n落在權威園區 {MATCH_RADIUS_M}m 範圍內：{hit_count} 筆"
          f"（{hit_count / len(df_matches) * 100:.1f}%）")
    print("輸出：park_matches.csv")
