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

import numpy as np
import pandas as pd

MATCH_RADIUS_M = 2000  # 跟 Phase 2 方法 B 的密度搜尋半徑保持一致，方便比較


def haversine(lat1, lon1, lat2, lon2):
    """純量版本，保留給其他地方需要單點計算時使用；批次比對改用下面的
    向量化版本，語意（球面距離公式）完全一樣，只是同時對整個陣列算。"""
    R = 6371000  # 公尺
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def match_all(df_geocoded, df_parks):
    """
    2026-07-29 向量化重寫：原本的寫法是「每一筆機構座標」都跑一次
    parks_df.apply(...)（231 個園區逐一算距離），外層再用 Python for 迴圈
    跑過所有機構座標——在 5,000 篇論文、約 9,440 筆有效座標的規模下幾秒鐘
    能跑完，但 55,000 篇論文規模下有效座標膨脹到近 7 萬筆，等於要跑
    7 萬 × 231 次「逐列 pandas apply」，實測這種寫法在 pandas 裡開銷很大，
    很容易拖到數十分鐘。改成 numpy 廣播：把「每個機構」跟「每個園區」的
    距離一次算成一個 (機構數 × 231) 的矩陣，數學上跟原本逐筆算 haversine
    完全等價，只是不用 Python 層級的逐列迴圈，通常幾秒內就能跑完全部。
    """
    valid = df_geocoded.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    R = 6371000.0
    lat1 = np.radians(valid["Latitude"].to_numpy())[:, None]      # (N, 1)
    lon1 = np.radians(valid["Longitude"].to_numpy())[:, None]     # (N, 1)
    lat2 = np.radians(df_parks["lat"].to_numpy())[None, :]        # (1, M)
    lon2 = np.radians(df_parks["lon"].to_numpy())[None, :]        # (1, M)

    dphi = lat2 - lat1
    dlambda = lon2 - lon1
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2) ** 2
    dist_matrix = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))    # (N, M)，跟純量版 haversine 數學等價

    nearest_idx = np.argmin(dist_matrix, axis=1)
    nearest_dist = dist_matrix[np.arange(len(valid)), nearest_idx]

    results = pd.DataFrame({
        "Raw_Affiliation": valid["Raw_Affiliation"],
        "Latitude": valid["Latitude"],
        "Longitude": valid["Longitude"],
        "nearest_park_name": df_parks["name"].to_numpy()[nearest_idx],
        "nearest_park_wikidata_id": df_parks["wikidata_id"].to_numpy()[nearest_idx],
        "distance_to_park_m": np.round(nearest_dist, 1),
        "in_science_park_gt": nearest_dist <= MATCH_RADIUS_M,
    })
    return results


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
