"""
把 fetch_park_polygons.py 抓到的精確園區邊界（park_polygons.json）套用到
park_matches.csv，對「最近的園區剛好是有精確邊界資料的那幾個」的機構，
做真正的 point-in-polygon 判斷，取代原本純距離門檻(2000m)的近似判定。

新增欄位：
  - in_park_polygon_precise：True/False（有精確邊界資料時的真實判定），
    沒有精確邊界資料的園區則是 None（沿用原本 in_science_park_gt 的
    距離門檻判定就好）。
  - in_park_best_guess：綜合兩個方法的最終判斷（見下方不對稱信任邏輯）。

park_polygons.json 現在存的是 **list of rings**（MultiPolygon），不是單一
多邊形——像新竹科學園區這種由好幾塊不相連基地組成的園區，落在「任一塊」
裡就算 True，不會再把不相干的分散基地硬湊成一個過度膨脹的凸包。

不對稱信任邏輯（重要）：fetch_park_polygons.py 現在只保留真正銜接成功的
封閉環，銜接不起來的碎片直接丟棄，不強制閉合、不算凸包——這代表我們手上
的多邊形資料**只可能少報，不可能多報**實際園區範圍。因此：
  - in_park_polygon_precise == True  → 可信，真的有完整的環包住這個點。
  - in_park_polygon_precise == False → **不代表確定在外面**，可能只是剛好
    落在被丟棄的殘缺資料範圍內（例如交大這個案例：新竹本部那塊的 way
    銜接不完整被丟棄，導致明明貼近園區卻查無多邊形資料）。
  - in_park_best_guess 的規則：精確多邊形說 True 就採信；說 False 或沒有
    資料時，一律退回距離門檻判定，不讓一個可能不完整的「False」推翻原本
    合理的距離判定。

不需要 shapely，point-in-polygon 用標準 ray casting 演算法手刻，這個
尺度(城市內範圍)不需要考慮地球曲率，平面近似完全足夠。
"""

import json

import pandas as pd


def point_in_ring(lon, lat, ring):
    """標準 ray casting 演算法。ring 是 [(lon, lat), ...] 的封閉環。"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            x_intersect = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def point_in_multipolygon(lon, lat, rings):
    """rings 是 list of ring，落在任一個 ring 裡就算 True。"""
    return any(point_in_ring(lon, lat, ring) for ring in rings)


if __name__ == "__main__":
    with open("park_polygons.json", "r", encoding="utf-8") as f:
        polygons = json.load(f)
    print(f"讀到 {len(polygons)} 個園區的精確邊界多邊形")

    df = pd.read_csv("park_matches.csv")

    precise_results = []
    changed_count = 0
    for row in df.itertuples():
        wikidata_id = row.nearest_park_wikidata_id
        if pd.isna(wikidata_id) or wikidata_id not in polygons:
            precise_results.append(None)
            continue

        rings = polygons[wikidata_id]
        result = point_in_multipolygon(row.Longitude, row.Latitude, rings)
        precise_results.append(result)

        if result != row.in_science_park_gt:
            changed_count += 1

    df["in_park_polygon_precise"] = precise_results
    # 不對稱信任：精確多邊形說 True 才採信，說 False/沒資料一律退回距離門檻
    df["in_park_best_guess"] = df["in_park_polygon_precise"].where(
        df["in_park_polygon_precise"] == True, df["in_science_park_gt"]
    )
    df.to_csv("park_matches.csv", index=False, encoding="utf-8-sig")

    n_precise = sum(1 for r in precise_results if r is not None)
    n_precise_true = sum(1 for r in precise_results if r is True)
    print(f"\n共 {n_precise} 筆機構的最近園區有精確邊界資料")
    print(f"精確判定為「真的在園區內」：{n_precise_true} 筆")
    print(f"跟原本距離門檻判定不一致的筆數：{changed_count}"
          f"（這些不是全部都會改變最終結果，只有精確判定為 True 的才會"
          f"覆蓋原本判定，說 False 的一律退回距離門檻，見 in_park_best_guess）")
    print("已更新 park_matches.csv（新增 in_park_polygon_precise / in_park_best_guess 欄位）")

    # 順便印出交大/清大的案例當驗證（如果有跑到）
    check = df[df["Raw_Affiliation"].str.contains("Chiao Tung|Tsing Hua", case=False, na=False)]
    if len(check) > 0:
        print("\n交大/清大驗證：")
        print(check[["Raw_Affiliation", "distance_to_park_m", "in_science_park_gt",
                      "in_park_polygon_precise", "in_park_best_guess"]]
              .drop_duplicates(subset=["Raw_Affiliation"]).to_string(index=False))
