"""
單獨重跑 Method B（Overpass 密度特徵）這一步，跳過已經完成的 Nominatim
地理編碼（geocoded.csv 已存在，不用重打 Nominatim）。

直接重用 phase2_geocoding_osm.py 裡寫好的 enrich_addresses()，邏輯完全
沒有改動——多鏡像站輪替、429 退避、座標層級快取、可中斷續跑都還在。
可以放心 Ctrl+C 中斷，重跑這支腳本會從 overpass_cache.json 已經完成的
座標繼續，不會整批重來。
"""

import pandas as pd

from phase2_geocoding_osm import enrich_addresses

if __name__ == "__main__":
    df_geocoded = pd.read_csv("geocoded.csv")
    print(f"讀取 geocoded.csv：{len(df_geocoded)} 筆地址\n")

    df_enriched = enrich_addresses(df_geocoded)
    df_enriched.to_csv("enriched.csv", index=False, encoding="utf-8-sig")

    hit_a = (df_enriched["method_used"] == "A_polygon").sum()
    hit_b = (df_enriched["method_used"] == "B_density_fallback").sum()
    no_coord = (df_enriched["method_used"] == "no_coordinates").sum()
    overpass_failed = (df_enriched["method_used"] == "overpass_failed").sum()
    print(f"\n方法 A 命中（既有園區多邊形）：{hit_a} 筆")
    print(f"方法 B fallback（密度指標估計）：{hit_b} 筆")
    print(f"沒有座標，無法分類（geocoding 失敗）：{no_coord} 筆")
    print(f"有座標但 Overpass 查詢失敗（重跑會自動補）：{overpass_failed} 筆")
    print("\n輸出：enriched.csv")
