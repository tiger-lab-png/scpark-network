"""
合併方法 A（Wikidata 權威清單，距離門檻判定）跟方法 B（Overpass is_in
多邊形 + 密度特徵）成一份完整表，兩種「是否落在科學園區」的判定並列，
方便論文報告兩者吻合率。

輸入：
  - enriched.csv       （phase2，含 OSM/Overpass 版本的 in_science_park、
                          method_used，以及密度特徵 univ_research_count/
                          nearest_station_m/nearest_junction_m）
  - park_matches.csv   （match_parks.py，含 Wikidata 權威清單版本的
                          in_science_park_gt、nearest_park_name、距離）

合併方式：以 Raw_Affiliation 為 key 做 left join（enriched.csv 為主表，
保留全部 10404 筆，包含沒有座標的失敗案例）。addr_uniq.csv 本身就是去重
後的地址清單，Raw_Affiliation 在兩份輸入裡都是唯一值，不會有 join 爆量
的問題。

輸出：combined.csv，欄位命名規則：
  - *_osm  結尾：方法 B 用的 Overpass/OSM 判定（in_science_park_osm、
    park_name_osm、method_used_osm）
  - *_gt   結尾：方法 A 用的 Wikidata 權威清單判定（in_science_park_gt、
    park_name_gt、distance_to_park_m）
  - agree：兩個方法都有值的情況下，判定是否一致（只看 True/False 落在
    園區與否，不要求配對到同一個園區名稱——名稱本來就可能不同來源命名
    不一樣）
"""

import pandas as pd

if __name__ == "__main__":
    df_enriched = pd.read_csv("enriched.csv")
    df_gt = pd.read_csv("park_matches.csv")

    df_enriched = df_enriched.rename(columns={
        "in_science_park": "in_science_park_osm",
        "park_name": "park_name_osm",
        "method_used": "method_used_osm",
    })

    df_gt_slim = df_gt[[
        "Raw_Affiliation", "nearest_park_name", "nearest_park_wikidata_id",
        "distance_to_park_m", "in_science_park_gt",
    ]].rename(columns={"nearest_park_name": "park_name_gt"})

    df_combined = df_enriched.merge(df_gt_slim, on="Raw_Affiliation", how="left")

    both_known = df_combined["in_science_park_osm"].notna() & df_combined["in_science_park_gt"].notna()
    df_combined["agree"] = None
    df_combined.loc[both_known, "agree"] = (
        df_combined.loc[both_known, "in_science_park_osm"]
        == df_combined.loc[both_known, "in_science_park_gt"]
    )

    df_combined.to_csv("combined.csv", index=False, encoding="utf-8-sig")

    n_both = both_known.sum()
    n_agree = (df_combined.loc[both_known, "agree"] == True).sum()
    n_gt_true = (df_combined["in_science_park_gt"] == True).sum()
    n_osm_true = (df_combined["in_science_park_osm"] == True).sum()
    n_both_true = ((df_combined["in_science_park_gt"] == True)
                   & (df_combined["in_science_park_osm"] == True)).sum()
    n_gt_only = ((df_combined["in_science_park_gt"] == True)
                 & (df_combined["in_science_park_osm"] == False)).sum()
    n_osm_only = ((df_combined["in_science_park_osm"] == True)
                  & (df_combined["in_science_park_gt"] == False)).sum()

    print(f"合併完成：combined.csv，共 {len(df_combined)} 筆")
    print(f"\n兩個方法都有有效判定的筆數：{n_both}")
    print(f"判定一致（True/False 相同）：{n_agree} 筆（{n_agree / n_both * 100:.1f}%）" if n_both else "")
    print(f"\n方法 A（Wikidata GT）判定落在園區：{n_gt_true} 筆")
    print(f"方法 B（Overpass OSM）判定落在園區：{n_osm_true} 筆")
    print(f"兩者都判定落在園區（交集）：{n_both_true} 筆")
    print(f"只有方法 A 判定落在園區（GT 有但 OSM 沒標記/查無多邊形）：{n_gt_only} 筆")
    print(f"只有方法 B 判定落在園區（OSM 有標記但不在 Wikidata 清單裡）：{n_osm_only} 筆")
    print("\n提醒：這個吻合率本身就是論文可以報告的一個數字——不一致的案例"
          "通常代表 OSM 有標記但 Wikidata 沒收錄的新興/較小園區，或反過來"
          "Wikidata 收錄了但 OSM 志工還沒把邊界畫出來，兩者恰好互補，這正是"
          "當初決定『Wikidata 當 ground truth、OSM 當輔助』的理由。")
