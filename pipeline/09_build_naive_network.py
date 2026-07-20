"""
Phase 3: 機構層級共著網絡(SNA)+ Leaflet/D3 HGIS 視覺化
------------------------------------------------------------------
輸入：
  - affil.csv          （Phase 1 產出，Paper_ID / Raw_Affiliation 對照）
  - park_matches.csv   （Phase 2 產出，Raw_Affiliation 對照經緯度與方法 A/Wikidata
                          園區判定。方法 B/enriched.csv 目前卡在 Overpass 自架匯入，
                          等它跑完再切換或合併，先用方法 A 的結果讓 Phase 3 跑完整。）
輸出：
  - nodes.json / edges.json       （網絡資料，供其他工具重用）
  - network_map.html              （單一檔案、可直接雙擊開啟的 Leaflet+D3 視覺化,
                                     由 network_map_template.html 套入資料產生）

設計說明：
1. 網絡節點是「機構」而不是「作者」。作者層級的共著網絡雜訊太大（同機構
   內部作者互相掛名會製造大量無意義的高權重邊），機構層級才符合「地理鄰近性」
   這個研究問題的顆粒度。
2. 節點 key 用 Raw_Affiliation 原始字串，跟 Phase 2 地理編碼的 key 完全對齊。
   機構層級的實體消歧（同機構、地址拼法不同）建議之後再另外處理。
3. 邊的權重 = 兩機構在多少篇論文中同時出現。
4. 中心性與社群偵測用 networkx 內建演算法，不額外依賴 python-louvain。
5. HTML 視覺化的樣板獨立成 network_map_template.html，本腳本只負責把
   nodes/edges 的 JSON 套進去，避免單一檔案過大、也方便你之後直接改樣板
   而不用碰 Python 邏輯。
"""

import json
import math

import networkx as nx
import pandas as pd

TEMPLATE_PATH = "map_tpl.html"  # 原本叫 network_map_template.html，檔名太長
# + 深層路徑加起來超過 Windows 260 字元限制（就是你看到 NETWOR~2.HTM 短檔名
# 的那個問題），改短名
OUTPUT_HTML_PATH = "network_map.html"


def build_institution_network(df_affil):
    """以 Paper_ID 為單位，把同一篇論文裡出現過的所有機構兩兩連邊。"""
    G = nx.Graph()

    for paper_id, group in df_affil.dropna(subset=["Raw_Affiliation"]).groupby("Paper_ID"):
        insts = sorted(set(group["Raw_Affiliation"]))
        for inst in insts:
            G.add_node(inst)
        for i in range(len(insts)):
            for j in range(i + 1, len(insts)):
                a, b = insts[i], insts[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    return G


def compute_sna_metrics(G):
    """回傳 dict: {node: {degree, betweenness, community}}"""
    degree = dict(G.degree())
    if G.number_of_nodes() > 3000:
        # 大圖用抽樣版本節省時間
        betweenness = nx.betweenness_centrality(G, k=500, weight="weight", seed=42)
    else:
        betweenness = nx.betweenness_centrality(G, weight="weight")

    communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    community_map = {}
    for cid, members in enumerate(communities):
        for m in members:
            community_map[m] = cid

    return {
        node: {
            "degree": degree.get(node, 0),
            "betweenness": round(betweenness.get(node, 0.0), 6),
            "community": community_map.get(node, -1),
        }
        for node in G.nodes()
    }


def _clean(v):
    """pandas 讀空值時是 float('nan')，統一轉成 None 避免 JSON/JS 混淆。"""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def assemble_nodes(G, metrics, df_geo):
    """
    把 SNA 指標跟 Phase 2 的地理/園區特徵合併成節點清單。
    df_geo 目前是 park_matches.csv（方法 A / Wikidata 權威清單版）。優先用
    in_park_best_guess（distance threshold + 精確多邊形不對稱信任後的最終
    判定，沒有這欄——例如還沒跑 apply_polygons.py——就退回 in_science_park_gt
    純距離門檻）。等方法 B (Overpass) 哪天跑完、跟方法 A merge 成
    combined.csv 之後，這裡要跟著改成讀 combined.csv 並改用 *_gt / *_osm
    欄位名。
    """
    geo_lookup = df_geo.set_index("Raw_Affiliation").to_dict(orient="index")
    has_best_guess = "in_park_best_guess" in df_geo.columns

    nodes = []
    for inst in G.nodes():
        geo = geo_lookup.get(inst, {})
        lat, lon = geo.get("Latitude"), geo.get("Longitude")
        if lat is None or lon is None or (isinstance(lat, float) and math.isnan(lat)):
            continue  # 沒地理座標的節點無法畫在地圖上，跳過

        m = metrics[inst]
        park_name = _clean(geo.get("nearest_park_name"))
        distance_m = _clean(geo.get("distance_to_park_m"))
        if has_best_guess:
            raw_in_park = _clean(geo.get("in_park_best_guess"))
        else:
            raw_in_park = _clean(geo.get("in_science_park_gt"))

        nodes.append({
            "id": inst,
            "name": inst[:80],
            "lat": lat,
            "lon": lon,
            "degree": m["degree"],
            "betweenness": m["betweenness"],
            "community": m["community"],
            "in_science_park": bool(raw_in_park) if raw_in_park is not None else False,
            "park_name": park_name,
            "distance_to_park_m": distance_m,
            "method_used": "wikidata_gt_plus_polygon" if has_best_guess else "wikidata_gt_distance_only",
        })
    return nodes


def assemble_edges(G, nodes):
    valid_ids = {n["id"] for n in nodes}
    edges = []
    for u, v, data in G.edges(data=True):
        if u in valid_ids and v in valid_ids:
            edges.append({"source": u, "target": v, "weight": data["weight"]})
    return edges


def generate_html(nodes, edges, template_path=TEMPLATE_PATH, out_path=OUTPUT_HTML_PATH):
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__NODES_JSON__", json.dumps(nodes, ensure_ascii=False))
    html = html.replace("__EDGES_JSON__", json.dumps(edges, ensure_ascii=False))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    # 檔名對齊 Phase 1 / Phase 2 縮短後的輸出，理由是深層 Windows 路徑容易
    # 超過 260 字元導致 FileNotFoundError。方法 B（enriched.csv）還在等
    # 自架 Overpass 跑完，先用方法 A（park_matches.csv，Wikidata 權威清單）
    # 讓 Phase 3 產出完整成果。
    df_affil = pd.read_csv("affil.csv")
    df_geo = pd.read_csv("park_matches.csv")

    G = build_institution_network(df_affil)
    print(f"機構網絡：{G.number_of_nodes()} 節點、{G.number_of_edges()} 邊")

    metrics = compute_sna_metrics(G)
    nodes = assemble_nodes(G, metrics, df_geo)
    edges = assemble_edges(G, nodes)

    with open("nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open("edges.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    generate_html(nodes, edges)
    print(f"完成：{len(nodes)} 個可視化節點（有地理座標）、{len(edges)} 條邊")
    print("輸出：nodes.json / edges.json / network_map.html")
