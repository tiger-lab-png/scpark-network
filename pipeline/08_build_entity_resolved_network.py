"""
Phase 3 穩健性版本：節點用 OpenAlex 自己做過實體消歧的 Standardized_Institutions
欄位，取代原始地址字串（Raw_Affiliation）。

動機：raw affiliation 版本的網絡把同一個機構的不同地址寫法（例如
"Department of ..., University of Arkansas," 跟 "Dept. of ..., Univ. of
Arkansas, Fayetteville, AR" ）當成不同節點，會讓某些機構的 degree
被人為拉高/拆散，汙染「園區內 vs 園區外」的比較。這支腳本用標準化過的
機構名稱重建網絡，當作主要分析（原始字串版本改當穩健性檢查對照）。

機構層級的園區判定怎麼接：Standardized_Institutions 是從 Raw_Affiliation
衍生出來的，一個標準化機構名稱可能對應好幾筆不同的 raw affiliation（不同
系所、不同拼法）。做法：蒐集每個標準化機構名稱底下、所有對應 raw
affiliation 的園區判定，多數決（tie 保守判 False，不隨便判 True）。
座標同理，取第一筆有效座標（同機構不同校區座標本來就會有些微差異，這是
已知的簡化）。

輸出：
  - std_nodes.json / std_edges.json
  - std_map.html （沿用 map_tpl.html 樣板）
  - 終端機直接印出 degree/betweenness 的科學園區內外比較 + 檢定，
    方便直接跟 compare_park.py（raw affiliation 版）的結果對照。
"""

import json
import math
import statistics as st
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd

TEMPLATE_PATH = "map_tpl.html"
OUTPUT_HTML_PATH = "std_map.html"


def split_institutions(value):
    if pd.isna(value):
        return []
    return [s.strip() for s in str(value).split(",") if s.strip()]


def build_std_network(df_affil):
    G = nx.Graph()
    for paper_id, group in df_affil.dropna(subset=["Standardized_Institutions"]).groupby("Paper_ID"):
        insts = set()
        for val in group["Standardized_Institutions"]:
            insts.update(split_institutions(val))
        insts = sorted(insts)
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
    degree = dict(G.degree())
    if G.number_of_nodes() > 3000:
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


def parse_bool(v):
    """
    park_matches.csv 存過 CSV 之後，布林欄位讀回來常常是字串 "True"/"False"，
    不是真的 Python bool——bool("False") 在 Python 裡是 True（非空字串都是
    truthy），這是之前踩到的一個真實 bug。這裡統一、明確地解析，順便處理
    偶爾出現的殘缺值（例如曾經看過整格被截斷成單一字元 "F" 的情況），
    看不懂的一律當 False（保守，不隨便判 True）。
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip().lower()
    if s in ("true", "t", "1"):
        return True
    return False  # 包含 "false"/"f"/"0"/空字串/看不懂的殘缺值，一律 False


def build_std_to_geo(df_affil, df_park):
    """
    標準化機構名稱 -> (多數決園區判定, 第一筆有效座標, 最常見園區名稱)。
    """
    raw_to_std = df_affil[["Raw_Affiliation", "Standardized_Institutions"]].dropna().drop_duplicates()
    park_lookup = df_park.set_index("Raw_Affiliation").to_dict(orient="index")

    verdict_col = "in_park_best_guess" if "in_park_best_guess" in df_park.columns else "in_science_park_gt"

    std_verdicts = defaultdict(list)
    std_coords = defaultdict(list)
    std_park_names = defaultdict(list)

    for row in raw_to_std.itertuples():
        geo = park_lookup.get(row.Raw_Affiliation)
        if geo is None:
            continue
        for std_name in split_institutions(row.Standardized_Institutions):
            verdict = geo.get(verdict_col)
            if verdict is not None:
                std_verdicts[std_name].append(parse_bool(verdict))
            lat, lon = geo.get("Latitude"), geo.get("Longitude")
            if lat is not None and lon is not None and not (isinstance(lat, float) and math.isnan(lat)):
                std_coords[std_name].append((lat, lon))
            park_name = geo.get("nearest_park_name")
            if isinstance(park_name, str):
                std_park_names[std_name].append(park_name)

    result = {}
    for std_name in set(list(std_verdicts.keys()) + list(std_coords.keys())):
        verdicts = std_verdicts.get(std_name, [])
        majority = (sum(verdicts) > len(verdicts) / 2) if verdicts else False
        coords = std_coords.get(std_name, [])
        lat, lon = coords[0] if coords else (None, None)
        names = std_park_names.get(std_name, [])
        top_name = Counter(names).most_common(1)[0][0] if names else None
        result[std_name] = {"in_science_park": majority, "lat": lat, "lon": lon, "park_name": top_name}
    return result


def assemble_nodes(G, metrics, std_geo):
    nodes = []
    for inst in G.nodes():
        geo = std_geo.get(inst, {})
        lat, lon = geo.get("lat"), geo.get("lon")
        if lat is None or lon is None:
            continue
        m = metrics[inst]
        nodes.append({
            "id": inst,
            "name": inst[:80],
            "lat": lat,
            "lon": lon,
            "degree": m["degree"],
            "betweenness": m["betweenness"],
            "community": m["community"],
            "in_science_park": bool(geo.get("in_science_park")),
            "park_name": geo.get("park_name"),
            "method_used": "wikidata_gt_std_institution",
        })
    return nodes


def assemble_edges(G, nodes):
    valid_ids = {n["id"] for n in nodes}
    return [{"source": u, "target": v, "weight": d["weight"]}
            for u, v, d in G.edges(data=True) if u in valid_ids and v in valid_ids]


def generate_html(nodes, edges, template_path=TEMPLATE_PATH, out_path=OUTPUT_HTML_PATH):
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__NODES_JSON__", json.dumps(nodes, ensure_ascii=False))
    html = html.replace("__EDGES_JSON__", json.dumps(edges, ensure_ascii=False))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def compare_park_vs_nonpark(nodes):
    park = [n for n in nodes if n["in_science_park"]]
    non_park = [n for n in nodes if not n["in_science_park"]]

    def summarize(group, label):
        deg = [n["degree"] for n in group]
        bet = [n["betweenness"] for n in group]
        print(f"{label}：n={len(group)}")
        print(f"  degree      平均={st.mean(deg):.2f}　中位數={st.median(deg):.1f}")
        print(f"  betweenness 平均={st.mean(bet):.6f}　中位數={st.median(bet):.6f}")

    summarize(park, "科學園區內機構（標準化版）")
    summarize(non_park, "非科學園區機構（標準化版）")

    try:
        from scipy.stats import mannwhitneyu
        _, p_deg = mannwhitneyu([n["degree"] for n in park], [n["degree"] for n in non_park], alternative="two-sided")
        _, p_bet = mannwhitneyu([n["betweenness"] for n in park], [n["betweenness"] for n in non_park], alternative="two-sided")
        print("\nMann-Whitney U 檢定（標準化機構版）：")
        print(f"  degree:      p = {p_deg:.4g}")
        print(f"  betweenness: p = {p_bet:.4g}")
    except ImportError:
        print("\n沒裝 scipy，跳過檢定。安裝：pip install scipy")


if __name__ == "__main__":
    df_affil = pd.read_csv("affil.csv")
    df_park = pd.read_csv("park_matches.csv")

    G = build_std_network(df_affil)
    print(f"標準化機構網絡：{G.number_of_nodes()} 節點、{G.number_of_edges()} 邊"
          f"（對照 raw affiliation 版本的節點數，差距越大代表原本實體消歧問題越嚴重）")

    std_geo = build_std_to_geo(df_affil, df_park)
    metrics = compute_sna_metrics(G)
    nodes = assemble_nodes(G, metrics, std_geo)
    edges = assemble_edges(G, nodes)

    with open("std_nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open("std_edges.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)
    generate_html(nodes, edges)
    print(f"可視化節點：{len(nodes)} 個，輸出：std_nodes.json / std_edges.json / std_map.html\n")

    compare_park_vs_nonpark(nodes)
