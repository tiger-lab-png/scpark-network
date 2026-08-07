"""
Build the entity-resolved co-affiliation network.

Nodes are OpenAlex's disambiguated Standardized_Institutions names, so the many
spellings of one institution collapse into a single node; 09 builds the
raw-string counterpart for comparison.

Reads:  affil.csv or affil_full.csv (step 01) and park_matches.csv (steps 05-06).
Writes: std_nodes.json / std_edges.json, optionally std_map.html when the
        map_tpl.html template is present, and prints the park vs non-park
        comparison of degree and betweenness.

A standardised name can map to several raw affiliations (different departments
or spellings), so its park verdict is a majority vote over those rows, with ties
resolved to False, and its coordinate is the first valid one found.
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
    """
    Connect every pair of institutions appearing on the same paper; `weight` is
    the number of co-authored papers.

    `inv_weight` (= 1/weight) is added because NetworkX treats a weight as a
    distance, where a larger value means a costlier edge. Betweenness must use
    inv_weight so that a more frequent collaboration is a cheaper path, which is
    what a co-occurrence count as an inverse path cost means; `weight` itself is
    kept unchanged for degree and for output.
    """
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
    for u, v, d in G.edges(data=True):
        d["inv_weight"] = 1.0 / d["weight"]
    return G


def compute_sna_metrics(G):
    degree = dict(G.degree())
    if G.number_of_nodes() > 3000:
        # exact betweenness is infeasible on large graphs; sample pivots instead
        betweenness = nx.betweenness_centrality(G, k=500, weight="inv_weight", seed=42)
    else:
        betweenness = nx.betweenness_centrality(G, weight="inv_weight")
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
    Parse the boolean columns of park_matches.csv explicitly.

    After a CSV round trip these arrive as the strings "True"/"False", and
    bool("False") is True in Python because any non-empty string is truthy.
    Truncated or unrecognised values are read as False, never as True.
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
    return False


def build_std_to_geo(df_affil, df_park):
    """
    Standardised institution name -> (majority park verdict, first valid
    coordinate, most frequent park name).
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
        majority = (sum(verdicts) > len(verdicts) / 2) if verdicts else False  # ties -> False
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
            continue  # no coordinate, so no micro-geographic classification
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
    """
    Render the optional interactive map. A missing template must not abort the
    script after the JSON outputs and the statistics have been produced, so the
    failure is caught and reported.
    """
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        print(f"template {template_path} not found, skipping the map "
              f"(std_nodes.json / std_edges.json and the statistics below are unaffected)")
        return
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
        print(f"{label}: n={len(group)}")
        print(f"  degree      mean={st.mean(deg):.2f}  median={st.median(deg):.1f}")
        print(f"  betweenness mean={st.mean(bet):.6f}  median={st.median(bet):.6f}")

    summarize(park, "park institutions (entity-resolved)")
    summarize(non_park, "non-park institutions (entity-resolved)")

    try:
        from scipy.stats import mannwhitneyu
        _, p_deg = mannwhitneyu([n["degree"] for n in park], [n["degree"] for n in non_park], alternative="two-sided")
        _, p_bet = mannwhitneyu([n["betweenness"] for n in park], [n["betweenness"] for n in non_park], alternative="two-sided")
        print("\nMann-Whitney U (entity-resolved):")
        print(f"  degree:      p = {p_deg:.4g}")
        print(f"  betweenness: p = {p_bet:.4g}")
    except ImportError:
        print("\nscipy not installed, tests skipped. Install with: pip install scipy")


def _load_affil_csv():
    """Accept either filename so the same script serves both dataset sizes."""
    import os
    for candidate in ("affil.csv", "affil_full.csv"):
        if os.path.exists(candidate):
            print(f"reading {candidate}")
            return pd.read_csv(candidate)
    raise FileNotFoundError("neither affil.csv nor affil_full.csv found in the working directory.")


if __name__ == "__main__":
    df_affil = _load_affil_csv()
    df_park = pd.read_csv("park_matches.csv")

    G = build_std_network(df_affil)
    print(f"entity-resolved network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(compare with the raw-string network from 09: the larger the gap, the "
          f"more name variation the raw strings contain)")

    std_geo = build_std_to_geo(df_affil, df_park)
    metrics = compute_sna_metrics(G)
    nodes = assemble_nodes(G, metrics, std_geo)
    edges = assemble_edges(G, nodes)

    with open("std_nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open("std_edges.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)
    generate_html(nodes, edges)
    print(f"{len(nodes)} mappable nodes; output: std_nodes.json / std_edges.json / std_map.html\n")

    compare_park_vs_nonpark(nodes)
