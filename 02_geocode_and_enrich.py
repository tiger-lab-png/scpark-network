"""
Build the raw-string co-affiliation network.

Nodes are the unmodified Raw_Affiliation strings, so the same institution can
appear under several spellings. This is the naive node-identity scheme against
which the entity-resolved network of step 08 is compared.

Reads:  affil.csv or affil_full.csv (step 01) and park_matches.csv (steps 05-06).
Writes: nodes.json / edges.json, and network_map.html when the map_tpl.html
        template is present.

Edges connect institutions co-occurring on a paper, weighted by the number of
such papers; nodes without coordinates are dropped from the output.
"""

import json
import math

import networkx as nx
import pandas as pd

TEMPLATE_PATH = "map_tpl.html"
OUTPUT_HTML_PATH = "network_map.html"


def build_institution_network(df_affil):
    """Connect every pair of institutions appearing on the same paper."""
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

    # As in step 08: NetworkX reads a weight as a distance, so betweenness uses
    # inv_weight (a frequent collaboration is a cheap path) while weight itself
    # keeps its co-occurrence-count meaning for degree and for output.
    for u, v, d in G.edges(data=True):
        d["inv_weight"] = 1.0 / d["weight"]

    return G


def compute_sna_metrics(G):
    """Return {node: {degree, betweenness, community}}."""
    degree = dict(G.degree())
    if G.number_of_nodes() > 3000:
        # Sampled betweenness on large graphs. The raw-string network has far
        # more nodes than the entity-resolved one, and greedy modularity
        # community detection on it is usually the slowest step of the pipeline.
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


def _clean(v):
    """pandas reads empty cells as float('nan'); normalise to None for JSON."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def assemble_nodes(G, metrics, df_geo):
    """
    Join the network metrics with the geographic classification from
    park_matches.csv, preferring in_park_best_guess (distance rule refined by the
    asymmetric polygon test) and falling back to the plain distance verdict when
    step 06 has not been run.
    """
    geo_lookup = df_geo.set_index("Raw_Affiliation").to_dict(orient="index")
    has_best_guess = "in_park_best_guess" in df_geo.columns

    nodes = []
    for inst in G.nodes():
        geo = geo_lookup.get(inst, {})
        lat, lon = geo.get("Latitude"), geo.get("Longitude")
        if lat is None or lon is None or (isinstance(lat, float) and math.isnan(lat)):
            continue  # nodes without coordinates cannot be mapped or classified

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
    """A missing map template must not discard the JSON outputs already written."""
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            html = f.read()
    except FileNotFoundError:
        print(f"template {template_path} not found, skipping the map "
              f"(nodes.json / edges.json are unaffected)")
        return
    html = html.replace("__NODES_JSON__", json.dumps(nodes, ensure_ascii=False))
    html = html.replace("__EDGES_JSON__", json.dumps(edges, ensure_ascii=False))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def _load_affil_csv():
    """Accept either filename so the same script serves both dataset sizes."""
    import os
    for candidate in ("affil.csv", "affil_full.csv"):
        if os.path.exists(candidate):
            print(f"reading {candidate}")
            return pd.read_csv(candidate)
    raise FileNotFoundError("neither affil.csv nor affil_full.csv found.")


if __name__ == "__main__":
    df_affil = _load_affil_csv()
    df_geo = pd.read_csv("park_matches.csv")

    G = build_institution_network(df_affil)
    print(f"raw-string network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    metrics = compute_sna_metrics(G)
    nodes = assemble_nodes(G, metrics, df_geo)
    edges = assemble_edges(G, nodes)

    with open("nodes.json", "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open("edges.json", "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    generate_html(nodes, edges)
    print(f"done: {len(nodes)} mappable nodes, {len(edges)} edges")
    print("output: nodes.json / edges.json / network_map.html")
