"""
Supplementary robustness checks that extend 10_robustness_checks.py.

Reads:  affil_full.csv, park_matches.csv, combined.csv, geocoded.csv,
        parks_wikidata.csv, std_nodes.json.
Writes: nothing; all results are printed.

Four checks, run directly on the data rather than read off earlier logs:
  A  distance-threshold sensitivity for betweenness as well as degree;
  B  leave-Hsinchu-out for betweenness as well as degree;
  C  exclusion of two registry entries treated as false-positive parks, with the
     nearest-park matching redone (only the park labels change, not the network);
  D  exclusion of hyper-authorship papers at 6, 8 and 15 institutions per paper,
     which does change the topology, so the network is rebuilt and degree
     recomputed.
Set before running: the radii in section A, the excluded park names in section C
and the institution-count thresholds in section D.
"""
import json
import math
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

SEARCH_RADIUS_M = 4000  # imputed distance when no station/junction was found


def split_institutions(value):
    if pd.isna(value):
        return []
    return [s.strip() for s in str(value).split(",") if s.strip()]


def build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000):
    """Same aggregation as 10_robustness_checks.build_std_lookup(); duplicated
    here so this script can be run on its own."""
    raw_to_std = df_affil[["Paper_ID", "Raw_Affiliation", "Standardized_Institutions"]].dropna(
        subset=["Raw_Affiliation", "Standardized_Institutions"]
    )
    park_lookup = df_park.set_index("Raw_Affiliation").to_dict(orient="index")
    density_lookup = df_combined.set_index("Raw_Affiliation")[
        ["univ_research_count", "nearest_station_m", "nearest_junction_m"]
    ].to_dict(orient="index") if df_combined is not None else {}

    std_verdicts = defaultdict(list)
    std_park_names = defaultdict(list)
    std_univ = defaultdict(list)
    std_station = defaultdict(list)
    std_junction = defaultdict(list)
    std_papers = defaultdict(set)

    raw_verdict_cache = {}
    for raw_aff, row in park_lookup.items():
        dist = row.get("distance_to_park_m")
        if dist is None or (isinstance(dist, float) and math.isnan(dist)):
            raw_verdict_cache[raw_aff] = None
            continue
        raw_verdict_cache[raw_aff] = {"in_park": dist <= match_radius_m, "park_name": row.get("nearest_park_name")}

    for row in raw_to_std.itertuples():
        raw_aff = row.Raw_Affiliation
        v = raw_verdict_cache.get(raw_aff)
        if v is not None:
            for std_name in split_institutions(row.Standardized_Institutions):
                std_verdicts[std_name].append(v["in_park"])
                if v["in_park"]:
                    std_park_names[std_name].append(v["park_name"])
        d = density_lookup.get(raw_aff)
        if d is not None:
            for std_name in split_institutions(row.Standardized_Institutions):
                if d.get("univ_research_count") is not None and not (isinstance(d["univ_research_count"], float) and math.isnan(d["univ_research_count"])):
                    std_univ[std_name].append(d["univ_research_count"])
                if d.get("nearest_station_m") is not None and not (isinstance(d["nearest_station_m"], float) and math.isnan(d["nearest_station_m"])):
                    std_station[std_name].append(d["nearest_station_m"])
                if d.get("nearest_junction_m") is not None and not (isinstance(d["nearest_junction_m"], float) and math.isnan(d["nearest_junction_m"])):
                    std_junction[std_name].append(d["nearest_junction_m"])
        for std_name in split_institutions(row.Standardized_Institutions):
            std_papers[std_name].add(row.Paper_ID)

    result = {}
    all_names = set(std_verdicts) | set(std_univ) | set(std_papers)
    for name in all_names:
        verdicts = std_verdicts.get(name, [])
        majority = (sum(verdicts) > len(verdicts) / 2) if verdicts else False  # ties -> False
        names = std_park_names.get(name, [])
        top_name = max(set(names), key=names.count) if names else None
        result[name] = {
            "in_science_park": majority,
            "park_name": top_name,
            "univ_research_count": std_univ[name][0] if std_univ.get(name) else None,
            "nearest_station_m": std_station[name][0] if std_station.get(name) else None,
            "nearest_junction_m": std_junction[name][0] if std_junction.get(name) else None,
            "paper_count": len(std_papers.get(name, set())),
        }
    return result


def build_regression_df(nodes, std_lookup):
    rows = []
    for n in nodes:
        d = std_lookup.get(n["id"])
        if d is None or d["univ_research_count"] is None:
            continue
        rows.append({
            "id": n["id"],
            "degree": n["degree"],
            "betweenness": n.get("betweenness", np.nan),
            "in_science_park": int(bool(d["in_science_park"])),
            "park_name": d["park_name"],
            "univ_research_count": d["univ_research_count"],
            "nearest_station_m": d["nearest_station_m"] if d["nearest_station_m"] is not None else SEARCH_RADIUS_M,
            "nearest_junction_m": d["nearest_junction_m"] if d["nearest_junction_m"] is not None else SEARCH_RADIUS_M,
            "paper_count": d["paper_count"],
        })
    df = pd.DataFrame(rows)
    df["log_univ"] = np.log1p(df["univ_research_count"])
    df["log_station"] = np.log1p(df["nearest_station_m"])
    df["log_junction"] = np.log1p(df["nearest_junction_m"])
    df["log_papers"] = np.log1p(df["paper_count"])
    return df


def match_all(df_geocoded, df_parks, match_radius_m=2000):
    """Vectorised nearest-park matching, as in pipeline/05_match_parks_distance.py,
    so section C can rematch against a reduced park list without re-running it."""
    valid = df_geocoded.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)
    R = 6371000.0
    lat1 = np.radians(valid["Latitude"].to_numpy())[:, None]
    lon1 = np.radians(valid["Longitude"].to_numpy())[:, None]
    lat2 = np.radians(df_parks["lat"].to_numpy())[None, :]
    lon2 = np.radians(df_parks["lon"].to_numpy())[None, :]
    dphi = lat2 - lat1
    dlambda = lon2 - lon1
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2) ** 2
    dist_matrix = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    nearest_idx = np.argmin(dist_matrix, axis=1)
    nearest_dist = dist_matrix[np.arange(len(valid)), nearest_idx]
    return pd.DataFrame({
        "Raw_Affiliation": valid["Raw_Affiliation"],
        "nearest_park_name": df_parks["name"].to_numpy()[nearest_idx],
        "distance_to_park_m": np.round(nearest_dist, 1),
        "in_science_park_gt": nearest_dist <= match_radius_m,
    })


print("loading data...")
df_affil = pd.read_csv("affil_full.csv")
df_park = pd.read_csv("park_matches.csv")
df_combined = pd.read_csv("combined.csv")
df_geocoded = pd.read_csv("geocoded.csv")
df_parks_gt = pd.read_csv("parks_wikidata.csv")
nodes = json.load(open("std_nodes.json", encoding="utf-8"))
print(f"std_nodes.json: {len(nodes)} nodes\n")

# ============================================================
# [A] distance-threshold sensitivity for degree and betweenness
# ============================================================
print("=" * 70)
print("[A] distance-threshold sensitivity: degree and betweenness")
print("=" * 70)
radii = [500, 1000, 2000, 3000, 5000]
results_a = []
for r in radii:
    std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=r)
    df = build_regression_df(nodes, std_lookup)
    n_park = int(df["in_science_park"].sum())
    n_nonpark = len(df) - n_park
    if n_park < 5:
        print(f"threshold {r} m: only {n_park} park institutions, skipped")
        continue
    park_deg = df.loc[df["in_science_park"] == 1, "degree"]
    nonpark_deg = df.loc[df["in_science_park"] == 0, "degree"]
    _, p_mw_deg = stats.mannwhitneyu(park_deg, nonpark_deg, alternative="two-sided")

    park_bet = df.loc[df["in_science_park"] == 1, "betweenness"]
    nonpark_bet = df.loc[df["in_science_park"] == 0, "betweenness"]
    _, p_mw_bet = stats.mannwhitneyu(park_bet, nonpark_bet, alternative="two-sided")

    model = smf.negativebinomial(
        "degree ~ in_science_park + log_univ + log_station + log_junction", data=df
    ).fit(disp=0)
    coef = model.params["in_science_park"]
    p_reg = model.pvalues["in_science_park"]
    irr = math.exp(coef)

    print(f"threshold {r:5d} m: n_park={n_park:4d} n_nonpark={n_nonpark:5d} | "
          f"degree MW p={p_mw_deg:.4g}{'*' if p_mw_deg<0.05 else ' '} | "
          f"betweenness MW p={p_mw_bet:.4g}{'*' if p_mw_bet<0.05 else ' '} | "
          f"reg IRR={irr:.3f} p={p_reg:.4g}{'*' if p_reg<0.05 else ''}")
    results_a.append(dict(radius_m=r, n_park=n_park, n_nonpark=n_nonpark,
                           mw_p_degree=p_mw_deg, mw_p_betweenness=p_mw_bet,
                           reg_irr=irr, reg_p=p_reg))

# ============================================================
# [B] leave-Hsinchu-out, with betweenness
# ============================================================
print("\n" + "=" * 70)
print("[B] leave-Hsinchu-out: degree and betweenness")
print("=" * 70)
std_lookup_2000 = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
df_2000 = build_regression_df(nodes, std_lookup_2000)
park_df = df_2000[df_2000["in_science_park"] == 1]
is_hsinchu = park_df["park_name"].fillna("").str.contains("Hsinchu", case=False)
n_hsinchu = int(is_hsinchu.sum())
print(f"of {len(park_df)} park-associated institutions, {n_hsinchu} are nearest to Hsinchu Science Park ({n_hsinchu/len(park_df)*100:.1f}%)")
hsinchu_ids = set(park_df.loc[is_hsinchu, "id"])
df_loo = df_2000[~df_2000["id"].isin(hsinchu_ids)].copy()
n_park_loo = int(df_loo["in_science_park"].sum())
print(f"excluding Hsinchu leaves {n_park_loo} park-associated institutions")
if n_park_loo >= 5:
    park_deg = df_loo.loc[df_loo["in_science_park"] == 1, "degree"]
    nonpark_deg = df_loo.loc[df_loo["in_science_park"] == 0, "degree"]
    _, p_deg_loo = stats.mannwhitneyu(park_deg, nonpark_deg, alternative="two-sided")
    park_bet = df_loo.loc[df_loo["in_science_park"] == 1, "betweenness"]
    nonpark_bet = df_loo.loc[df_loo["in_science_park"] == 0, "betweenness"]
    _, p_bet_loo = stats.mannwhitneyu(park_bet, nonpark_bet, alternative="two-sided")
    model_loo = smf.negativebinomial(
        "degree ~ in_science_park + log_univ + log_station + log_junction", data=df_loo
    ).fit(disp=0)
    coef = model_loo.params["in_science_park"]
    p = model_loo.pvalues["in_science_park"]
    irr = math.exp(coef)
    ci = model_loo.conf_int().loc["in_science_park"]
    print(f"without Hsinchu: degree MW p={p_deg_loo:.4g}, betweenness MW p={p_bet_loo:.4g}")
    print(f"without Hsinchu, regression: IRR={irr:.3f}, 95% CI [{math.exp(ci[0]):.3f}, {math.exp(ci[1]):.3f}], p={p:.4g}")

# ============================================================
# [C] exclude two false-positive registry parks and rematch
# ============================================================
print("\n" + "=" * 70)
print("[C] excluding Australian Technology Park and Freiburger Innovationszentrum")
print("=" * 70)
bad_parks = ["Australian Technology Park", "Freiburger Innovationszentrum"]
before_n = len(df_parks_gt)
df_parks_clean = df_parks_gt[~df_parks_gt["name"].isin(bad_parks)].reset_index(drop=True)
print(f"registry: {before_n} -> {len(df_parks_clean)} parks ({before_n-len(df_parks_clean)} removed)")

df_park_clean = match_all(df_geocoded, df_parks_clean, match_radius_m=2000)
std_lookup_clean = build_std_lookup(df_affil, df_park_clean, df_combined, match_radius_m=2000)
df_clean = build_regression_df(nodes, std_lookup_clean)
n_park_clean = int(df_clean["in_science_park"].sum())
print(f"park-associated institutions after exclusion: {n_park_clean} (at the 2,000 m threshold it was {int(df_2000['in_science_park'].sum())})")

park_deg = df_clean.loc[df_clean["in_science_park"] == 1, "degree"]
nonpark_deg = df_clean.loc[df_clean["in_science_park"] == 0, "degree"]
_, p_deg_c8 = stats.mannwhitneyu(park_deg, nonpark_deg, alternative="two-sided")
park_bet = df_clean.loc[df_clean["in_science_park"] == 1, "betweenness"]
nonpark_bet = df_clean.loc[df_clean["in_science_park"] == 0, "betweenness"]
_, p_bet_c8 = stats.mannwhitneyu(park_bet, nonpark_bet, alternative="two-sided")
model_c8 = smf.negativebinomial(
    "degree ~ in_science_park + log_univ + log_station + log_junction", data=df_clean
).fit(disp=0)
coef = model_c8.params["in_science_park"]
p_c8 = model_c8.pvalues["in_science_park"]
irr_c8 = math.exp(coef)
ci_c8 = model_c8.conf_int().loc["in_science_park"]
print(f"entity-resolved degree MW p = {p_deg_c8:.4g} (compare with the 2,000 m row in section A)")
print(f"entity-resolved betweenness MW p = {p_bet_c8:.4g}")
print(f"regression: IRR={irr_c8:.3f}, 95% CI [{math.exp(ci_c8[0]):.3f}, {math.exp(ci_c8[1]):.3f}], p={p_c8:.4g}")

# ============================================================
# [D] exclude hyper-authorship papers and rebuild the network (degree only)
# ============================================================
print("\n" + "=" * 70)
print("[D] excluding hyper-authorship papers, network rebuilt (degree only)")
print("=" * 70)

# institutions per paper, using exactly the grouping of build_std_network below
df_std = df_affil.dropna(subset=["Standardized_Institutions"])
inst_per_paper = df_std.groupby("Paper_ID")["Standardized_Institutions"].apply(
    lambda vals: len(set().union(*[set(split_institutions(v)) for v in vals])) if len(vals) else 0
)
total_papers = df_affil["Paper_ID"].nunique()
print(f"mean standardised institutions per paper: {inst_per_paper.mean():.2f} (over {total_papers} papers)")

def build_std_network(df_a):
    G = nx.Graph()
    for paper_id, group in df_a.dropna(subset=["Standardized_Institutions"]).groupby("Paper_ID"):
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

for thresh in [6, 8, 15]:
    excluded_papers = set(inst_per_paper[inst_per_paper >= thresh].index)
    pct = len(excluded_papers) / total_papers * 100
    df_a_filt = df_affil[~df_affil["Paper_ID"].isin(excluded_papers)]

    G_filt = build_std_network(df_a_filt)
    degree_filt = dict(G_filt.degree())

    # Park labels come from the unfiltered 2,000 m lookup: excluding papers
    # changes the topology, not where an institution is.
    df_deg_rows = []
    for node_id, deg in degree_filt.items():
        d = std_lookup_2000.get(node_id)
        if d is None or d["univ_research_count"] is None:
            continue
        df_deg_rows.append({"id": node_id, "degree": deg, "in_science_park": int(bool(d["in_science_park"]))})
    df_d = pd.DataFrame(df_deg_rows)
    n_park_d = int(df_d["in_science_park"].sum())
    park_deg = df_d.loc[df_d["in_science_park"] == 1, "degree"]
    nonpark_deg = df_d.loc[df_d["in_science_park"] == 0, "degree"]
    _, p_d = stats.mannwhitneyu(park_deg, nonpark_deg, alternative="two-sided")
    print(f"threshold >= {thresh:2d} institutions: {len(excluded_papers)} papers excluded ({pct:.2f}% of {total_papers})"
          f" -> network {G_filt.number_of_nodes()} nodes / {G_filt.number_of_edges()} edges, "
          f"n_park={n_park_d}, degree MW p={p_d:.4g}{'*' if p_d<0.05 else ''}")

print("\nDone.")
