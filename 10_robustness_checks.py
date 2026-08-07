"""
Recompute every statistic reported in the article from the current data.

Reads:  affil_full.csv, park_matches.csv, combined.csv, geocoded.csv,
        enriched.csv, std_nodes.json (entity-resolved network) and nodes.json
        (raw-string network).
Writes: nothing; every figure is printed, section by section, in the order the
        article reports them.

Sections: classification coverage for methods A and B, unique coordinate counts,
raw-string network size, Mann-Whitney comparisons for both node-identity
schemes, the four- and five-predictor negative binomial models with VIF, TOST
equivalence over four bound choices, and the institutions-per-paper baseline.
Set before running: SEARCH_RADIUS_M (the value imputed when no station or
junction was found) and the TOST bounds in section 8.
"""
import json, math, warnings
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
    from collections import defaultdict
    raw_to_std = df_affil[["Paper_ID", "Raw_Affiliation", "Standardized_Institutions"]].dropna(
        subset=["Raw_Affiliation", "Standardized_Institutions"]
    )
    park_lookup = df_park.set_index("Raw_Affiliation").to_dict(orient="index")
    density_lookup = df_combined.set_index("Raw_Affiliation")[
        ["univ_research_count", "nearest_station_m", "nearest_junction_m"]
    ].to_dict(orient="index") if df_combined is not None else {}
    std_verdicts = defaultdict(list); std_park_names = defaultdict(list)
    std_univ = defaultdict(list); std_station = defaultdict(list); std_junction = defaultdict(list)
    std_papers = defaultdict(set)
    raw_verdict_cache = {}
    for raw_aff, row in park_lookup.items():
        dist = row.get("distance_to_park_m")
        if dist is None or (isinstance(dist, float) and math.isnan(dist)):
            raw_verdict_cache[raw_aff] = None; continue
        raw_verdict_cache[raw_aff] = {"in_park": dist <= match_radius_m, "park_name": row.get("nearest_park_name")}
    for row in raw_to_std.itertuples():
        raw_aff = row.Raw_Affiliation
        v = raw_verdict_cache.get(raw_aff)
        if v is not None:
            for std_name in split_institutions(row.Standardized_Institutions):
                std_verdicts[std_name].append(v["in_park"])
                if v["in_park"]: std_park_names[std_name].append(v["park_name"])
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
            "in_science_park": majority, "park_name": top_name,
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
        if d is None or d["univ_research_count"] is None: continue
        rows.append({
            "id": n["id"], "degree": n["degree"], "betweenness": n.get("betweenness", np.nan),
            "in_science_park": int(bool(d["in_science_park"])), "park_name": d["park_name"],
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

print("loading data...")
df_affil = pd.read_csv("affil_full.csv")
df_park = pd.read_csv("park_matches.csv")
df_combined = pd.read_csv("combined.csv")
df_geocoded = pd.read_csv("geocoded.csv")
df_enriched = pd.read_csv("enriched.csv")
nodes = json.load(open("std_nodes.json", encoding="utf-8"))
naive_nodes = json.load(open("nodes.json", encoding="utf-8"))

std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
df = build_regression_df(nodes, std_lookup)
print(f"regression sample N={len(df)}, in parks={int(df['in_science_park'].sum())}\n")

print("="*70); print("[1] Method A distance classification (raw-affiliation level, 2000 m, before polygon refinement)"); print("="*70)
n_geo = df_geocoded.dropna(subset=["Latitude","Longitude"]).shape[0]
n_parkA = int(df_park["in_science_park_gt"].sum())
print(f"geocoded coordinates: {n_geo}")
print(f"Method A distance-only, inside a park: {n_parkA} / {len(df_park)} = {n_parkA/len(df_park)*100:.2f}%")

print("\n"+"="*70); print("[2] Method B coverage"); print("="*70)
print(f"rows in enriched.csv (raw affiliations processed by method B): {len(df_enriched)}")
if "in_osm_park" in df_enriched.columns:
    print("in_osm_park present, count:", df_enriched["in_osm_park"].sum())
else:
    print("columns:", df_enriched.columns.tolist())

print("\n"+"="*70); print("[3] Distinct coordinates (deduplicated lat/lon, for the Overpass batching description)"); print("="*70)
valid_coords = df_geocoded.dropna(subset=["Latitude","Longitude"])
n_unique_coords = valid_coords[["Latitude","Longitude"]].drop_duplicates().shape[0]
print(f"distinct coordinates: {n_unique_coords} (across {len(valid_coords)} geocoded institutions)")

print("\n"+"="*70); print("[4] Raw-string network, before filtering on coordinates"); print("="*70)
def build_naive_network(df_a):
    G = nx.Graph()
    for paper_id, group in df_a.dropna(subset=["Raw_Affiliation"]).groupby("Paper_ID"):
        insts = sorted(set(group["Raw_Affiliation"]))
        for inst in insts: G.add_node(inst)
        for i in range(len(insts)):
            for j in range(i+1, len(insts)):
                a,b = insts[i], insts[j]
                if G.has_edge(a,b): G[a][b]["weight"] += 1
                else: G.add_edge(a,b,weight=1)
    return G
G_naive = build_naive_network(df_affil)
print(f"raw-string network (before coordinate filtering): {G_naive.number_of_nodes()} nodes, {G_naive.number_of_edges()} edges")

print("\n"+"="*70); print("[5] Table 1: raw-string network Mann-Whitney (degree), park vs non-park"); print("="*70)
naive_park_deg = [n["degree"] for n in naive_nodes if n.get("in_science_park")]
naive_nonpark_deg = [n["degree"] for n in naive_nodes if not n.get("in_science_park")]
U, p_naive_mw = stats.mannwhitneyu(naive_park_deg, naive_nonpark_deg, alternative="two-sided")
n1,n2 = len(naive_park_deg), len(naive_nonpark_deg)
# common-language effect size, and the rank-biserial correlation derived from it
cles = U/(n1*n2); r = 2*cles-1
print(f"naive: n_park={n1} n_nonpark={n2}, mean_park={np.mean(naive_park_deg):.2f}, mean_nonpark={np.mean(naive_nonpark_deg):.2f}")
print(f"median_park={np.median(naive_park_deg):.1f}, median_nonpark={np.median(naive_nonpark_deg):.1f}")
print(f"p={p_naive_mw:.4g}, r={r:.4f}, CLES={cles:.4f}")

print("\n"+"="*70); print("[6] Table 1: entity-resolved Mann-Whitney (degree & betweenness), classification as assigned in step 08"); print("="*70)
er_park_deg = [n["degree"] for n in nodes if n.get("in_science_park")]
er_nonpark_deg = [n["degree"] for n in nodes if not n.get("in_science_park")]
U2, p_er_deg = stats.mannwhitneyu(er_park_deg, er_nonpark_deg, alternative="two-sided")
n1b,n2b = len(er_park_deg), len(er_nonpark_deg)
cles2 = U2/(n1b*n2b); r2 = 2*cles2-1
print(f"entity-resolved (in_park_best_guess): n_park={n1b} n_nonpark={n2b}")
print(f"mean_park={np.mean(er_park_deg):.2f} mean_nonpark={np.mean(er_nonpark_deg):.2f}")
print(f"median_park={np.median(er_park_deg):.1f} median_nonpark={np.median(er_nonpark_deg):.1f}")
print(f"degree: p={p_er_deg:.4g} r={r2:.4f} CLES={cles2:.4f}")

er_park_bet = [n["betweenness"] for n in nodes if n.get("in_science_park")]
er_nonpark_bet = [n["betweenness"] for n in nodes if not n.get("in_science_park")]
U3, p_er_bet = stats.mannwhitneyu(er_park_bet, er_nonpark_bet, alternative="two-sided")
cles3 = U3/(n1b*n2b); r3 = 2*cles3-1
print(f"betweenness: p={p_er_bet:.4g} r={r3:.4f} CLES={cles3:.4f}")

print("\n"+"="*70); print("[7] Table 2: four-predictor NB model (density only)"); print("="*70)
formula2 = "degree ~ in_science_park + log_univ + log_station + log_junction"
model2 = smf.negativebinomial(formula2, data=df).fit(disp=0)
poisson2 = smf.poisson(formula2, data=df).fit(disp=0)
alpha2 = model2.params.get("alpha", math.exp(model2.params.get("lnalpha", float("nan"))))
lr2 = 2*(model2.llf - poisson2.llf); lr2_p = 1 - stats.chi2.cdf(lr2, df=1)
print(f"alpha={alpha2:.4f}  NB AIC={model2.aic:.1f}  Poisson AIC={poisson2.aic:.1f}  LR={lr2:.1f} p={lr2_p:.4g}")
ci2 = model2.conf_int()
for name in ["Intercept","in_science_park","log_univ","log_station","log_junction"]:
    coef = model2.params[name]; p = model2.pvalues[name]; irr = math.exp(coef)
    lo, hi = math.exp(ci2.loc[name,0]), math.exp(ci2.loc[name,1])
    print(f"  {name:20s} coef={coef:8.4f} p={p:10.4g} IRR={irr:6.3f} CI=[{lo:.3f},{hi:.3f}]")

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
def compute_vif(df_, predictors):
    X = sm.add_constant(df_[predictors])
    return {predictors[i-1] if i>0 else "const": variance_inflation_factor(X.values,i) for i in range(X.shape[1])}
vif2 = compute_vif(df, ["in_science_park","log_univ","log_station","log_junction"])
print("VIF:", {k: round(v,3) for k,v in vif2.items()})

print("\n"+"="*70); print("[8] TOST at the 2000 m primary specification, plus bound sensitivity"); print("="*70)
def run_tost(coef, se, df_resid, log_low, log_high):
    t_upper = (coef-log_high)/se; p_upper = stats.t.cdf(t_upper, df_resid)
    t_lower = (coef-log_low)/se; p_lower = 1-stats.t.cdf(t_lower, df_resid)
    return max(p_upper,p_lower), p_upper, p_lower
coef = model2.params["in_science_park"]; se = model2.bse["in_science_park"]; dfresid = model2.df_resid
for lo,hi,label in [(0.80,1.25,"primary 80-125%"),(0.85,1.18,"narrow 85-118%"),(0.90,1.11,"narrower 90-111%"),(0.75,1.33,"wide 75-133%")]:
    p_tost,p_up,p_lo = run_tost(coef, se, dfresid, math.log(lo), math.log(hi))
    print(f"  {label}: TOST p={p_tost:.4g} (p_upper={p_up:.4g}, p_lower={p_lo:.4g}) {'EQUIV' if p_tost<0.05 else 'not equiv'}")

print("\n"+"="*70); print("[9] Table 4: five-predictor NB model (density + productivity)"); print("="*70)
formula4 = "degree ~ in_science_park + log_univ + log_station + log_junction + log_papers"
model4 = smf.negativebinomial(formula4, data=df).fit(disp=0)
poisson4 = smf.poisson(formula4, data=df).fit(disp=0)
alpha4 = model4.params.get("alpha", math.exp(model4.params.get("lnalpha", float("nan"))))
lr4 = 2*(model4.llf - poisson4.llf); lr4_p = 1 - stats.chi2.cdf(lr4, df=1)
print(f"alpha={alpha4:.4f}  NB AIC={model4.aic:.1f}  Poisson AIC={poisson4.aic:.1f}  LR={lr4:.1f} p={lr4_p:.4g}")
ci4 = model4.conf_int()
for name in ["Intercept","in_science_park","log_univ","log_station","log_junction","log_papers"]:
    coefv = model4.params[name]; p = model4.pvalues[name]; irr = math.exp(coefv)
    lo, hi = math.exp(ci4.loc[name,0]), math.exp(ci4.loc[name,1])
    print(f"  {name:20s} coef={coefv:8.4f} p={p:10.4g} IRR={irr:6.3f} CI=[{lo:.3f},{hi:.3f}]")
vif4 = compute_vif(df, ["in_science_park","log_univ","log_station","log_junction","log_papers"])
print("VIF:", {k: round(v,3) for k,v in vif4.items()})
print("corr(log_papers, log_univ) =", df["log_papers"].corr(df["log_univ"]))
print("corr(log_papers, in_science_park) =", df["log_papers"].corr(df["in_science_park"]))

print("\n"+"="*70); print("[10] Institutions per paper, and the share with >= 10 institutions"); print("="*70)
df_std = df_affil.dropna(subset=["Standardized_Institutions"])
inst_per_paper = df_std.groupby("Paper_ID")["Standardized_Institutions"].apply(
    lambda vals: len(set().union(*[set(split_institutions(v)) for v in vals])) if len(vals) else 0)
total_papers = df_affil["Paper_ID"].nunique()
print(f"mean={inst_per_paper.mean():.2f} median={inst_per_paper.median():.0f} max={inst_per_paper.max():.0f}")
n_ge10 = int((inst_per_paper>=10).sum())
print(f"papers with >=10 institutions: {n_ge10} / {total_papers} = {n_ge10/total_papers*100:.2f}%")

print("\nDone.")
