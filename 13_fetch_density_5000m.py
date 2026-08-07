"""
Robustness battery for the entity-resolved network.

Reads:  affil.csv or affil_full.csv, park_matches.csv, combined.csv,
        geocoded.csv, std_nodes.json, std_edges.json, and nodes.json when the
        raw-string network from step 09 is available.
Writes: nothing; all results are printed.

Covers negative-binomial diagnostics against Poisson, TOST equivalence testing,
VIF, geocoding missingness by country, distance-threshold sensitivity, park
concentration and leave-one-park-out, an institution-size control, community
composition, weighted vs unweighted degree, a label-permutation null model, and
Holm-Bonferroni correction.
Set before running: the radii passed to threshold_sensitivity(), the TOST
equivalence bounds (IRR 0.80-1.25) and SEARCH_RADIUS_M, which is the value
imputed when no station or junction was found within the search window.
"""

import math
import warnings
from collections import defaultdict

import json
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

SEARCH_RADIUS_M = 4000  # imputed distance when no station/junction was found


# ---------- link raw-affiliation data to standardised institutions ----------

def split_institutions(value):
    if pd.isna(value):
        return []
    return [s.strip() for s in str(value).split(",") if s.strip()]


def build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000):
    """
    Standardised institution name -> park verdict (majority vote at the given
    radius), modal park name, density and accessibility covariates (first valid
    value), and paper_count, the number of distinct papers the institution
    appears on, used as a proxy for institutional size.

    Same aggregation as step 08, but with the distance threshold as a parameter
    so the sensitivity analysis can vary it; the covariates are collected in the
    same pass to avoid scanning the affiliation table repeatedly.
    """
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
        raw_verdict_cache[raw_aff] = {
            "in_park": dist <= match_radius_m,
            "park_name": row.get("nearest_park_name"),
        }

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
                if d.get("univ_research_count") is not None and not (
                    isinstance(d["univ_research_count"], float) and math.isnan(d["univ_research_count"])
                ):
                    std_univ[std_name].append(d["univ_research_count"])
                if d.get("nearest_station_m") is not None and not (
                    isinstance(d["nearest_station_m"], float) and math.isnan(d["nearest_station_m"])
                ):
                    std_station[std_name].append(d["nearest_station_m"])
                if d.get("nearest_junction_m") is not None and not (
                    isinstance(d["nearest_junction_m"], float) and math.isnan(d["nearest_junction_m"])
                ):
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


# ---------- TOST equivalence test ----------

def run_tost(coef, se, df_resid, log_low, log_high, alpha=0.05):
    """
    Two one-sided tests:
      upper: H0 coef >= log_high vs H1 coef < log_high
      lower: H0 coef <= log_low  vs H1 coef > log_low
    Equivalence may be claimed only when both reject at alpha.

    The default bounds (IRR 0.80-1.25) are the 80-125% convention from
    bioequivalence testing, used here as a pre-specified, externally justified
    definition of a negligible effect rather than one chosen post hoc.
    """
    t_upper = (coef - log_high) / se
    p_upper = stats.t.cdf(t_upper, df_resid)
    t_lower = (coef - log_low) / se
    p_lower = 1 - stats.t.cdf(t_lower, df_resid)
    p_tost = max(p_upper, p_lower)
    return {
        "p_upper": p_upper, "p_lower": p_lower, "p_tost": p_tost,
        "equivalent": p_tost < alpha,
    }


# ---------- negative binomial diagnostics (alpha estimated by MLE) ----------

def fit_and_diagnose(df, formula):
    nb_model = smf.negativebinomial(formula, data=df).fit(disp=0)
    poisson_model = smf.poisson(formula, data=df).fit(disp=0)

    alpha = nb_model.params.get("alpha", None)
    if alpha is None:
        alpha = math.exp(nb_model.params.get("lnalpha", float("nan")))

    lr_stat = 2 * (nb_model.llf - poisson_model.llf)
    lr_p = 1 - stats.chi2.cdf(lr_stat, df=1)

    return {
        "nb_model": nb_model,
        "poisson_model": poisson_model,
        "alpha": alpha,
        "nb_aic": nb_model.aic,
        "poisson_aic": poisson_model.aic,
        "lr_stat": lr_stat,
        "lr_p": lr_p,
    }


# ---------- collinearity ----------

def compute_vif(df, predictors):
    X = sm.add_constant(df[predictors])
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


# ---------- distance-threshold sensitivity ----------

def threshold_sensitivity(df_affil, df_park, df_combined, nodes, radii_m):
    print("\n" + "=" * 70)
    print("Distance-threshold sensitivity")
    print("=" * 70)
    results = []
    for radius in radii_m:
        std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=radius)
        df = build_regression_df(nodes, std_lookup)
        n_park = df["in_science_park"].sum()
        n_nonpark = len(df) - n_park
        if n_park < 5:
            print(f"threshold {radius} m: only {n_park} park institutions, too few to test")
            continue

        park_deg = df.loc[df["in_science_park"] == 1, "degree"]
        nonpark_deg = df.loc[df["in_science_park"] == 0, "degree"]
        _, p_mw = stats.mannwhitneyu(park_deg, nonpark_deg, alternative="two-sided")

        model = smf.glm(
            "degree ~ in_science_park + log_univ + log_station + log_junction",
            data=df, family=sm.families.NegativeBinomial(),
        ).fit()
        coef = model.params["in_science_park"]
        p_reg = model.pvalues["in_science_park"]
        irr = math.exp(coef)

        print(f"threshold {radius} m: n_park={n_park}, n_nonpark={n_nonpark}, "
              f"MW p={p_mw:.4g}, regression IRR={irr:.3f} (p={p_reg:.4g})")
        results.append({
            "radius_m": radius, "n_park": int(n_park), "n_nonpark": int(n_nonpark),
            "mw_p": p_mw, "reg_irr": irr, "reg_p": p_reg,
        })
    return pd.DataFrame(results)


# ---------- concentration in one park, and leave-one-park-out ----------

def hsinchu_concentration_and_loo(df_affil, df_park, df_combined, nodes):
    print("\n" + "=" * 70)
    print("Share of park institutions nearest to Hsinchu, and leave-one-park-out")
    print("=" * 70)
    std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
    df = build_regression_df(nodes, std_lookup)

    park_df = df[df["in_science_park"] == 1]
    is_hsinchu = park_df["park_name"].fillna("").str.contains("Hsinchu", case=False)
    n_hsinchu = is_hsinchu.sum()
    print(f"of {len(park_df)} park-associated institutions, {n_hsinchu} have Hsinchu "
          f"Science Park as their nearest park ({n_hsinchu / len(park_df) * 100:.1f}%)")

    hsinchu_ids = set(park_df.loc[is_hsinchu, "id"])
    df_loo = df[~df["id"].isin(hsinchu_ids)].copy()

    n_park_loo = df_loo["in_science_park"].sum()
    print(f"excluding Hsinchu leaves {n_park_loo} park-associated institutions")
    if n_park_loo >= 5:
        model_loo = smf.glm(
            "degree ~ in_science_park + log_univ + log_station + log_junction",
            data=df_loo, family=sm.families.NegativeBinomial(),
        ).fit()
        coef = model_loo.params["in_science_park"]
        p = model_loo.pvalues["in_science_park"]
        irr = math.exp(coef)
        ci = model_loo.conf_int().loc["in_science_park"]
        print(f"regression without Hsinchu: IRR={irr:.3f}, 95% CI [{math.exp(ci[0]):.3f}, {math.exp(ci[1]):.3f}], p={p:.4g}")
    return n_hsinchu, len(park_df)


# ---------- institution-size control ----------

def regression_with_size_control(df):
    print("\n" + "=" * 70)
    print("Regression with institutional size (paper count) as a control")
    print("=" * 70)
    model = smf.glm(
        "degree ~ in_science_park + log_univ + log_station + log_junction + log_papers",
        data=df, family=sm.families.NegativeBinomial(),
    ).fit()
    print(model.summary())
    return model


# ---------- null model: permute park labels, keep the degree sequence ----------

def null_model_comparison(nodes, n_perm=5000, seed=42):
    """
    Permutation test on the park label.

    A degree-preserving configuration model would be circular here: it preserves
    exactly the node degrees whose park/non-park difference is under test. Instead
    the observed degree sequence is held fixed and the park label is reshuffled,
    which asks how often a purely random labelling produces a group difference as
    large as the observed one. It is distribution-free and an independent check on
    the Mann-Whitney result.
    """
    print("\n" + "=" * 70)
    print("Label-permutation null model")
    print("=" * 70)
    deg = np.array([n["degree"] for n in nodes])
    is_park = np.array([bool(n["in_science_park"]) for n in nodes])
    n_park = is_park.sum()

    observed_diff = deg[is_park].mean() - deg[~is_park].mean()
    print(f"observed mean degree difference (park - non-park) = {observed_diff:.3f}")

    rng = np.random.default_rng(seed)
    count_ge = 0
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(len(deg))
        park_idx = perm[:n_park]
        nonpark_idx = perm[n_park:]
        d = deg[park_idx].mean() - deg[nonpark_idx].mean()
        diffs[i] = d
        if abs(d) >= abs(observed_diff):
            count_ge += 1

    p_perm = count_ge / n_perm
    print(f"permutation p ({n_perm} relabellings, two-sided) = {p_perm:.4f}")
    print(f"mean/sd of the difference under random labelling = {diffs.mean():.3f} / {diffs.std():.3f}")

    # The comparison value is recomputed from the same nodes rather than quoted,
    # so it always matches the dataset actually being analysed.
    _park_deg = deg[is_park]
    _nonpark_deg = deg[~is_park]
    _, p_mw_ref = stats.mannwhitneyu(_park_deg, _nonpark_deg, alternative="two-sided")
    same_direction = "consistent" if (p_perm < 0.05) == (p_mw_ref < 0.05) else "INCONSISTENT, check this"
    print(f"(reference: entity-resolved Mann-Whitney on the same data gives p = {p_mw_ref:.4g}; "
          f"the permutation test is {same_direction} with it — two tests resting on "
          f"different assumptions, not the same test twice.)")
    return p_perm


# ---------- community structure vs park classification ----------

def community_park_composition(nodes):
    print("\n" + "=" * 70)
    print("Community structure vs park classification")
    print("=" * 70)
    from collections import Counter
    comm_park = defaultdict(lambda: [0, 0])
    for n in nodes:
        c = n["community"]
        comm_park[c][1] += 1
        if n["in_science_park"]:
            comm_park[c][0] += 1

    comm_sizes = Counter(n["community"] for n in nodes)
    print("share of park institutions in the 8 largest communities:")
    for cid, size in comm_sizes.most_common(8):
        park_n, total_n = comm_park[cid]
        pct = park_n / total_n * 100 if total_n else 0
        print(f"  community {cid} ({total_n} institutions): {park_n} park institutions ({pct:.1f}%)")

    total_park = sum(v[0] for v in comm_park.values())
    n_communities_with_park = sum(1 for v in comm_park.values() if v[0] > 0)
    print(f"\n{total_park} park institutions spread over {n_communities_with_park} "
          f"communities (out of {len(comm_park)} in total)")


# ---------- weighted vs unweighted degree ----------

def weighted_vs_unweighted_degree(std_edges_path="std_edges.json"):
    print("\n" + "=" * 70)
    print("Degree: weighted vs unweighted")
    print("=" * 70)
    edges = json.load(open(std_edges_path, encoding="utf-8"))
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e.get("weight", 1))
    unweighted = dict(G.degree())
    weighted = dict(G.degree(weight="weight"))
    diffs = [(k, unweighted[k], weighted[k]) for k in list(unweighted)[:5]]
    print("first 5 nodes (unweighted = distinct partners, weighted = total co-authored papers):")
    for name, u, w in diffs:
        print(f"  {name[:50]}: unweighted={u}, weighted={w}")
    print("The reported analysis uses unweighted degree (distinct collaborating "
          "institutions), which is a different scale from the weighted distance "
          "used for betweenness; state this explicitly in the methods.")


# ---------- missingness: geocoding failure rate by country ----------

def missingness_by_country(df_affil, df_geocoded):
    print("\n" + "=" * 70)
    print("Geocoding failure rate by country")
    print("=" * 70)
    merged = df_affil[["Raw_Affiliation", "Institution_Countries"]].drop_duplicates(subset=["Raw_Affiliation"])
    merged = merged.merge(df_geocoded[["Raw_Affiliation", "Latitude"]], on="Raw_Affiliation", how="left")
    merged["failed"] = merged["Latitude"].isna()
    merged["first_country"] = merged["Institution_Countries"].fillna("Unknown").apply(
        lambda s: str(s).split(",")[0].strip() if s else "Unknown"
    )
    summary = merged.groupby("first_country")["failed"].agg(["sum", "count"])
    summary["fail_rate"] = summary["sum"] / summary["count"]
    summary = summary[summary["count"] >= 20].sort_values("fail_rate", ascending=False)
    print("highest failure rates (countries with at least 20 addresses, to avoid small-sample noise):")
    print(summary.head(15))

    for code in ["TW", "CN", "JP", "KR"]:
        if code in summary.index:
            row = summary.loc[code]
            print(f"reference: {code} failure rate = {row['fail_rate']*100:.1f}% (n={int(row['count'])})")
    return summary


# ---------- multiple-comparison correction ----------

def holm_bonferroni(pvalue_dict, alpha=0.05):
    print("\n" + "=" * 70)
    print("Holm-Bonferroni correction")
    print("=" * 70)
    items = sorted(pvalue_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    print(f"{'test':40s} {'raw p':>10s} {'threshold':>10s} {'signif?':>8s}")
    still_significant = True
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        significant = still_significant and (p < threshold)
        if not significant:
            still_significant = False
        print(f"{name[:40]:40s} {p:>10.4g} {threshold:>10.4g} {'yes' if significant else 'no':>8s}")


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
    df_park = pd.read_csv("park_matches.csv")
    df_combined = pd.read_csv("combined.csv")
    df_geocoded = pd.read_csv("geocoded.csv")
    nodes = json.load(open("std_nodes.json", encoding="utf-8"))

    std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
    df = build_regression_df(nodes, std_lookup)
    print(f"regression sample: {len(df)} standardised institutions "
          f"({df['in_science_park'].sum()} in parks)\n")

    diag = fit_and_diagnose(df, "degree ~ in_science_park + log_univ + log_station + log_junction")
    print("=" * 70)
    print("Negative binomial diagnostics")
    print("=" * 70)
    print(f"dispersion alpha = {diag['alpha']:.4f} (alpha clearly above 0 confirms "
          f"overdispersion, i.e. NB rather than Poisson)")
    print(f"NB AIC = {diag['nb_aic']:.1f}, Poisson AIC = {diag['poisson_aic']:.1f} (lower is better)")
    print(f"likelihood-ratio test (NB vs Poisson): LR = {diag['lr_stat']:.1f}, p = {diag['lr_p']:.4g}")
    print("\nNB model with alpha estimated by MLE:")
    print(diag["nb_model"].summary())

    coef = diag["nb_model"].params["in_science_park"]
    se = diag["nb_model"].bse["in_science_park"]
    df_resid = diag["nb_model"].df_resid
    tost = run_tost(coef, se, df_resid, log_low=math.log(0.80), log_high=math.log(1.25))
    print("\n" + "=" * 70)
    print("TOST equivalence test (bounds IRR 0.80-1.25)")
    print("=" * 70)
    print(f"p_upper = {tost['p_upper']:.4g}, p_lower = {tost['p_lower']:.4g}")
    print(f"TOST p = {tost['p_tost']:.4g}"
          f"  {'(equivalence supported)' if tost['equivalent'] else '(equivalence not established)'}")

    vif_df = compute_vif(df, ["in_science_park", "log_univ", "log_station", "log_junction"])
    print("\n" + "=" * 70)
    print("VIF collinearity diagnostics")
    print("=" * 70)
    print(vif_df.to_string(index=False))

    missingness_by_country(df_affil, df_geocoded)

    threshold_sensitivity(df_affil, df_park, df_combined, nodes, radii_m=[500, 1000, 2000, 3000, 5000])

    hsinchu_concentration_and_loo(df_affil, df_park, df_combined, nodes)

    regression_with_size_control(df)

    community_park_composition(nodes)

    weighted_vs_unweighted_degree()

    null_model_comparison(nodes)

    # Every p value entering the correction is recomputed from the current data.
    # The naive-network test needs nodes.json from 09_build_naive_network.py; if
    # that file is absent the term is skipped and reported as skipped rather than
    # filled in from another run.
    import os as _os
    park_deg_er = df.loc[df["in_science_park"] == 1, "degree"]
    nonpark_deg_er = df.loc[df["in_science_park"] == 0, "degree"]
    _, p_entity_resolved = stats.mannwhitneyu(park_deg_er, nonpark_deg_er, alternative="two-sided")

    pvalue_dict = {
        "entity-resolved degree Mann-Whitney": p_entity_resolved,
        "NB regression park coefficient (proper alpha)": diag["nb_model"].pvalues["in_science_park"],
    }

    if _os.path.exists("nodes.json"):
        naive_nodes = json.load(open("nodes.json", encoding="utf-8"))
        naive_park = [n["degree"] for n in naive_nodes if n.get("in_science_park")]
        naive_nonpark = [n["degree"] for n in naive_nodes if not n.get("in_science_park")]
        _, p_naive = stats.mannwhitneyu(naive_park, naive_nonpark, alternative="two-sided")
        pvalue_dict["naive degree Mann-Whitney"] = p_naive
    else:
        print("\nnodes.json (output of 09_build_naive_network.py) not found, so the "
              "naive-network term is omitted from the Holm-Bonferroni correction. "
              "Run step 09 and re-run this script for the complete three-term correction.")

    holm_bonferroni(pvalue_dict)

    print("\nDone.")
