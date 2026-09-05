#!/usr/bin/env python3
"""16_review_response_2026_09.py - analyses added in response to the simulated
five-reviewer panel (September 2026) on the Short Communication.

Run from the repository root:
    python analysis/16_review_response_2026_09.py

Everything reuses the estimation functions of 15_reviewer_response_analyses.py
(negative binomial GLM with MLE dispersion, full-model label permutation), so
that every new number is on the same footing as the reported ones.

Blocks
  A  placebo exclusion: random work subsets matched on tie-instance share
  B  covariate attenuation under the >=10 cap (is the park term special?)
  C  2,000 m specification under the >=10 cap and fractional counting
  D  CAR-T under the >=10 cap and fractional counting
  E  5,000-work silicon-carbide subsamples through the four-clause criterion
  F  full-model permutation on the naive 69,455-node network
  G  permutation-null diagnostics: convergence, shape, density-stratified nulls
  H  park-level dependence: catchment-clustered SEs and catchment permutation
  I  sandwich SEs against bootstrap SEs
  J  geography of hyperauthored works: same-country share, pairwise distance
"""
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "full_run"
CART = ROOT / "data" / "cart_run"
OUT = ROOT / "analysis" / "review_response_2026_09"
OUT.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location("rr", ROOT / "analysis" / "15_reviewer_response_analyses.py")
rr = importlib.util.module_from_spec(spec)
_cwd = os.getcwd(); os.chdir(FULL); spec.loader.exec_module(rr); os.chdir(_cwd)

N_PERM = 5000
N_PLACEBO = 2000
N_SUBSAMPLES = 50
N_PERM_SUB = 1000
SEED = 20260905

T0 = time.time()
RESULTS = {"settings": {"n_perm": N_PERM, "n_placebo": N_PLACEBO, "n_subsamples": N_SUBSAMPLES,
                        "n_perm_subsample": N_PERM_SUB, "seed": SEED}}


def log(msg):
    print(f"[{time.time()-T0:7.0f}s] {msg}", flush=True)


def save():
    json.dump(RESULTS, open(OUT / "results.json", "w"), indent=1, default=float)


def fit(frame, covariates=rr.COVARIATES, cov_type="nonrobust", groups=None):
    alpha = rr.mle_dispersion(frame, covariates)
    X, y = rr.design_matrix(frame, covariates)
    fam = sm.families.NegativeBinomial(alpha=alpha)
    if cov_type == "cluster":
        m = sm.GLM(y, X, family=fam).fit(cov_type="cluster", cov_kwds={"groups": groups})
    elif cov_type == "nonrobust":
        m = sm.GLM(y, X, family=fam).fit()
    else:
        m = sm.GLM(y, X, family=fam).fit(cov_type=cov_type)
    ci = m.conf_int()[1]
    return {"n": len(frame), "treated": int(frame["in_science_park"].sum()),
            "irr": math.exp(m.params[1]), "ci95_low": math.exp(ci[0]), "ci95_high": math.exp(ci[1]),
            "coef": float(m.params[1]), "se": float(m.bse[1]), "model_p": float(m.pvalues[1]),
            "alpha": float(alpha), "converged": bool(getattr(m, "converged", True)),
            "covariate_coefs": {c: float(m.params[2 + i]) for i, c in enumerate(covariates)}}


def perm_coefs(frame, covariates=rr.COVARIATES, n=N_PERM, seed=rr.SEED_PERMUTATION, strata=None):
    """Return the vector of permuted park coefficients (and convergence flags)."""
    alpha = rr.mle_dispersion(frame, covariates)
    X, y = rr.design_matrix(frame, covariates)
    fam = sm.families.NegativeBinomial(alpha=alpha)
    labels = X[:, 1].copy()
    rng = np.random.default_rng(seed)
    if strata is not None:
        groups = [np.where(strata == g)[0] for g in np.unique(strata)]
    coefs = np.empty(n); conv = np.empty(n, dtype=bool)
    for i in range(n):
        if strata is None:
            X[:, 1] = rng.permutation(labels)
        else:
            p = labels.copy()
            for idx in groups:
                if len(idx) > 1:
                    p[idx] = rng.permutation(labels[idx])
            X[:, 1] = p
        m = sm.GLM(y, X, family=fam).fit()
        coefs[i] = m.params[1]; conv[i] = bool(getattr(m, "converged", True))
    X[:, 1] = labels
    return coefs, conv


def perm_p(observed_coef, coefs):
    return float((np.sum(np.abs(coefs) >= abs(observed_coef)) + 1) / (len(coefs) + 1))


def tie_instances(pi):
    return {p: len(v) * (len(v) - 1) // 2 for p, v in pi.items()}


# =========================================================================== #
log("loading full run")
os.chdir(FULL)
affil, park, combined, geocoded, nodes, density5 = rr.load_inputs()
density2 = pd.read_csv("density_2000m_v2.csv")
os.chdir(_cwd)
dmap5 = rr.density_by_affiliation(geocoded, density5)
dmap2 = rr.density_by_affiliation(geocoded, density2)
lookup5 = rr.build_institution_lookup(affil, park, combined, 5000, dmap5)
headline = rr.build_frame(nodes, lookup5)
pi = rr.papers_to_institutions(affil)
TI = tie_instances(pi)
TI_total = sum(TI.values())
head = fit(headline)
log(f"headline IRR {head['irr']:.4f} p {head['model_p']:.2e} n {head['n']} treated {head['treated']}")
RESULTS["headline"] = head
LOG_HEAD = math.log(head["irr"])


def capped_frame(base, pi_, cap=None, fractional=False, exclude_set=None):
    if exclude_set is not None:
        pi_use = {p: v for p, v in pi_.items() if p not in exclude_set}
        present = set(); [present.update(v) for v in pi_use.values()]
        deg = rr.rebuild_degree(pi_use, fractional=fractional)
    else:
        pi_use = pi_
        present = rr.institutions_present(pi_, cap) if cap else None
        deg = rr.rebuild_degree(pi_, exclude_at=cap, fractional=fractional)
    f = base if present is None else base[base["id"].isin(present)]
    f = f.copy(); f["degree"] = f["id"].map(lambda i: deg.get(i, 0.0 if fractional else 0))
    return f


# --------------------------------------------------------------------------- #
# A. placebo exclusion
# --------------------------------------------------------------------------- #
log("A. placebo exclusion")
big = [p for p, v in pi.items() if len(v) >= 10]
share_big = sum(TI[p] for p in big) / TI_total
cap10 = capped_frame(headline, pi, cap=10)
cap10_fit = fit(cap10)
removed_cap10 = 1 - math.log(cap10_fit["irr"]) / LOG_HEAD
RESULTS["cap10"] = dict(cap10_fit, log_effect_removed=removed_cap10, works_excluded=len(big),
                        tie_instance_share=share_big)
log(f"   >=10 cap: {len(big)} works, tie share {share_big:.4f}, IRR {cap10_fit['irr']:.4f}, removed {removed_cap10:.3f}")

rng = np.random.default_rng(SEED)
multi = [p for p, v in pi.items() if len(v) >= 2]
small = [p for p in multi if len(pi[p]) < 10]


def placebo(universe, target_share, n_draws, label):
    irr, rem, nn, nworks = [], [], [], []
    for d in range(n_draws):
        order = rng.permutation(len(universe))
        acc = 0; chosen = set()
        for j in order:
            p = universe[j]; chosen.add(p); acc += TI[p]
            if acc / TI_total >= target_share:
                break
        f = capped_frame(headline, pi, exclude_set=chosen)
        r = fit(f)
        irr.append(r["irr"]); rem.append(1 - math.log(r["irr"]) / LOG_HEAD); nn.append(r["n"]); nworks.append(len(chosen))
        if d % 200 == 0:
            log(f"   {label} draw {d}: IRR {r['irr']:.4f} removed {rem[-1]:.3f} works {len(chosen)} n {r['n']}")
    irr = np.array(irr); rem = np.array(rem)
    out = {"draws": n_draws, "target_tie_share": target_share,
           "works_excluded_mean": float(np.mean(nworks)), "n_mean": float(np.mean(nn)),
           "irr_mean": float(irr.mean()), "irr_sd": float(irr.std(ddof=1)),
           "irr_p2_5": float(np.percentile(irr, 2.5)), "irr_p5": float(np.percentile(irr, 5)),
           "irr_median": float(np.median(irr)), "irr_p95": float(np.percentile(irr, 95)),
           "removed_mean": float(rem.mean()), "removed_p95": float(np.percentile(rem, 95)),
           "removed_p99": float(np.percentile(rem, 99)), "removed_max": float(rem.max()),
           "share_removed_ge_cap10": float(np.mean(rem >= removed_cap10)),
           "share_irr_le_cap10": float(np.mean(irr <= cap10_fit["irr"])),
           "placebo_p_one_sided": float((np.sum(rem >= removed_cap10) + 1) / (n_draws + 1)),
           "z_of_cap10": float((removed_cap10 - rem.mean()) / rem.std(ddof=1))}
    np.save(OUT / f"placebo_{label}_irr.npy", irr)
    return out


RESULTS["placebo_all_works"] = placebo(multi, share_big, N_PLACEBO, "all")
save()
RESULTS["placebo_small_works_only"] = placebo(small, share_big, N_PLACEBO, "small")
save()
log(f"   placebo (all): removed mean {RESULTS['placebo_all_works']['removed_mean']:.3f}, p {RESULTS['placebo_all_works']['placebo_p_one_sided']:.4f}")
log(f"   placebo (k<10): removed mean {RESULTS['placebo_small_works_only']['removed_mean']:.3f}, p {RESULTS['placebo_small_works_only']['placebo_p_one_sided']:.4f}")

# --------------------------------------------------------------------------- #
# B. covariate attenuation under the cap and fractional counting
# --------------------------------------------------------------------------- #
log("B. covariate coefficients")
frac = capped_frame(headline, pi, fractional=True)
frac_fit = fit(frac)
RESULTS["fractional"] = dict(frac_fit, log_effect_removed=1 - math.log(frac_fit["irr"]) / LOG_HEAD)
RESULTS["covariate_attenuation"] = {
    "headline": dict(park=head["coef"], **head["covariate_coefs"]),
    "cap10": dict(park=cap10_fit["coef"], **cap10_fit["covariate_coefs"]),
    "fractional": dict(park=frac_fit["coef"], **frac_fit["covariate_coefs"]),
}
for k in ["park"] + rr.COVARIATES:
    h = RESULTS["covariate_attenuation"]["headline"][k]
    RESULTS["covariate_attenuation"].setdefault("share_removed_cap10", {})[k] = 1 - RESULTS["covariate_attenuation"]["cap10"][k] / h if h else None
    RESULTS["covariate_attenuation"].setdefault("share_removed_fractional", {})[k] = 1 - RESULTS["covariate_attenuation"]["fractional"][k] / h if h else None
log(f"   {json.dumps(RESULTS['covariate_attenuation']['share_removed_cap10'])}")
save()

# --------------------------------------------------------------------------- #
# C. 2,000 m specification (treatment 2,000 m, density 2,000 m same vintage)
# --------------------------------------------------------------------------- #
log("C. 2,000 m rows")
lookup2 = rr.build_institution_lookup(affil, park, combined, 2000, dmap2)
base2 = rr.build_frame(nodes, lookup2)
rows2 = {}
for label, f in [("full", base2), ("cap10", capped_frame(base2, pi, cap=10)), ("fractional", capped_frame(base2, pi, fractional=True))]:
    r = fit(f); c, cv = perm_coefs(f); r["permutation_p"] = perm_p(r["coef"], c); r["nonconverged"] = int((~cv).sum())
    rows2[label] = r
    log(f"   2000m {label}: n {r['n']} treated {r['treated']} IRR {r['irr']:.4f} p {r['model_p']:.4f} perm {r['permutation_p']:.4f}")
lh2 = math.log(rows2["full"]["irr"])
for label in ("cap10", "fractional"):
    rows2[label]["log_effect_removed"] = 1 - math.log(rows2[label]["irr"]) / lh2 if lh2 else None
RESULTS["spec_2000m"] = rows2
save()

# --------------------------------------------------------------------------- #
# D. CAR-T under the cap and fractional counting
# --------------------------------------------------------------------------- #
log("D. CAR-T")
os.chdir(CART)
c_affil, c_park, c_comb, c_geo, c_nodes, c_dens = rr.load_inputs()
os.chdir(_cwd)
c_dmap = rr.density_by_affiliation(c_geo, c_dens)
c_lookup = rr.build_institution_lookup(c_affil, c_park, c_comb, 5000, c_dmap)
c_base = rr.build_frame(c_nodes, c_lookup)
c_pi = rr.papers_to_institutions(c_affil)
c_TI = tie_instances(c_pi); c_big = [p for p, v in c_pi.items() if len(v) >= 10]
cart = {"works_with_institutions": len(c_pi), "works_ge10": len(c_big),
        "share_works_ge10": len(c_big) / len(c_pi), "tie_share_ge10": sum(c_TI[p] for p in c_big) / sum(c_TI.values()),
        "max_institutions": max(len(v) for v in c_pi.values())}
for label, f in [("headline", c_base), ("cap10", capped_frame(c_base, c_pi, cap=10)), ("fractional", capped_frame(c_base, c_pi, fractional=True))]:
    r = fit(f); c, cv = perm_coefs(f); r["permutation_p"] = perm_p(r["coef"], c); r["nonconverged"] = int((~cv).sum())
    cart[label] = r
    log(f"   CAR-T {label}: n {r['n']} treated {r['treated']} IRR {r['irr']:.4f} p {r['model_p']:.4f} perm {r['permutation_p']:.4f}")
lhc = math.log(cart["headline"]["irr"])
for label in ("cap10", "fractional"):
    cart[label]["log_effect_removed"] = 1 - math.log(cart[label]["irr"]) / lhc if lhc else None
r = fit(c_base, covariates=rr.COVARIATES + ["log_papers"]); cart["productivity"] = r
RESULTS["cart"] = cart
save()

# --------------------------------------------------------------------------- #
# E. 5,000-work silicon-carbide subsamples through the four clauses
# --------------------------------------------------------------------------- #
log("E. subsamples")
paper_ids = np.array(sorted(pi.keys()))
all_paper_ids = np.array(sorted(affil["Paper_ID"].unique()))
node_by_id = {n["id"]: n for n in nodes}
sub_rows = []
rng_sub = np.random.default_rng(SEED + 1)
for d in range(N_SUBSAMPLES):
    chosen = set(rng_sub.choice(all_paper_ids, 5000, replace=False))
    a_sub = affil[affil["Paper_ID"].isin(chosen)]
    lk = rr.build_institution_lookup(a_sub, park, combined, 5000, dmap5)
    pi_sub = rr.papers_to_institutions(a_sub)
    deg = rr.rebuild_degree(pi_sub)
    present = set(); [present.update(v) for v in pi_sub.values()]
    sub_nodes = [node_by_id[i] for i in present if i in node_by_id]
    f = rr.build_frame(sub_nodes, lk, degree=deg)
    row = {"draw": d, "n": len(f), "treated": int(f["in_science_park"].sum())}
    try:
        r1 = fit(f); row.update(irr=r1["irr"], model_p=r1["model_p"])
        c, _ = perm_coefs(f, n=N_PERM_SUB, seed=SEED + 100 + d); row["perm_p"] = perm_p(r1["coef"], c)
        r3 = fit(f, covariates=rr.COVARIATES + ["log_papers"]); row.update(prod_irr=r3["irr"], prod_p=r3["model_p"])
        f4 = capped_frame(f, pi_sub, cap=10); r4 = fit(f4)
        c4, _ = perm_coefs(f4, n=N_PERM_SUB, seed=SEED + 200 + d)
        row.update(cap10_irr=r4["irr"], cap10_p=r4["model_p"], cap10_perm_p=perm_p(r4["coef"], c4), cap10_n=r4["n"])
        row.update(clause1=r1["model_p"] < .05 and r1["irr"] > 1, clause2=row["perm_p"] < .05 and r1["irr"] > 1,
                   clause3=r3["model_p"] < .05 and r3["irr"] > 1, clause4=row["cap10_perm_p"] < .05 and r4["irr"] > 1)
    except Exception as e:
        row["error"] = str(e)
    sub_rows.append(row)
    if d % 5 == 0:
        log(f"   subsample {d}: n {row['n']} treated {row['treated']} IRR {row.get('irr', float('nan')):.3f} p {row.get('model_p', float('nan')):.3f} perm {row.get('perm_p', float('nan')):.3f} prod {row.get('prod_irr', float('nan')):.3f}")
sub = pd.DataFrame(sub_rows); sub.to_csv(OUT / "subsamples_5000.csv", index=False)
ok = sub.dropna(subset=["irr"])
RESULTS["subsamples_5000"] = {
    "draws": len(sub), "valid": len(ok),
    "irr_median": float(ok["irr"].median()), "irr_iqr": [float(ok["irr"].quantile(.25)), float(ok["irr"].quantile(.75))],
    "treated_median": float(ok["treated"].median()), "n_median": float(ok["n"].median()),
    "share_pass_clause1": float(ok["clause1"].mean()), "share_pass_clause2": float(ok["clause2"].mean()),
    "share_pass_clause3": float(ok["clause3"].mean()), "share_pass_clauses_1_to_3": float((ok["clause1"] & ok["clause2"] & ok["clause3"]).mean()),
    "share_pass_all_four": float((ok["clause1"] & ok["clause2"] & ok["clause3"] & ok["clause4"]).mean()),
    "share_fail_clause1": float((~ok["clause1"]).mean()),
    "model_p_median": float(ok["model_p"].median()), "perm_p_median": float(ok["perm_p"].median()),
    "prod_irr_median": float(ok["prod_irr"].median()),
}
log(f"   subsamples: pass clause1 {RESULTS['subsamples_5000']['share_pass_clause1']:.2f}, 1-3 {RESULTS['subsamples_5000']['share_pass_clauses_1_to_3']:.2f}, all four {RESULTS['subsamples_5000']['share_pass_all_four']:.2f}")
save()

# --------------------------------------------------------------------------- #
# F. naive-network permutation
# --------------------------------------------------------------------------- #
log("F. naive network permutation")
naive = rr.build_naive_frame(affil, park, combined)
nf = fit(naive)
c, cv = perm_coefs(naive, n=N_PERM)
nf["permutation_p"] = perm_p(nf["coef"], c); nf["nonconverged"] = int((~cv).sum())
nf["null_sd"] = float(c.std(ddof=1)); nf["null_p95_abs"] = float(np.percentile(np.abs(c), 95))
nf["observed_over_null_sd"] = float(abs(nf["coef"]) / nf["null_sd"])
RESULTS["naive_network"] = nf
log(f"   naive: n {nf['n']} IRR {nf['irr']:.4f} model p {nf['model_p']:.2e} perm p {nf['permutation_p']:.5f} null sd {nf['null_sd']:.4f}")
save()

# --------------------------------------------------------------------------- #
# G. permutation-null diagnostics on the headline
# --------------------------------------------------------------------------- #
log("G. null diagnostics")
c_un, cv_un = perm_coefs(headline, n=N_PERM)
np.save(OUT / "null_unrestricted.npy", c_un)
diag = {"unrestricted": {"permutation_p": perm_p(head["coef"], c_un), "nonconverged": int((~cv_un).sum()),
                         "sd": float(c_un.std(ddof=1)), "mean": float(c_un.mean()),
                         "skew": float(stats.skew(c_un)), "excess_kurtosis": float(stats.kurtosis(c_un)),
                         "p95_abs": float(np.percentile(np.abs(c_un), 95)), "p99_abs": float(np.percentile(np.abs(c_un), 99)),
                         "p95_abs_over_sd": float(np.percentile(np.abs(c_un), 95) / c_un.std(ddof=1)),
                         "share_beyond_3sd": float(np.mean(np.abs(c_un) > 3 * c_un.std(ddof=1))),
                         "observed_over_sd": float(abs(head["coef"]) / c_un.std(ddof=1)),
                         "shapiro_p": float(stats.shapiro(c_un[:5000]).pvalue)}}
for q, label in [(5, "density_quintile"), (10, "density_decile")]:
    strata = pd.qcut(headline["log_univ"].rank(method="first"), q, labels=False).values
    cs, cvs = perm_coefs(headline, n=N_PERM, seed=rr.SEED_STRATIFIED_PERMUTATION, strata=strata)
    diag[label] = {"permutation_p": perm_p(head["coef"], cs), "nonconverged": int((~cvs).sum()),
                   "sd": float(cs.std(ddof=1)), "p95_abs_over_sd": float(np.percentile(np.abs(cs), 95) / cs.std(ddof=1)),
                   "excess_kurtosis": float(stats.kurtosis(cs)), "strata": q}
    log(f"   {label}: perm p {diag[label]['permutation_p']:.4f} sd {diag[label]['sd']:.4f}")
    np.save(OUT / f"null_{label}.npy", cs)
# treated-density contrast that motivates the stratification
diag["density_by_arm"] = {"treated_mean_univ": float(headline.loc[headline.in_science_park == 1, "univ_research_count"].mean()),
                          "control_mean_univ": float(headline.loc[headline.in_science_park == 0, "univ_research_count"].mean())}
# residual-permutation (Freedman-Lane style on the linear predictor scale): permute
# the park label after residualising it on the covariates, i.e. permute the part of
# treatment that is orthogonal to density, then refit.
Xc = np.column_stack([np.ones(len(headline))] + [headline[c].values for c in rr.COVARIATES])
lab = headline["in_science_park"].values.astype(float)
beta = np.linalg.lstsq(Xc, lab, rcond=None)[0]; fitted = Xc @ beta; resid = lab - fitted
alpha = rr.mle_dispersion(headline); X, y = rr.design_matrix(headline)
fam = sm.families.NegativeBinomial(alpha=alpha)
rng_fl = np.random.default_rng(SEED + 7); cfl = np.empty(N_PERM)
for i in range(N_PERM):
    X[:, 1] = fitted + rng_fl.permutation(resid)
    cfl[i] = sm.GLM(y, X, family=fam).fit().params[1]
X[:, 1] = lab
diag["freedman_lane"] = {"permutation_p": perm_p(head["coef"], cfl), "sd": float(cfl.std(ddof=1)),
                         "p95_abs_over_sd": float(np.percentile(np.abs(cfl), 95) / cfl.std(ddof=1)),
                         "excess_kurtosis": float(stats.kurtosis(cfl))}
log(f"   Freedman-Lane: perm p {diag['freedman_lane']['permutation_p']:.4f}")
# where do the extreme null draws come from? treated-hub loading
hub = np.empty(N_PERM)
alpha = rr.mle_dispersion(headline); X, y = rr.design_matrix(headline); rng2 = np.random.default_rng(rr.SEED_PERMUTATION)
top = headline["degree"].values >= np.percentile(headline["degree"].values, 99)
for i in range(N_PERM):
    p = rng2.permutation(lab); hub[i] = p[top].sum()
diag["tail_diagnosis"] = {"corr_abs_coef_with_top1pct_treated": float(np.corrcoef(np.abs(c_un), hub)[0, 1]),
                          "mean_top1pct_treated_in_top5pct_draws": float(hub[np.abs(c_un) >= np.percentile(np.abs(c_un), 95)].mean()),
                          "mean_top1pct_treated_overall": float(hub.mean())}
RESULTS["null_diagnostics"] = diag
save()
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    ax[0].hist(c_un, bins=80, color="0.6"); ax[0].axvline(head["coef"], color="k"); ax[0].axvline(-head["coef"], color="k", ls=":")
    ax[0].set_title("Unrestricted label permutation (5,000)", fontsize=9); ax[0].set_xlabel("permuted park coefficient (log IRR)", fontsize=8)
    stats.probplot(c_un, dist="norm", plot=ax[1]); ax[1].set_title("Normal Q-Q", fontsize=9); ax[1].get_lines()[0].set(markersize=2, color="0.4")
    for a in ax: a.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(OUT / "Figure_S1_permutation_null.png", dpi=300); plt.savefig(OUT / "Figure_S1_permutation_null.pdf")
except Exception as e:
    log(f"   figure skipped: {e}")

# --------------------------------------------------------------------------- #
# H. park-level dependence
# --------------------------------------------------------------------------- #
log("H. catchment clustering")
# nearest-park catchment per institution: majority over its affiliation strings
nearest = park.set_index("Raw_Affiliation")["nearest_park_name"].to_dict()
votes = defaultdict(list)
for row in affil[["Raw_Affiliation", "Standardized_Institutions"]].dropna().itertuples():
    nm = nearest.get(row.Raw_Affiliation)
    if isinstance(nm, str):
        for inst in rr.split_institutions(row.Standardized_Institutions):
            votes[inst].append(nm)
catch = headline["id"].map(lambda i: Counter(votes[i]).most_common(1)[0][0] if votes.get(i) else "__none__")
codes = pd.factorize(catch)[0]
cl = fit(headline, cov_type="cluster", groups=codes)
countries = rr.country_lookup(affil)
ccodes = pd.factorize(headline["id"].map(countries).fillna("__none__"))[0]
clc = fit(headline, cov_type="cluster", groups=ccodes)
treated_catch = catch[headline["in_science_park"] == 1]
RESULTS["park_level"] = {
    "catchments": int(len(set(catch))), "catchments_with_treated": int(treated_catch.nunique()),
    "treated_share_top5_catchments": float(sum(n for _, n in Counter(treated_catch).most_common(5)) / max(1, len(treated_catch))),
    "largest_treated_catchments": Counter(treated_catch).most_common(8),
    "cluster_catchment": cl, "cluster_country": clc, "clusters_country": int(len(set(ccodes))),
}
# catchment-level permutation: permute the per-catchment treated counts across
# catchments, then draw that many treated institutions inside each receiving catchment
tc = Counter(treated_catch); catch_ids = list(set(catch)); counts = np.array([tc.get(c, 0) for c in catch_ids])
members = {c: np.where(catch.values == c)[0] for c in catch_ids}
alpha = rr.mle_dispersion(headline); X, y = rr.design_matrix(headline); fam = sm.families.NegativeBinomial(alpha=alpha)
rng_c = np.random.default_rng(SEED + 11); cc = []
for i in range(N_PERM):
    perm_counts = rng_c.permutation(counts); lab_new = np.zeros(len(headline))
    for c_id, k in zip(catch_ids, perm_counts):
        if k:
            idx = members[c_id]; take = min(k, len(idx))
            lab_new[rng_c.choice(idx, take, replace=False)] = 1
    X[:, 1] = lab_new
    cc.append(sm.GLM(y, X, family=fam).fit().params[1])
X[:, 1] = lab; cc = np.array(cc)
RESULTS["park_level"]["catchment_permutation"] = {"permutation_p": perm_p(head["coef"], cc), "sd": float(cc.std(ddof=1)),
                                                  "p95_abs_over_sd": float(np.percentile(np.abs(cc), 95) / cc.std(ddof=1)), "refits": N_PERM}
log(f"   catchment-clustered SE {cl['se']:.4f} p {cl['model_p']:.4f}; country-clustered p {clc['model_p']:.4f}; catchment perm p {RESULTS['park_level']['catchment_permutation']['permutation_p']:.4f}")
save()

# --------------------------------------------------------------------------- #
# I. sandwich vs bootstrap
# --------------------------------------------------------------------------- #
log("I. sandwich SEs")
hc = {k: fit(headline, cov_type=k) for k in ("HC0", "HC3")}
boot = rr.bootstrap_interval(headline)
RESULTS["standard_errors"] = {"model_se": head["se"], "hc0_se": hc["HC0"]["se"], "hc3_se": hc["HC3"]["se"],
                              "bootstrap_se": boot["se_log"], "bootstrap_ci95": [boot["ci95_low"], boot["ci95_high"]],
                              "hc0_p": hc["HC0"]["model_p"], "hc3_ci95": [hc["HC3"]["ci95_low"], hc["HC3"]["ci95_high"]],
                              "cap10_model_ci95_low": cap10_fit["ci95_low"],
                              "criterion1_bound_model": head["ci95_low"], "criterion1_bound_bootstrap": boot["ci95_low"],
                              "cap10_above_bootstrap_bound": cap10_fit["irr"] > boot["ci95_low"]}
log(f"   model {head['se']:.4f} HC0 {hc['HC0']['se']:.4f} HC3 {hc['HC3']['se']:.4f} bootstrap {boot['se_log']:.4f}")
save()

# --------------------------------------------------------------------------- #
# J. geography of hyperauthored works
# --------------------------------------------------------------------------- #
log("J. geography")
coord = {n["id"]: (n["lat"], n["lon"]) for n in nodes}
cvotes = defaultdict(Counter)
for row in affil[["Standardized_Institutions", "Institution_Countries"]].dropna().itertuples():
    insts = rr.split_institutions(row.Standardized_Institutions); ctry = [c.strip() for c in str(row.Institution_Countries).split(",") if c.strip()]
    if len(insts) == 1 and len(ctry) >= 1:
        cvotes[insts[0]][ctry[0]] += 1
    elif len(insts) == len(ctry):
        for i_, c_ in zip(insts, ctry): cvotes[i_][c_] += 1
country = {i: c.most_common(1)[0][0] for i, c in cvotes.items()}
treated_ids = set(headline.loc[headline.in_science_park == 1, "id"])


def hav(a, b):
    la1, lo1 = map(math.radians, a); la2, lo2 = map(math.radians, b)
    d = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(d))


def geo_stats(works):
    same = tot = local50 = 0; wmed = []; tsame = ttot = tlocal = 0
    for p in works:
        inst = pi[p]; ds = []
        for a, b in combinations(inst, 2):
            ca, cb = country.get(a), country.get(b)
            if ca and cb:
                tot += 1; same += ca == cb
                if a in treated_ids or b in treated_ids:
                    ttot += 1; tsame += ca == cb
            if a in coord and b in coord:
                d = hav(coord[a], coord[b]); ds.append(d); local50 += d < 50
                if a in treated_ids or b in treated_ids:
                    tlocal += d < 50
        if ds: wmed.append(np.median(ds))
    return {"works": len(works), "pairs_with_country": tot, "same_country_share": same / tot if tot else None,
            "pairs_within_50km_share": local50 / max(1, sum(len(pi[p]) * (len(pi[p]) - 1) // 2 for p in works)),
            "median_of_work_median_distance_km": float(np.median(wmed)) if wmed else None,
            "treated_pairs_same_country_share": tsame / ttot if ttot else None,
            "treated_pairs_within_50km_share": tlocal / ttot if ttot else None}


RESULTS["geography"] = {"works_ge10": geo_stats(big), "works_2_to_9": geo_stats(small),
                        "works_ge6": geo_stats([p for p in pi if len(pi[p]) >= 6])}
log(f"   >=10: same-country {RESULTS['geography']['works_ge10']['same_country_share']:.3f}, <50km {RESULTS['geography']['works_ge10']['pairs_within_50km_share']:.3f}; 2-9: {RESULTS['geography']['works_2_to_9']['same_country_share']:.3f}, {RESULTS['geography']['works_2_to_9']['pairs_within_50km_share']:.3f}")
save()
log("done")
