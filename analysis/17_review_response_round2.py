#!/usr/bin/env python3
"""17_review_response_round2.py - second-round additions (September 2026).

Blocks
  A  Table 1 recomputed with one labelling rule on both node schemes
     (distance-only 2,000 m: 2,491 strings; 187 majority-vote institutions),
     plus a cluster bootstrap of the between-scheme difference in rank-biserial r
  B  paired institution-bootstrap intervals for the share of log-effect removed
     at each cap and under fractional counting
  C  placebo exclusions matched to the >=6 and >=15 caps' tie-instance shares
  D  prior power recomputed under a data-generating process calibrated to the
     empirical coefficient SD (0.084) rather than the model-based one (0.055)
  E  the decomposition against the July-vintage denominator (IRR 1.167)
  F  subsample benchmarks matched on institutions (~4,800) and on treated (~480)
  G  node-count reconciliation (7,811) and the institutions dropped by the cap
  H  permutation p for the country fixed-effects model
"""
import importlib.util, json, math, os, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]; FULL = ROOT / "data" / "full_run"
OUT = ROOT / "analysis" / "review_response_2026_09"; OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("rr", ROOT / "analysis" / "15_reviewer_response_analyses.py")
rr = importlib.util.module_from_spec(spec); _cwd = os.getcwd(); os.chdir(FULL); spec.loader.exec_module(rr); os.chdir(_cwd)
SEED = 20260907; T0 = time.time(); R = {"settings": {"seed": SEED}}
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
def save(): json.dump(R, open(OUT / "results_round2.json", "w"), indent=1, default=float)

def fit(frame, covariates=rr.COVARIATES):
    alpha = rr.mle_dispersion(frame, covariates); X, y = rr.design_matrix(frame, covariates)
    m = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit(); ci = m.conf_int()[1]
    return {"n": len(frame), "treated": int(frame.in_science_park.sum()), "irr": math.exp(m.params[1]),
            "ci95_low": math.exp(ci[0]), "ci95_high": math.exp(ci[1]), "coef": float(m.params[1]), "se": float(m.bse[1]),
            "model_p": float(m.pvalues[1]), "alpha": float(alpha)}

def perm_p(frame, covariates=rr.COVARIATES, n=5000, seed=rr.SEED_PERMUTATION):
    alpha = rr.mle_dispersion(frame, covariates); X, y = rr.design_matrix(frame, covariates)
    fam = sm.families.NegativeBinomial(alpha=alpha); obs = abs(sm.GLM(y, X, family=fam).fit().params[1])
    labels = X[:, 1].copy(); rng = np.random.default_rng(seed); ex = 0
    for _ in range(n):
        X[:, 1] = rng.permutation(labels); ex += abs(sm.GLM(y, X, family=fam).fit().params[1]) >= obs
    X[:, 1] = labels; return (ex + 1) / (n + 1)

def rank_biserial(a, b):
    u = stats.mannwhitneyu(a, b, alternative="two-sided"); r = 1 - 2 * u.statistic / (len(a) * len(b))
    return -r, float(u.pvalue)  # sign so that positive = treated higher

def cles(a, b):
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic; return float(u / (len(a) * len(b)))

log("loading"); os.chdir(FULL)
affil, park, combined, geocoded, nodes, density5 = rr.load_inputs(); density2 = pd.read_csv("density_2000m_v2.csv"); os.chdir(_cwd)
dmap5 = rr.density_by_affiliation(geocoded, density5); dmap2 = rr.density_by_affiliation(geocoded, density2)
lookup5 = rr.build_institution_lookup(affil, park, combined, 5000, dmap5); headline = rr.build_frame(nodes, lookup5)
lookup2 = rr.build_institution_lookup(affil, park, combined, 2000, dmap2); base2 = rr.build_frame(nodes, lookup2)
pi = rr.papers_to_institutions(affil); head = fit(headline); LOG_HEAD = math.log(head["irr"])
TI = {p: len(v) * (len(v) - 1) // 2 for p, v in pi.items()}; TI_total = sum(TI.values())

# --------------------------------------------------------------------------- A
log("A. Table 1 with one labelling rule")
naive = rr.build_naive_frame(affil, park, combined)          # degree per string, coords present
dist = park.set_index("Raw_Affiliation")["distance_to_park_m"].to_dict()
naive["park_dist2000"] = naive["id"].map(lambda s: 1 if (dist.get(s) is not None and not pd.isna(dist.get(s)) and dist.get(s) <= 2000) else 0)
res = base2[["id", "degree", "in_science_park"]].copy()   # 187 majority-vote (distance-only)
# string -> institution link for cluster bootstrap
link = defaultdict(set)
for row in affil[["Raw_Affiliation", "Standardized_Institutions"]].dropna().itertuples():
    for inst in rr.split_institutions(row.Standardized_Institutions): link[row.Raw_Affiliation].add(inst)
frame_ids = set(res.id); str2inst = {s: next(iter(i & frame_ids)) if (i & frame_ids) else None for s, i in link.items()}
naive["inst"] = naive["id"].map(lambda s: str2inst.get(s))

def table1_row(df, lab, deg="degree"):
    a = df.loc[df[lab] == 1, deg].values.astype(float); b = df.loc[df[lab] == 0, deg].values.astype(float)
    r, p = rank_biserial(a, b)
    return {"n": len(df), "park_n": len(a), "nonpark_n": len(b), "park_mean": float(a.mean()), "park_median": float(np.median(a)),
            "nonpark_mean": float(b.mean()), "nonpark_median": float(np.median(b)), "mw_p": p, "r": float(r), "cles": cles(a, b),
            "mean_ratio": float(a.mean() / b.mean())}
t1 = {"naive_distance_only": table1_row(naive, "park_dist2000"), "naive_combined_flag": table1_row(naive, "in_science_park"),
      "resolved_187": table1_row(res, "in_science_park")}
# bootstrap CIs for r (institution clusters for naive, institutions for resolved) and the difference
rng = np.random.default_rng(SEED); inst_ids = res.id.values; groups = naive.groupby("inst").indices
unlinked = np.where(naive["inst"].isna().values)[0]
r_n, r_r = [], []
for _ in range(2000):
    pick = rng.choice(len(inst_ids), len(inst_ids), replace=True)
    rs = res.iloc[pick]; a = rs.loc[rs.in_science_park == 1, "degree"].values; b = rs.loc[rs.in_science_park == 0, "degree"].values
    r_r.append(rank_biserial(a.astype(float), b.astype(float))[0])
    idx = np.concatenate([groups.get(inst_ids[i], np.array([], dtype=int)) for i in pick] + [rng.choice(unlinked, len(unlinked), replace=True)])
    ns = naive.iloc[idx]; a = ns.loc[ns.park_dist2000 == 1, "degree"].values; b = ns.loc[ns.park_dist2000 == 0, "degree"].values
    r_n.append(rank_biserial(a.astype(float), b.astype(float))[0])
r_n = np.array(r_n); r_r = np.array(r_r); d = r_n - r_r
t1["bootstrap"] = {"reps": 2000, "r_naive_ci95": [float(np.percentile(r_n, 2.5)), float(np.percentile(r_n, 97.5))],
                   "r_resolved_ci95": [float(np.percentile(r_r, 2.5)), float(np.percentile(r_r, 97.5))],
                   "diff_mean": float(d.mean()), "diff_ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))],
                   "diff_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                   "tost_equivalent_within_0.10": bool(np.percentile(d, 5) > -0.10 and np.percentile(d, 95) < 0.10),
                   "tost_equivalent_within_0.05": bool(np.percentile(d, 5) > -0.05 and np.percentile(d, 95) < 0.05)}
R["table1"] = t1; log(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in t1.items() if k != "bootstrap"})); log(json.dumps(t1["bootstrap"])); save()

# --------------------------------------------------------------------------- B
log("B. share intervals")
def capped(base, cap=None, fractional=False):
    present = rr.institutions_present(pi, cap) if cap else None; deg = rr.rebuild_degree(pi, exclude_at=cap, fractional=fractional)
    f = base if present is None else base[base.id.isin(present)]; f = f.copy(); f["degree"] = f.id.map(lambda i: deg.get(i, 0.0 if fractional else 0)); return f
specs = {f"cap{c}": capped(headline, cap=c) for c in (6, 8, 10, 15)}; specs["fractional"] = capped(headline, fractional=True)
alpha_h = rr.mle_dispersion(headline); Xh, yh = rr.design_matrix(headline); fam_h = sm.families.NegativeBinomial(alpha=alpha_h)
shares = {}
for name, f in specs.items():
    alpha_c = rr.mle_dispersion(f); Xc, yc = rr.design_matrix(f); fam_c = sm.families.NegativeBinomial(alpha=alpha_c)
    pos = {i: k for k, i in enumerate(f.id.values)}; hid = headline.id.values
    rng = np.random.default_rng(SEED + 3); sh = []
    for _ in range(1000):
        pick = rng.choice(len(hid), len(hid), replace=True)
        bf = sm.GLM(yh[pick], Xh[pick], family=fam_h).fit().params[1]
        cp = np.array([pos[hid[i]] for i in pick if hid[i] in pos])
        bc = sm.GLM(yc[cp], Xc[cp], family=fam_c).fit().params[1]
        sh.append(1 - bc / bf if bf > 0 else np.nan)
    sh = np.array(sh); sh = sh[np.isfinite(sh)]
    shares[name] = {"point": 1 - math.log(fit(f)["irr"]) / LOG_HEAD, "boot_mean": float(sh.mean()), "ci95": [float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))], "reps": int(len(sh))}
    log(f"   {name}: point {shares[name]['point']:.3f} CI {shares[name]['ci95']}")
R["share_intervals"] = shares; save()

# --------------------------------------------------------------------------- C
log("C. placebo at other caps")
multi = [p for p, v in pi.items() if len(v) >= 2]
def placebo(target, n_draws, seed):
    rng = np.random.default_rng(seed); rem = []
    for _ in range(n_draws):
        order = rng.permutation(len(multi)); acc = 0; chosen = set()
        for j in order:
            p = multi[j]; chosen.add(p); acc += TI[p]
            if acc / TI_total >= target: break
        pi_use = {p: v for p, v in pi.items() if p not in chosen}; present = set(); [present.update(v) for v in pi_use.values()]
        deg = rr.rebuild_degree(pi_use); f = headline[headline.id.isin(present)].copy(); f["degree"] = f.id.map(lambda i: deg.get(i, 0))
        rem.append(1 - math.log(fit(f)["irr"]) / LOG_HEAD)
    rem = np.array(rem); return rem
pl = {}
for cap, n_seed in ((6, 4), (15, 5)):
    big = [p for p, v in pi.items() if len(v) >= cap]; share = sum(TI[p] for p in big) / TI_total
    removed = 1 - math.log(fit(specs[f"cap{cap}"])["irr"]) / LOG_HEAD
    rem = placebo(share, 1000, SEED + n_seed)
    pl[f"cap{cap}"] = {"tie_share": share, "removed_cap": removed, "placebo_mean": float(rem.mean()), "placebo_sd": float(rem.std(ddof=1)),
                       "placebo_p99": float(np.percentile(rem, 99)), "placebo_max": float(rem.max()), "share_ge_cap": float(np.mean(rem >= removed)),
                       "placebo_p": float((np.sum(rem >= removed) + 1) / 1001), "z": float((removed - rem.mean()) / rem.std(ddof=1)), "draws": 1000}
    log(f"   cap{cap}: share {share:.3f} removed {removed:.3f} placebo mean {rem.mean():.3f} max {rem.max():.3f} p {pl[f'cap{cap}']['placebo_p']:.4f}")
R["placebo_other_caps"] = pl; save()

# --------------------------------------------------------------------------- D
log("D. power recalibration")
capf = specs["cap10"]; alpha = rr.mle_dispersion(capf); X, _ = rr.design_matrix(capf); y = capf.degree.values.astype(float)
null_formula = "degree ~ " + " + ".join(rr.COVARIATES)
import statsmodels.formula.api as smf
fitted = smf.glm(null_formula, data=capf, family=sm.families.NegativeBinomial(alpha=alpha)).fit().fittedvalues.values
treated = X[:, 1]; target_sd = 0.0837; crit = 0.16476
def sim_sd(mult, n=400, seed=1):
    rng = np.random.default_rng(seed); size = 1.0 / (alpha * mult); cs = []
    for _ in range(n):
        ys = rng.negative_binomial(size, size / (size + fitted)).astype(float)
        cs.append(rr.park_coefficient(X, ys, alpha)[0])
    return float(np.std(cs, ddof=1))
lo, hi = 1.0, 8.0; sd_lo, sd_hi = sim_sd(lo), sim_sd(hi); log(f"   sd at mult 1: {sd_lo:.4f}; at 8: {sd_hi:.4f}")
for _ in range(7):
    mid = (lo + hi) / 2; s = sim_sd(mid)
    if s < target_sd: lo = mid
    else: hi = mid
mult = (lo + hi) / 2; sd_cal = sim_sd(mult, n=800, seed=2); log(f"   calibrated dispersion multiplier {mult:.3f} (sd {sd_cal:.4f})")
grid = {}
rng = np.random.default_rng(rr.SEED_POWER); size = 1.0 / (alpha * mult)
for irr in (1.0, 1.10, 1.1145, 1.15, 1.25, 1.50):
    mu = fitted * np.power(irr, treated); mh = ph = 0; n_sim = 2000
    for _ in range(n_sim):
        ys = rng.negative_binomial(size, size / (size + mu)).astype(float); c, _, p = rr.park_coefficient(X, ys, alpha)
        mh += (p < .05 and c > 0); ph += (c >= crit)
    grid[str(irr)] = {"power_model_nominal": mh / n_sim, "power_permutation": ph / n_sim, "mc_se": math.sqrt(0.25 / n_sim)}
    log(f"   IRR {irr}: model {mh/n_sim:.3f} perm {ph/n_sim:.3f}")
R["power_recalibrated"] = {"dispersion_multiplier": mult, "simulated_coef_sd": sd_cal, "target_sd": target_sd, "critical_value": crit, "grid": grid, "n_sim": 2000}
save()

# --------------------------------------------------------------------------- E
log("E. July-vintage denominator")
july = combined.set_index("Raw_Affiliation")["univ_research_count"].to_dict()
dmapJ = {k: float(v) for k, v in july.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
lookupJ = rr.build_institution_lookup(affil, park, combined, 5000, dmapJ); baseJ = rr.build_frame(nodes, lookupJ)
fJ = fit(baseJ); lJ = math.log(fJ["irr"]); ej = {"headline": fJ}
for name, f in (("cap10", capped(baseJ, cap=10)), ("fractional", capped(baseJ, fractional=True))):
    r_ = fit(f); r_["log_effect_removed"] = 1 - math.log(r_["irr"]) / lJ; ej[name] = r_
    log(f"   July {name}: IRR {r_['irr']:.4f} removed {r_['log_effect_removed']:.3f}")
R["july_vintage"] = ej; save()

# --------------------------------------------------------------------------- F
log("F. subsample benchmarks matched on institutions / treated")
all_ids = np.array(sorted(affil.Paper_ID.unique())); node_by_id = {n["id"]: n for n in nodes}
def one_draw(W, rng, d):
    chosen = set(rng.choice(all_ids, W, replace=False)); a_sub = affil[affil.Paper_ID.isin(chosen)]
    lk = rr.build_institution_lookup(a_sub, park, combined, 5000, dmap5); pi_sub = rr.papers_to_institutions(a_sub)
    deg = rr.rebuild_degree(pi_sub); present = set(); [present.update(v) for v in pi_sub.values()]
    f = rr.build_frame([node_by_id[i] for i in present if i in node_by_id], lk, degree=deg)
    row = {"draw": d, "works": W, "n": len(f), "treated": int(f.in_science_park.sum())}
    r1 = fit(f); row.update(irr=r1["irr"], model_p=r1["model_p"], perm_p=perm_p(f, n=1000, seed=SEED + 300 + d))
    r3 = fit(f, covariates=rr.COVARIATES + ["log_papers"]); row.update(prod_irr=r3["irr"], prod_p=r3["model_p"])
    present10 = rr.institutions_present(pi_sub, 10); deg10 = rr.rebuild_degree(pi_sub, exclude_at=10)
    f4 = f[f.id.isin(present10)].copy(); f4["degree"] = f4.id.map(lambda i: deg10.get(i, 0)); r4 = fit(f4)
    row.update(cap10_irr=r4["irr"], cap10_perm_p=perm_p(f4, n=1000, seed=SEED + 400 + d))
    row.update(clause1=r1["model_p"] < .05 and r1["irr"] > 1, clause2=row["perm_p"] < .05 and r1["irr"] > 1, clause3=r3["model_p"] < .05 and r3["irr"] > 1, clause4=row["cap10_perm_p"] < .05 and r4["irr"] > 1)
    return row
subs = {}
for label, W, seed_off in (("institutions_4800", 11500, 20), ("treated_480", 16500, 30)):
    rng = np.random.default_rng(SEED + seed_off); rows = []
    for d in range(30):
        rows.append(one_draw(W, rng, d))
        if d % 10 == 0: log(f"   {label} draw {d}: n {rows[-1]['n']} treated {rows[-1]['treated']} IRR {rows[-1]['irr']:.3f} p {rows[-1]['model_p']:.3f} perm {rows[-1]['perm_p']:.3f}")
    df = pd.DataFrame(rows); df.to_csv(OUT / f"subsamples_{label}.csv", index=False)
    subs[label] = {"works": W, "draws": 30, "n_median": float(df.n.median()), "treated_median": float(df.treated.median()), "irr_median": float(df.irr.median()),
                   "pass_clause1": float(df.clause1.mean()), "pass_clause2": float(df.clause2.mean()), "pass_clause3": float(df.clause3.mean()),
                   "pass_1_to_3": float((df.clause1 & df.clause2 & df.clause3).mean()), "pass_all": float((df.clause1 & df.clause2 & df.clause3 & df.clause4).mean()),
                   "cap10_removed_median": float(np.median(1 - np.log(df.cap10_irr) / np.log(df.irr)))}
    log(f"   {label}: n {subs[label]['n_median']:.0f} treated {subs[label]['treated_median']:.0f} pass1 {subs[label]['pass_clause1']:.2f} 1-3 {subs[label]['pass_1_to_3']:.2f} all {subs[label]['pass_all']:.2f}")
R["subsamples_matched"] = subs; save()

# --------------------------------------------------------------------------- G
log("G. node counts and dropped institutions")
deg_all = rr.rebuild_degree(pi); n_all = len({i for v in pi.values() for i in v}); nonisolated = sum(1 for i in {i for v in pi.values() for i in v} if deg_all.get(i, 0) > 0)
edges_all = sum(deg_all.values()) // 2
present10 = rr.institutions_present(pi, 10); dropped = headline[~headline.id.isin(present10)]
R["node_counts"] = {"institutions_in_works_with_institution_data": n_all, "non_isolated": nonisolated, "distinct_edges_all": edges_all,
                    "geocoded_frame": len(headline), "geocoded_zero_degree": int((headline.degree == 0).sum()),
                    "dropped_by_cap10": {"n": len(dropped), "treated": int(dropped.in_science_park.sum()),
                                         "mean_degree_treated": float(dropped.loc[dropped.in_science_park == 1, "degree"].mean()) if dropped.in_science_park.sum() else None,
                                         "mean_degree_control": float(dropped.loc[dropped.in_science_park == 0, "degree"].mean()),
                                         "share_treated_dropped": float(dropped.in_science_park.sum() / headline.in_science_park.sum()),
                                         "share_control_dropped": float((dropped.in_science_park == 0).sum() / (headline.in_science_park == 0).sum())}}
log(json.dumps(R["node_counts"])); save()

# --------------------------------------------------------------------------- H
log("H. country fixed-effects permutation")
countries = rr.country_lookup(affil); fe = headline.copy(); fe["country"] = fe.id.map(countries)
fe = fe.dropna(subset=["country"]); dummies = pd.get_dummies(fe.country, drop_first=True).astype(float)
Xf = np.column_stack([np.ones(len(fe)), fe.in_science_park.values.astype(float)] + [fe[c].values for c in rr.COVARIATES] + [dummies.values])
yf = fe.degree.values.astype(float); alpha_f = rr.mle_dispersion(fe); fam_f = sm.families.NegativeBinomial(alpha=alpha_f)
mf = sm.GLM(yf, Xf, family=fam_f).fit(); obs = abs(mf.params[1]); labels = Xf[:, 1].copy(); rng = np.random.default_rng(SEED + 9); ex = 0; N = 2000
for _ in range(N):
    Xf[:, 1] = rng.permutation(labels); ex += abs(sm.GLM(yf, Xf, family=fam_f).fit().params[1]) >= obs
R["country_fe"] = {"n": len(fe), "countries": int(fe.country.nunique()), "irr": math.exp(mf.params[1]), "model_p": float(mf.pvalues[1]), "se": float(mf.bse[1]), "permutation_p": (ex + 1) / (N + 1), "refits": N}
log(json.dumps(R["country_fe"])); save(); log("done")
