#!/usr/bin/env python3
"""Figure 2 - the headline park coefficient under institution-count caps, fractional
counting, joint perturbation, and a placebo exclusion, all recomputed from the
deposited data.

Run from the repository root after `python prepare_data.py`:
    python figures/fig2_forest_caps.py            # full: 5,000-refit permutations, 2,000 placebo draws (~20 min)
    python figures/fig2_forest_caps.py --fast     # 500 refits, 200 placebo draws (~3 min; p-values approximate)
    python figures/fig2_forest_caps.py --replot   # redraw from the saved fig2_values.json
Writes figures/Figure_2.png and figures/Figure_2.pdf and prints the plotted numbers.
"""
import argparse, importlib.util, json, math, os, time
from pathlib import Path
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true"); ap.add_argument("--replot", action="store_true", help="redraw from figures/fig2_values.json without recomputing"); args = ap.parse_args()
N_PERM = 500 if args.fast else 5000; N_PLACEBO = 200 if args.fast else 2000
ROOT = Path(__file__).resolve().parents[1]; FULL = ROOT / "data" / "full_run"
T0 = time.time()
if args.replot:
    order = json.load(open(ROOT / "figures" / "fig2_values.json")); rows = [order[0]] + order[2:]
else:
    spec = importlib.util.spec_from_file_location("rr", ROOT / "analysis" / "15_reviewer_response_analyses.py")
    rr = importlib.util.module_from_spec(spec); cwd = os.getcwd(); os.chdir(FULL); spec.loader.exec_module(rr)
    affil, park, combined, geocoded, nodes, density = rr.load_inputs(); os.chdir(cwd)
    dmap = rr.density_by_affiliation(geocoded, density)
    lookup = rr.build_institution_lookup(affil, park, combined, rr.TREATMENT_RADIUS_M, dmap)
    headline = rr.build_frame(nodes, lookup); pi = rr.papers_to_institutions(affil)
    TI = {p: len(v) * (len(v) - 1) // 2 for p, v in pi.items()}; TI_total = sum(TI.values())


    def capped(cap=None, fractional=False, exclude=None):
        if exclude is not None:
            pi_use = {p: v for p, v in pi.items() if p not in exclude}; present = set(); [present.update(v) for v in pi_use.values()]
            deg = rr.rebuild_degree(pi_use)
        else:
            present = rr.institutions_present(pi, cap) if cap else None; deg = rr.rebuild_degree(pi, exclude_at=cap, fractional=fractional)
        f = headline if present is None else headline[headline.id.isin(present)]
        f = f.copy(); f["degree"] = f.id.map(lambda i: deg.get(i, 0.0 if fractional else 0)); return f


    def fit_row(frame, label, n_perm=N_PERM):
        s = rr.summarise(frame, label=label); s.update(rr.permutation_p(frame, n=n_perm)); return s


    rows = [fit_row(headline, "Full sample (headline)")]
    print(f"[{time.time()-T0:5.0f}s] headline IRR {rows[0]['irr']:.4f} perm p {rows[0]['permutation_p']:.4f}", flush=True)
    LOG_HEAD = math.log(rows[0]["irr"])
    # placebo: random work sets matched to the >=10 cap's tie-instance share
    big = [p for p, v in pi.items() if len(v) >= 10]; share = sum(TI[p] for p in big) / TI_total
    multi = [p for p, v in pi.items() if len(v) >= 2]; rng = np.random.default_rng(20260905); irr_pl = []
    for d in range(N_PLACEBO):
        order = rng.permutation(len(multi)); acc = 0; chosen = set()
        for j in order:
            p = multi[j]; chosen.add(p); acc += TI[p]
            if acc / TI_total >= share: break
        irr_pl.append(rr.summarise(capped(exclude=chosen))["irr"])
    irr_pl = np.array(irr_pl); removed_pl = 1 - np.log(irr_pl) / LOG_HEAD
    print(f"[{time.time()-T0:5.0f}s] placebo mean {irr_pl.mean():.4f} 2.5–97.5th {np.percentile(irr_pl, 2.5):.3f}–{np.percentile(irr_pl, 97.5):.3f}", flush=True)
    labels = {6: "Excl. works with ≥ 6 institutions", 8: "Excl. ≥ 8", 10: "Excl. ≥ 10", 15: "Excl. ≥ 15"}
    for cap in (6, 8, 10, 15):
        share_w = np.mean([len(v) >= cap for v in pi.values()]) * 100
        r = fit_row(capped(cap=cap), f"{labels[cap]} ({share_w:.2f}%)"); rows.append(r)
        print(f"[{time.time()-T0:5.0f}s] cap {cap}: IRR {r['irr']:.4f} perm p {r['permutation_p']:.4f}", flush=True)
    rows.append(fit_row(capped(fractional=True), "Fractional counting 1/(k−1)"))
    joint = capped(cap=10); bad = joint["park_name"].astype(str).str.contains(rr.MISCLASSIFIED_PARKS, case=False, na=False); joint.loc[bad, "in_science_park"] = 0
    rows.append(fit_row(joint, "Joint perturbation"))
    for r in rows[1:]:
        r["removed"] = 1 - math.log(r["irr"]) / LOG_HEAD
    cap10_removed = [r for r in rows if r["label"].startswith("Excl. ≥ 10")][0]["removed"]
    placebo = {"label": "Placebo: random works, equal tie mass", "irr": float(irr_pl.mean()), "ci95_low": float(np.percentile(irr_pl, 2.5)),
               "ci95_high": float(np.percentile(irr_pl, 97.5)), "share_ge_cap10": float(np.mean(removed_pl >= cap10_removed)),
               "removed": float(removed_pl.mean()), "draws": N_PLACEBO}
    order = [rows[0], placebo] + rows[1:]
    json.dump(order, open(ROOT / "figures" / "fig2_values.json", "w"), indent=1, default=float)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
fig = plt.figure(figsize=(7.5, 4.2)); ax = fig.add_axes([0.33, 0.26, 0.25, 0.67]); y = np.arange(len(order))[::-1]
ax.axvspan(rows[0]["ci95_low"], rows[0]["ci95_high"], color="0.93", zorder=0); ax.axvline(1.0, color="0.55", lw=0.9, ls="--", zorder=1)
for yi, r in zip(y, order):
    if "draws" in r:
        ax.plot([r["ci95_low"], r["ci95_high"]], [yi, yi], color="0.55", lw=4, alpha=.5, solid_capstyle="butt", zorder=2)
        ax.plot(r["irr"], yi, marker="D", ms=5, mfc="0.55", mec="0.55", ls="", zorder=3); continue
    ax.plot([r["ci95_low"], r["ci95_high"]], [yi, yi], color="0.3", lw=1.3, zorder=2)
    ax.plot(r["irr"], yi, marker="o", ms=7, mfc=("0.15" if r["permutation_p"] < .05 else "white"), mec="0.15", mew=1.3, ls="", zorder=3)
ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in order], fontsize=7.6); ax.set_ylim(-0.7, len(order) - 0.3)
ax.set_xscale("log"); ax.set_xlim(0.95, 1.42); ax.xaxis.set_major_locator(FixedLocator([1.0, 1.1, 1.2, 1.3, 1.4])); ax.xaxis.set_minor_locator(FixedLocator([]))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.1f}")); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("IRR for park proximity (5,000 m)", fontsize=8.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
cols = [(1.08, "IRR [95% CI]"), (1.72, "Model p"), (2.03, "Perm. p"), (2.34, "Log-effect\nremoved")]
for x, h in cols: ax.text(x, 1.01, h, transform=ax.transAxes, fontsize=7.6, ha="left", va="bottom", color="0.25", fontweight="bold")
yfrac = lambda yi: (yi - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])


def p3(v): return f"{v:.2g}" if v < .001 else (f"{v:.4f}" if v < .01 else f"{v:.3f}").lstrip("0")


for yi, r in zip(y, order):
    if "draws" in r:
        vals = [f"{r['irr']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]*", "—", ("< .001" if r["share_ge_cap10"] == 0 else p3(r["share_ge_cap10"])) + "†", f"{r['removed']*100:.1f}%"]
    else:
        vals = [f"{r['irr']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]", p3(r["model_p"]), p3(r["permutation_p"]), ("—" if "removed" not in r else f"{r['removed']*100:.1f}%")]
    for (x, _), v in zip(cols, vals): ax.text(x, yfrac(yi), v, transform=ax.transAxes, fontsize=7.6, ha="left", va="center", color="0.15")
ax.plot([], [], marker="o", mfc="0.15", mec="0.15", ls="", label="permutation p < .05"); ax.plot([], [], marker="o", mfc="white", mec="0.15", ls="", label="permutation p ≥ .05")
ax.plot([], [], marker="D", color="0.55", ls="", label="placebo mean and 2.5–97.5th percentiles")
ax.legend(loc="upper left", bbox_to_anchor=(-0.55, -0.20), frameon=False, fontsize=7.2, ncol=3, handletextpad=0.4, columnspacing=1.2)
fig.savefig(ROOT / "figures" / "Figure_2.png", dpi=300); fig.savefig(ROOT / "figures" / "Figure_2.pdf")
for r in order: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("label", "irr", "ci95_low", "ci95_high", "model_p", "permutation_p", "removed", "share_ge_cap10")})
print(f"done in {time.time()-T0:.0f}s")
