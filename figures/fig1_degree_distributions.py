#!/usr/bin/env python3
"""Figure 1 - degree-centrality distributions under the two node-identity schemes.

Rebuilt from the deposited data, with one labelling rule on both sides:
distance-only 2,000 m classification; strings labelled directly (naive scheme),
institutions by majority vote over their strings (entity-resolved scheme).

Run from the repository root after `python prepare_data.py`:
    python figures/fig1_degree_distributions.py
Writes figures/Figure_1.png and figures/Figure_1.pdf.
"""
import importlib.util, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]; FULL = ROOT / "data" / "full_run"
spec = importlib.util.spec_from_file_location("rr", ROOT / "analysis" / "15_reviewer_response_analyses.py")
rr = importlib.util.module_from_spec(spec); cwd = os.getcwd(); os.chdir(FULL); spec.loader.exec_module(rr)
affil, park, combined, geocoded, nodes, _ = rr.load_inputs(); density2 = pd.read_csv("density_2000m_v2.csv"); os.chdir(cwd)

# naive scheme: raw strings with coordinates, labelled by distance to nearest park
naive = rr.build_naive_frame(affil, park, combined)
dist = park.set_index("Raw_Affiliation")["distance_to_park_m"].to_dict()
naive["park"] = naive["id"].map(lambda s: int(dist.get(s) is not None and not pd.isna(dist.get(s)) and dist.get(s) <= 2000))
# entity-resolved scheme: 7,886 geocoded institutions, majority vote at 2,000 m
dmap2 = rr.density_by_affiliation(geocoded, density2)
lookup2 = rr.build_institution_lookup(affil, park, combined, 2000, dmap2)
resolved = rr.build_frame(nodes, lookup2); resolved["park"] = resolved["in_science_park"]


def summarise(df):
    a = df.loc[df.park == 1, "degree"].values.astype(float); b = df.loc[df.park == 0, "degree"].values.astype(float)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b); allv = np.concatenate([a, b]); _, cnt = np.unique(allv, return_counts=True)
    N = n1 + n2; sd = np.sqrt(n1 * n2 / 12 * ((N + 1) - (cnt ** 3 - cnt).sum() / (N * (N - 1)))); z = (u.statistic - n1 * n2 / 2) / sd
    logp = stats.norm.logsf(abs(z)) + np.log(2)
    return {"park": a, "nonpark": b, "r": 2 * u.statistic / (n1 * n2) - 1, "cles": u.statistic / (n1 * n2), "p": float(np.exp(logp)),
            "mean_p": a.mean(), "mean_n": b.mean(), "med_p": np.median(a), "med_n": np.median(b), "n_park": n1, "n_non": n2}


def ptxt(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e; return f"{m:.1f}×10$^{{{e}}}$"
    return f"{p:.4f}".lstrip("0")


D = {"naive": summarise(naive), "resolved": summarise(resolved)}
GREY, RED = "#808790", "#B5484A"; INK, MUTED, SURFACE = "#1F2328", "#5A6169", "#FFFFFF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#B8BDC2", "axes.linewidth": 0.8,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "pdf.fonttype": 42})
PANELS = [("naive", "Naive scheme: raw affiliation strings as nodes"), ("resolved", "Entity-resolved scheme: standardized institutions")]
fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.0), sharey=True, gridspec_kw={"wspace": 0.06})
for ax, (key, title) in zip(axes, PANELS):
    d = D[key]; data = [d["nonpark"], d["park"]]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.78, showextrema=False, showmedians=False, bw_method=0.25)
    for body, col in zip(parts["bodies"], [GREY, RED]):
        v = body.get_paths()[0].vertices; v[:, 1] = np.clip(v[:, 1], 0, None)
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(1.0); body.set_linewidth(0.8)
    ax.boxplot(data, positions=[0, 1], widths=0.16, showfliers=False, patch_artist=True,
               medianprops=dict(color=INK, linewidth=1.6), whiskerprops=dict(color=INK, linewidth=1.0),
               capprops=dict(color=INK, linewidth=1.0), boxprops=dict(facecolor="white", edgecolor=INK, linewidth=1.0))
    for arr, x in zip(data, [0, 1]):
        ax.plot(x, arr.mean(), marker="D", markersize=6, color="white", markeredgecolor=INK, markeredgewidth=1.1, zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Non-park\n(n = {d['n_non']:,})", f"Park-affiliated\n(n = {d['n_park']:,})"], color=INK)
    ax.set_xlim(-0.62, 1.62); ax.set_title(title, fontsize=9.8, color=INK, pad=16, fontweight="bold")
    ax.text(0.5, 1.012, f"N = {d['n_park'] + d['n_non']:,} nodes", transform=ax.transAxes, ha="center", fontsize=8.4, color=MUTED)
    ax.tick_params(axis="both", colors=MUTED, length=3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E6E8EA", linewidth=0.7); ax.set_axisbelow(True)
    txt = (f"Mann–Whitney $p$ = {ptxt(d['p'])}\nrank-biserial $r$ = {d['r']:.3f}   ·   CLES = {d['cles']:.3f}\n"
           f"mean {d['mean_p']:.2f} (park) vs {d['mean_n']:.2f} (non-park)   ·   median {d['med_p']:.0f} vs {d['med_n']:.0f}")
    ax.text(0.5, -0.165, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8.6, color=INK, linespacing=1.55)
axes[0].set_yscale("symlog", linthresh=1); axes[0].set_ylim(-0.35, 520)
axes[0].set_ylabel("Degree centrality (symlog scale)", color=INK, fontsize=9.4)
fig.legend(handles=[Patch(facecolor=GREY, edgecolor=GREY, label="Non-park"), Patch(facecolor=RED, edgecolor=RED, label="Park-affiliated"),
                    Line2D([], [], marker="D", color="white", markeredgecolor=INK, markersize=6, linestyle="none", label="Mean"),
                    Line2D([], [], color=INK, linewidth=1.6, label="Median")],
           loc="lower center", ncol=4, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.012), handlelength=1.5, columnspacing=2.2)
fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.30)
out = ROOT / "figures"; fig.savefig(out / "Figure_1.png", dpi=300, facecolor=SURFACE); fig.savefig(out / "Figure_1.pdf", facecolor=SURFACE)
print({k: {kk: (round(vv, 4) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in v.items() if kk not in ("park", "nonpark")} for k, v in D.items()})
