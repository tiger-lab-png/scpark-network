import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

D = json.load(open("fig3_data.json"))
GREY, RED = "#808790", "#B5484A"          # validated: CVD ΔE 10.1 deutan, 16.9 normal, contrast >= 3:1
INK, MUTED, SURFACE = "#1F2328", "#5A6169", "#FCFCFB"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#B8BDC2", "axes.linewidth": 0.8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

PANELS = [
    ("naive",    "Naive scheme — raw affiliation strings as nodes",
     "N = 69,455 nodes", "Mann–Whitney $p$ = 8.9×10$^{-23}$"),
    ("resolved", "Entity-resolved scheme — standardized institutions",
     "N = 7,886 nodes",  "Mann–Whitney $p$ = .0073"),
]

fig, axes = plt.subplots(1, 2, figsize=(10.0, 6.6), sharey=True,
                         gridspec_kw={"wspace": 0.06})

for ax, (key, title, nlab, plab) in zip(axes, PANELS):
    d = D[key]
    data = [np.array(d["nonpark"]), np.array(d["park"])]
    colors = [GREY, RED]

    parts = ax.violinplot(data, positions=[0, 1], widths=0.78,
                          showextrema=False, showmedians=False, bw_method=0.25)
    for body, col in zip(parts["bodies"], colors):
        # clip the kernel tail at the observed range: degree cannot be negative
        v = body.get_paths()[0].vertices
        v[:, 1] = np.clip(v[:, 1], 0, None)
        body.set_facecolor(col); body.set_edgecolor(col)
        body.set_alpha(1.0); body.set_linewidth(0.8)

    bp = ax.boxplot(data, positions=[0, 1], widths=0.16, showfliers=False,
                    patch_artist=True, medianprops=dict(color=INK, linewidth=1.6),
                    whiskerprops=dict(color=INK, linewidth=1.0),
                    capprops=dict(color=INK, linewidth=1.0),
                    boxprops=dict(facecolor="white", edgecolor=INK, linewidth=1.0))
    for arr, x in zip(data, [0, 1]):
        ax.plot(x, arr.mean(), marker="D", markersize=6, color="white",
                markeredgecolor=INK, markeredgewidth=1.1, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Non-park\n(n = {d['n_non']:,})",
                        f"Park-affiliated\n(n = {d['n_park']:,})"], color=INK)
    ax.set_xlim(-0.62, 1.62)
    ax.set_title(title, fontsize=9.8, color=INK, pad=13, fontweight="bold")
    ax.text(0.5, 1.012, nlab, transform=ax.transAxes, ha="center",
            fontsize=8.4, color=MUTED)
    ax.tick_params(axis="both", colors=MUTED, length=3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E6E8EA", linewidth=0.7)
    ax.set_axisbelow(True)

    stats = (f"{plab}\n"
             f"rank-biserial $r$ = {d['r']:.3f}   ·   CLES = {d['cles']:.3f}\n"
             f"mean {d['mean_p']:.2f} (park) vs {d['mean_n']:.2f} (non-park)   ·   "
             f"median {d['med_p']:.0f} vs {d['med_n']:.0f}")
    ax.text(0.5, -0.165, stats, transform=ax.transAxes, ha="center", va="top",
            fontsize=8.6, color=INK, linespacing=1.55)

axes[0].set_yscale("symlog", linthresh=1)
axes[0].set_ylim(-0.35, 520)
axes[0].set_ylabel("Degree centrality  (symlog scale)", color=INK, fontsize=9.4)

fig.suptitle("Entity resolution changes the certainty, not the effect",
             fontsize=12.4, color=INK, fontweight="bold", y=0.985)
fig.text(0.5, 0.935,
         "The two node-identity schemes return the same standardized effect size "
         "($r$ = .114 vs .121) and $p$ values twenty orders of magnitude apart.",
         ha="center", fontsize=9.2, color=MUTED)

fig.legend(handles=[Patch(facecolor=GREY, edgecolor=GREY, label="Non-park"),
                    Patch(facecolor=RED, edgecolor=RED, label="Park-affiliated"),
                    Line2D([], [], marker="D", color="white", markeredgecolor=INK,
                           markersize=6, linestyle="none", label="Mean"),
                    Line2D([], [], color=INK, linewidth=1.6, label="Median")],
           loc="lower center", ncol=4, frameon=False, fontsize=8.8,
           bbox_to_anchor=(0.5, 0.012), handlelength=1.5, columnspacing=2.2)

fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.275)
fig.savefig("Fig3_degree_distributions.png", dpi=300, facecolor=SURFACE)
fig.savefig("Fig3_degree_distributions.pdf", facecolor=SURFACE)
print("saved")
