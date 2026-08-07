"""
Institution-size control model, refitted with the dispersion parameter estimated.

Reads:  affil.csv or affil_full.csv, park_matches.csv, combined.csv, std_nodes.json.
Writes: nothing; results are printed.

Two things the size-control model in 10_robustness_checks.py leaves open are
settled here: the model is refitted with smf.negativebinomial(), which estimates
alpha by MLE instead of the GLM family's fixed alpha=1, and VIF is recomputed on
the full five-predictor specification, since institutional size (log_papers) and
the surrounding density of universities and research institutes (log_univ) can
overlap conceptually.

Run this from the same directory as 10_robustness_checks.py: it is loaded with
importlib for build_std_lookup / build_regression_df, because a module name
beginning with a digit cannot be imported with a normal import statement.
"""

import math
import warnings
import importlib

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

_robustness = importlib.import_module("10_robustness_checks")
build_std_lookup = _robustness.build_std_lookup
build_regression_df = _robustness.build_regression_df

warnings.filterwarnings("ignore")


def compute_vif(df, predictors):
    X = sm.add_constant(df[predictors])
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


def _load_affil_csv():
    """Accept either filename so the same script serves both dataset sizes."""
    import os
    for candidate in ("affil.csv", "affil_full.csv"):
        if os.path.exists(candidate):
            print(f"reading {candidate}")
            return pd.read_csv(candidate)
    raise FileNotFoundError("neither affil.csv nor affil_full.csv found.")


def main():
    df_affil = _load_affil_csv()
    df_park = pd.read_csv("park_matches.csv")
    df_combined = pd.read_csv("combined.csv")
    nodes = json.load(open("std_nodes.json", encoding="utf-8"))

    std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
    df = build_regression_df(nodes, std_lookup)
    print(f"regression sample: {len(df)} standardised institutions "
          f"({df['in_science_park'].sum()} in parks)\n")

    formula_size = "degree ~ in_science_park + log_univ + log_station + log_junction + log_papers"

    print("=" * 70)
    print("Size-control model: negative binomial with alpha estimated by MLE")
    print("=" * 70)
    nb_size = smf.negativebinomial(formula_size, data=df).fit(disp=0)
    poisson_size = smf.poisson(formula_size, data=df).fit(disp=0)

    alpha = nb_size.params.get("alpha", None)
    if alpha is None:
        alpha = math.exp(nb_size.params.get("lnalpha", float("nan")))
    print(f"dispersion alpha = {alpha:.4f}")
    print(f"NB AIC = {nb_size.aic:.1f}, Poisson AIC = {poisson_size.aic:.1f}")
    lr_stat = 2 * (nb_size.llf - poisson_size.llf)
    from scipy import stats as scistats
    lr_p = 1 - scistats.chi2.cdf(lr_stat, df=1)
    print(f"likelihood-ratio test (NB vs Poisson): LR = {lr_stat:.1f}, p = {lr_p:.4g}")

    print("\nfull model summary:")
    print(nb_size.summary())

    print("\ncoefficients / IRR / 95% CI:")
    ci = nb_size.conf_int()
    param_names = ["Intercept", "in_science_park", "log_univ", "log_station", "log_junction", "log_papers"]
    for name in param_names:
        coef = nb_size.params[name]
        pval = nb_size.pvalues[name]
        irr = math.exp(coef)
        lo = math.exp(ci.loc[name, 0])
        hi = math.exp(ci.loc[name, 1])
        line = "  {:20s} coef={:8.4f}  p={:8.4g}  IRR={:6.3f}  95% CI=[{:.3f}, {:.3f}]".format(
            name, coef, pval, irr, lo, hi
        )
        print(line)

    print("")
    print("=" * 70)
    print("VIF for the five-predictor model (including log_papers)")
    print("=" * 70)
    predictors = ["in_science_park", "log_univ", "log_station", "log_junction", "log_papers"]
    vif_df = compute_vif(df, predictors)
    print(vif_df.to_string(index=False))

    print("")
    print("reference: Pearson correlation of log_papers with log_univ and with park proximity")
    corr_univ = df["log_papers"].corr(df["log_univ"])
    corr_park = df["log_papers"].corr(df["in_science_park"])
    print("  corr(log_papers, log_univ) = {:.4f}".format(corr_univ))
    print("  corr(log_papers, in_science_park) = {:.4f}".format(corr_park))

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
