"""
robustness_v3.py 的 regression_with_size_control() 有兩個沒補齊的地方：

  1. 該迴歸仍然用 sm.families.NegativeBinomial() 的 GLM，alpha 被固定=1，
     跟論文正文 Table 2 已經修正過的問題（審稿人一：未報告/未正確估計離散
     參數）是同一個坑，只是換了個函數又踩一次——加入 log_papers 之後沒有
     跟著换成 smf.negativebinomial()（MLE 估計 alpha）。

  2. VIF 只在原本 4 個變數的模型上算過（park + 3 個密度共變量），
     沒有把 log_papers 也放進去一起檢查——log_papers（機構規模）跟
     log_univ（鄰近大學/研究機構密度）兩者概念上有重疊的可能
     （大機構本身也可能同時是「大學/研究機構」而被鄰近密度查詢算到),
     所以在最終的 5 變數模型上要重新算一次 VIF，不能只看原本 4 變數的結果。

這支腳本只修正、重跑這兩點，其他 robustness_v3.py 已經算過的維持不動。

執行方式：跟 10_robustness_checks.py 放在同一個資料夾下執行（本腳本用
importlib 動態載入它，取用 build_std_lookup / build_regression_df 兩個函數，
因為檔名開頭是數字，沒辦法用一般的 import 語法）。
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
    """2026-07-29：55k 規模檔名是 affil_full.csv，兩個都找找看，找不到才報錯。"""
    import os
    for candidate in ("affil.csv", "affil_full.csv"):
        if os.path.exists(candidate):
            print(f"讀取 {candidate}")
            return pd.read_csv(candidate)
    raise FileNotFoundError("找不到 affil.csv 或 affil_full.csv。")


def main():
    df_affil = _load_affil_csv()
    df_park = pd.read_csv("park_matches.csv")
    df_combined = pd.read_csv("combined.csv")
    nodes = json.load(open("std_nodes.json", encoding="utf-8"))

    std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
    df = build_regression_df(nodes, std_lookup)
    print(f"迴歸樣本：{len(df)} 個標準化機構（園區內 {df['in_science_park'].sum()}）\n")

    formula_size = "degree ~ in_science_park + log_univ + log_station + log_junction + log_papers"

    print("=" * 70)
    print("機構規模控制模型：正確估計 alpha 的負二項模型（取代原本固定 alpha=1 的 GLM 版本）")
    print("=" * 70)
    nb_size = smf.negativebinomial(formula_size, data=df).fit(disp=0)
    poisson_size = smf.poisson(formula_size, data=df).fit(disp=0)

    alpha = nb_size.params.get("alpha", None)
    if alpha is None:
        alpha = math.exp(nb_size.params.get("lnalpha", float("nan")))
    print(f"離散參數 alpha = {alpha:.4f}")
    print(f"NB AIC = {nb_size.aic:.1f}，Poisson AIC = {poisson_size.aic:.1f}")
    lr_stat = 2 * (nb_size.llf - poisson_size.llf)
    from scipy import stats as scistats
    lr_p = 1 - scistats.chi2.cdf(lr_stat, df=1)
    print(f"概似比檢定（NB vs Poisson）：LR = {lr_stat:.1f}，p = {lr_p:.4g}")

    print("\n完整模型摘要：")
    print(nb_size.summary())

    print("\n係數 / IRR / 95% CI：")
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
    print("5 變數模型（含 log_papers）的 VIF 共線性診斷")
    print("=" * 70)
    predictors = ["in_science_park", "log_univ", "log_station", "log_junction", "log_papers"]
    vif_df = compute_vif(df, predictors)
    print(vif_df.to_string(index=False))

    print("")
    print("對照：log_papers 分別跟 log_univ / park proximity 的皮爾森相關係數")
    corr_univ = df["log_papers"].corr(df["log_univ"])
    corr_park = df["log_papers"].corr(df["in_science_park"])
    print("  corr(log_papers, log_univ) = {:.4f}".format(corr_univ))
    print("  corr(log_papers, in_science_park) = {:.4f}".format(corr_park))

    print("")
    print("完成。")


if __name__ == "__main__":
    main()
