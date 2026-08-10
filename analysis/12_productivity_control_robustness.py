"""
壓力測試 11_institution_size_control.py 算出來的那個邊緣顯著結果
------------------------------------------------------------------
背景：控制機構產能（log_papers）之後，in_science_park 係數 p = .042，
IRR = 1.132，95% CI [1.005, 1.275]——下界緊貼著 1，是個勉強跨過顯著
門檻的結果，方向跟原論文「控制產能後園區效應消失（IRR≈1.00, p=.987）」
的結論相反。論文自己的方法論立場是「一個沒跨過顯著的 p 值不能直接當成
沒有效果」，同樣地，一個剛好跨過顯著的 p 值也不能直接當成「效果存在」
——尤其是在還沒針對這個特定結果做過任何額外檢驗的情況下。這支腳本補上
跟論文對主要結果一樣等級的三個檢查：

  [檢查 1] 距離門檻敏感度：這個邊緣顯著只在 2000m 這個門檻剛好出現，
           還是 500m-5000m 都穩定存在？
  [檢查 2] 標籤排列檢定：注意這裡不能用簡單的平均數差排列檢定（那個沒
           控制 log_univ/log_station/log_junction/log_papers），必須
           每次重新分配園區標籤後，重新配適一次完整的負二項迴歸，才是
           真正對應「控制了這些共變量之後，這個係數還站不站得住腳」
           這個問題的檢定方式。
  [檢查 3] 把這個產能控制模型也一起納入 Holm-Bonferroni 多重比較校正
           （原本 10_robustness_checks.py 只校正三項，沒有這一項）。

用法：跟 10_robustness_checks.py 放在同一個資料夾執行（用 importlib 動態
載入它的 build_std_lookup / build_regression_df / holm_bonferroni 三個
函數，原因跟 11 一樣：檔名開頭是數字沒辦法用一般 import 語法）。

跑完一樣把整段輸出貼回去，一起判斷這個結果站不站得住腳。
"""

import importlib
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

_robustness = importlib.import_module("10_robustness_checks")
build_std_lookup = _robustness.build_std_lookup
build_regression_df = _robustness.build_regression_df
holm_bonferroni = _robustness.holm_bonferroni

warnings.filterwarnings("ignore")

FORMULA_SIZE = "degree ~ in_science_park + log_univ + log_station + log_junction + log_papers"


def _load_affil_csv():
    """跟 08/09/10/11 同一套防呆：55k 規模檔名是 affil_full.csv。"""
    for candidate in ("affil.csv", "affil_full.csv"):
        if os.path.exists(candidate):
            print(f"讀取 {candidate}")
            return pd.read_csv(candidate)
    raise FileNotFoundError("找不到 affil.csv 或 affil_full.csv。")


def fit_size_control_model(df, formula=FORMULA_SIZE):
    """跟 11 一模一樣的模型設定：正確用 MLE 估計 alpha 的負二項迴歸，
    不是固定 alpha=1 的 GLM 版本——要壓力測試的正是這個模型算出來的係數，
    所以測試時必須用同一個模型設定，換了設定就不是在測同一件事。"""
    return smf.negativebinomial(formula, data=df).fit(disp=0)


# ---------- 檢查 1：產能控制模型的距離門檻敏感度分析 ----------

def threshold_sensitivity_size_control(df_affil, df_park, df_combined, nodes, radii_m):
    print("=" * 70)
    print("[檢查 1] 產能控制模型的距離門檻敏感度分析")
    print("=" * 70)
    results = []
    for radius in radii_m:
        std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=radius)
        df = build_regression_df(nodes, std_lookup)
        n_park = df["in_science_park"].sum()
        if n_park < 5:
            print(f"門檻 {radius}m：園區內機構只有 {n_park} 筆，樣本太小跳過")
            continue
        try:
            model = fit_size_control_model(df)
        except Exception as e:
            print(f"門檻 {radius}m：模型無法收斂（{e}），跳過")
            continue
        coef = model.params["in_science_park"]
        p = model.pvalues["in_science_park"]
        irr = math.exp(coef)
        ci = model.conf_int().loc["in_science_park"]
        print(f"門檻 {radius}m：n_park={int(n_park)}, IRR={irr:.3f}, "
              f"95% CI=[{math.exp(ci[0]):.3f}, {math.exp(ci[1]):.3f}], p={p:.4g}"
              f"{'  ← 顯著' if p < 0.05 else ''}")
        results.append({"radius_m": radius, "n_park": int(n_park), "irr": irr, "p": p})
    return pd.DataFrame(results)


# ---------- 檢查 2：標籤排列檢定（每次重新配適完整迴歸，不是簡單平均數差） ----------

def null_model_regression_coefficient(df, formula=FORMULA_SIZE, n_perm=1000, seed=42):
    """
    做法：degree、log_univ、log_station、log_junction、log_papers 全部維持
    原樣不動，只把 in_science_park 這個標籤在所有機構間隨機重新分配（園區
    機構數維持跟觀察到的一樣多），每次重貼標籤後重新配適一次完整的負二項
    模型，記錄新的 in_science_park 係數。重複 n_perm 次，算「隨機重貼標籤
    時，係數絕對值 >= 觀察到的係數絕對值」的比例，就是排列檢定的 p 值。

    這個檢定每次都要重新配適一次完整模型（不是單純算平均數差），比 10 裡
    R4b 那個簡單排列檢定慢很多，預設 1000 次、實測數千筆規模的負二項 MLE
    大約幾分鐘內能跑完；如果你的機器跑起來太久，把 n_perm 調小（例如 300）
    先看大概的量級也可以，只是精確度會下降。
    """
    print("\n" + "=" * 70)
    print(f"[檢查 2] 產能控制模型的標籤排列檢定（{n_perm} 次，每次重新配適完整模型，"
          "可能要幾分鐘，請耐心等）")
    print("=" * 70)

    observed_model = fit_size_control_model(df)
    observed_coef = observed_model.params["in_science_park"]
    print(f"觀察到的 in_science_park 係數 = {observed_coef:.4f}"
          f"（IRR = {math.exp(observed_coef):.3f}，應該跟 11_institution_size_control.py"
          f" 印出來的數字一致，可以互相對照）")

    rng = np.random.default_rng(seed)
    n = len(df)
    n_park = int(df["in_science_park"].sum())
    base = df.drop(columns=["in_science_park"]).copy()

    perm_coefs = []
    n_failed = 0
    count_ge = 0

    for i in range(n_perm):
        perm_labels = np.zeros(n, dtype=int)
        park_idx = rng.choice(n, size=n_park, replace=False)
        perm_labels[park_idx] = 1
        df_perm = base.copy()
        df_perm["in_science_park"] = perm_labels
        try:
            m = smf.negativebinomial(formula, data=df_perm).fit(disp=0, maxiter=100)
            c = m.params["in_science_park"]
        except Exception:
            n_failed += 1
            continue
        perm_coefs.append(c)
        if abs(c) >= abs(observed_coef):
            count_ge += 1

        if (i + 1) % 100 == 0:
            print(f"  已完成 {i + 1}/{n_perm}（{n_failed} 次模型未收斂，已跳過不計入）")

    n_valid = len(perm_coefs)
    p_perm = count_ge / n_valid if n_valid else float("nan")
    perm_coefs = np.array(perm_coefs)
    print(f"\n排列檢定 p 值（{n_valid} 次有效排列，雙尾）= {p_perm:.4f}"
          f"（另有 {n_failed} 次模型未收斂，已排除、不計入分母）")
    print(f"隨機重貼標籤情況下，係數的平均值/標準差 = {perm_coefs.mean():.4f} / {perm_coefs.std():.4f}"
          f"（平均值應該接近 0，代表隨機貼標籤時模型抓不到系統性差異，是預期中的健全性檢查）")
    return p_perm, observed_model


if __name__ == "__main__":
    df_affil = _load_affil_csv()
    df_park = pd.read_csv("park_matches.csv")
    df_combined = pd.read_csv("combined.csv")
    nodes = json.load(open("std_nodes.json", encoding="utf-8"))

    std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
    df = build_regression_df(nodes, std_lookup)
    print(f"迴歸樣本：{len(df)} 個標準化機構（園區內 {int(df['in_science_park'].sum())}）\n")

    threshold_df = threshold_sensitivity_size_control(
        df_affil, df_park, df_combined, nodes, radii_m=[500, 1000, 2000, 3000, 5000]
    )

    p_perm, observed_model = null_model_regression_coefficient(df, n_perm=1000)

    # ---------- 檢查 3：把產能控制模型一起納入 Holm-Bonferroni 校正 ----------
    print("\n" + "=" * 70)
    print("[檢查 3] 把產能控制模型一起納入 Holm-Bonferroni 校正（4 項）")
    print("=" * 70)

    # 跟 10_robustness_checks.py 同一套邏輯：entity-resolved Mann-Whitney、
    # 不含產能控制的迴歸，都用這次資料當場動態算，不寫死；第三項才是這次
    # 真正要壓力測試的產能控制模型係數。
    park_deg_er = df.loc[df["in_science_park"] == 1, "degree"]
    nonpark_deg_er = df.loc[df["in_science_park"] == 0, "degree"]
    _, p_entity_resolved = stats.mannwhitneyu(park_deg_er, nonpark_deg_er, alternative="two-sided")

    base_model = smf.negativebinomial(
        "degree ~ in_science_park + log_univ + log_station + log_junction", data=df
    ).fit(disp=0)

    pvalue_dict = {
        "entity-resolved degree Mann-Whitney": p_entity_resolved,
        "NB regression, no productivity control": base_model.pvalues["in_science_park"],
        "NB regression, WITH productivity control": observed_model.pvalues["in_science_park"],
    }

    if os.path.exists("nodes.json"):
        naive_nodes = json.load(open("nodes.json", encoding="utf-8"))
        naive_park = [n["degree"] for n in naive_nodes if n.get("in_science_park")]
        naive_nonpark = [n["degree"] for n in naive_nodes if not n.get("in_science_park")]
        _, p_naive = stats.mannwhitneyu(naive_park, naive_nonpark, alternative="two-sided")
        pvalue_dict["naive degree Mann-Whitney"] = p_naive
    else:
        print("⚠ 找不到 nodes.json，naive 那項這次先跳過（不影響其他三項）。")

    holm_bonferroni(pvalue_dict)

    print("\n完成。把上面全部輸出貼回去，我幫你判斷這個邊緣顯著的結果站不站得住腳。")
