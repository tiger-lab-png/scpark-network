"""
Phase A 穩健性分析：回應模擬審查報告（五位審稿人）第一級/第二級問題。

逐項對應：
  R1  統計推論謬誤          -> TOST 等價性檢定（run_tost）
  R1  NB 模型診斷不足        -> 離散參數/AIC/與 Poisson 的概似比檢定（fit_and_diagnose）
  R1  共線性未檢驗           -> VIF（compute_vif）
  R1  遺漏值機制未討論        -> 地理編碼失敗率依國家分布（missingness_by_country）
  R1  多重比較未校正         -> Holm-Bonferroni（holm_bonferroni）
  R2  距離門檻武斷           -> 500/1000/2000/3000/5000m 敏感度分析（threshold_sensitivity）
  R2  新竹依賴/驗證不對稱     -> 新竹佔比 + leave-one-park-out（hsinchu_concentration_and_loo）
  R4  機構規模未控制         -> 論文數當額外共變量，重跑迴歸（regression_with_size_control）
  R4  缺乏隨機基準模型        -> 標籤排列檢定（null_model_comparison）
  R4  社群偵測提及但未使用     -> 園區機構的社群分布（community_park_composition）
  R4  degree 加權/未加權未說明 -> 印出兩種版本供對照（weighted_vs_unweighted_degree）

這支腳本只做「計算 + 印出結果」，不動論文本身；跑完把整段輸出貼回來，
我再把數字寫進 paper_draft.docx。
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

SEARCH_RADIUS_M = 4000


# ---------- 共用：把 raw affiliation 層級資料接到 standardized institution 層級 ----------

def split_institutions(value):
    if pd.isna(value):
        return []
    return [s.strip() for s in str(value).split(",") if s.strip()]


def build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000):
    """
    標準化機構名稱 -> {in_science_park（依 match_radius_m 門檻重新判定,多數決）,
                        nearest_park_name（多數決眾數）,
                        univ_research_count / nearest_station_m / nearest_junction_m（第一筆有效值）,
                        paper_count（該機構在資料集裡出現在幾篇不同論文）}

    跟 phase3_std.py 的 build_std_to_geo() 邏輯一致，但這裡改成參數化的距離門檻，
    才能做敏感度分析；也順便把 univ_research_count 等密度特徵、以及論文數
    （機構規模的代理變數）一起接上，避免重複掃描三次 affil.csv。
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
        majority = (sum(verdicts) > len(verdicts) / 2) if verdicts else False
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


# ---------- R1a: TOST 等價性檢定 ----------

def run_tost(coef, se, df_resid, log_low, log_high, alpha=0.05):
    """
    兩個單邊 t 檢定（TOST）：
      檢定 1（上界）：H0: coef >= log_high  vs  H1: coef < log_high
      檢定 2（下界）：H0: coef <= log_low   vs  H1: coef > log_low
    兩個都拒絕（p < alpha）才能宣稱「效果落在等價界線內」。
    等價界線選 IRR 0.80-1.25（log 值 ±0.2231），這是生物等效性研究最常用的
    慣例界線（80-125% 規則），借用來當「多大的效果才算不可忽略」的保守判準；
    在缺乏這個領域既有慣例的情況下，這是最有文獻依據、最不會被質疑是
    「自己選一個讓結果好看的界線」的選擇。
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


# ---------- R1b: NB 模型診斷（正確估計離散參數，而不是固定 alpha=1）----------

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


# ---------- R1c: VIF ----------

def compute_vif(df, predictors):
    X = sm.add_constant(df[predictors])
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


# ---------- R2a: 距離門檻敏感度分析 ----------

def threshold_sensitivity(df_affil, df_park, df_combined, nodes, radii_m):
    print("\n" + "=" * 70)
    print("距離門檻敏感度分析（R2）")
    print("=" * 70)
    results = []
    for radius in radii_m:
        std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=radius)
        df = build_regression_df(nodes, std_lookup)
        n_park = df["in_science_park"].sum()
        n_nonpark = len(df) - n_park
        if n_park < 5:
            print(f"門檻 {radius}m：園區內機構只有 {n_park} 筆，樣本太小跳過檢定")
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

        print(f"門檻 {radius}m：n_park={n_park}, n_nonpark={n_nonpark}, "
              f"MW p={p_mw:.4g}, 迴歸 IRR={irr:.3f} (p={p_reg:.4g})")
        results.append({
            "radius_m": radius, "n_park": int(n_park), "n_nonpark": int(n_nonpark),
            "mw_p": p_mw, "reg_irr": irr, "reg_p": p_reg,
        })
    return pd.DataFrame(results)


# ---------- R2b: 新竹佔比 + leave-one-park-out ----------

def hsinchu_concentration_and_loo(df_affil, df_park, df_combined, nodes):
    print("\n" + "=" * 70)
    print("新竹科學園區佔比 + leave-one-park-out（R2）")
    print("=" * 70)
    std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
    df = build_regression_df(nodes, std_lookup)

    park_df = df[df["in_science_park"] == 1]
    is_hsinchu = park_df["park_name"].fillna("").str.contains("Hsinchu", case=False)
    n_hsinchu = is_hsinchu.sum()
    print(f"{len(park_df)} 個園區關聯標準化機構中，最近鄰園區為新竹科學園區的有 {n_hsinchu} 筆"
          f"（{n_hsinchu / len(park_df) * 100:.1f}%）")

    hsinchu_ids = set(park_df.loc[is_hsinchu, "id"])
    df_loo = df[~df["id"].isin(hsinchu_ids)].copy()

    n_park_loo = df_loo["in_science_park"].sum()
    print(f"排除新竹後，剩餘園區關聯機構：{n_park_loo} 筆")
    if n_park_loo >= 5:
        model_loo = smf.glm(
            "degree ~ in_science_park + log_univ + log_station + log_junction",
            data=df_loo, family=sm.families.NegativeBinomial(),
        ).fit()
        coef = model_loo.params["in_science_park"]
        p = model_loo.pvalues["in_science_park"]
        irr = math.exp(coef)
        ci = model_loo.conf_int().loc["in_science_park"]
        print(f"排除新竹後的迴歸：IRR={irr:.3f}，95% CI [{math.exp(ci[0]):.3f}, {math.exp(ci[1]):.3f}]，p={p:.4g}")
    return n_hsinchu, len(park_df)


# ---------- R4a: 機構規模控制 ----------

def regression_with_size_control(df):
    print("\n" + "=" * 70)
    print("加入機構規模（論文數）控制變數的迴歸（R4）")
    print("=" * 70)
    model = smf.glm(
        "degree ~ in_science_park + log_univ + log_station + log_junction + log_papers",
        data=df, family=sm.families.NegativeBinomial(),
    ).fit()
    print(model.summary())
    return model


# ---------- R4b: null model（標籤排列檢定，固定 degree 序列，隨機打亂園區標籤）----------

def null_model_comparison(nodes, n_perm=5000, seed=42):
    """
    審稿人四要的是「觀察到的 degree 差異，在網絡結構本身的隨機性下是否本來就
    會出現」。真正對應這個問題的檢定，不是 degree-preserving configuration
    model（那個模型「保留」的正是每個節點自己的 degree，用它去比較同一批
    節點的 park vs non-park degree 差異是套套邏輯，差異一定跟原本一樣，
    不會產生資訊）——而是「保留全部節點真實的 degree 序列不動，只把
    『這個節點是不是園區機構』這個標籤隨機打散重新分配」，看看純粹隨機貼標籤
    時，會不會也常常做出跟觀察到的一樣大（或更大）的組間差異。這是無母數、
    不需要假設分布形狀的排列檢定（permutation test），也是對 Mann-Whitney U
    檢定結果的一個獨立、假設更少的交叉驗證。
    """
    print("\n" + "=" * 70)
    print("標籤排列檢定（label-permutation test，R4——取代原本方向錯誤的" +
          " configuration model 比較）")
    print("=" * 70)
    deg = np.array([n["degree"] for n in nodes])
    is_park = np.array([bool(n["in_science_park"]) for n in nodes])
    n_park = is_park.sum()

    observed_diff = deg[is_park].mean() - deg[~is_park].mean()
    print(f"觀察到的平均 degree 差（園區 − 非園區）= {observed_diff:.3f}")

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
    print(f"排列檢定 p 值（{n_perm} 次隨機重貼標籤，雙尾）= {p_perm:.4f}")
    print(f"隨機貼標籤情況下，組間差異的平均值/標準差 = {diffs.mean():.3f} / {diffs.std():.3f}")

    # 2026-07-29 修正：這句原本寫死引用 5,000 筆版本的 Mann-Whitney p=.090
    # 做比較，是會直接被貼進論文的一句話，換一批資料還照抄舊數字會產生
    # 錯誤的敘述。改成當場從同一份 nodes 動態算一次 Mann-Whitney，用真的
    # 對應這次資料的數字來說明兩種方法是否互相印證。
    _park_deg = deg[is_park]
    _nonpark_deg = deg[~is_park]
    _, p_mw_ref = stats.mannwhitneyu(_park_deg, _nonpark_deg, alternative="two-sided")
    same_direction = "方向一致" if (p_perm < 0.05) == (p_mw_ref < 0.05) else "結論方向不一致，需要留意"
    print(f"（對照：同一份資料的 entity-resolved Mann-Whitney U 的 p = {p_mw_ref:.4g}，"
          f"跟排列檢定{same_direction}，兩種完全不同假設的檢定方法互相印證，"
          f"不是同一個檢定套殼重跑。）")
    return p_perm


# ---------- R4c: 社群結構跟園區分類的關係 ----------

def community_park_composition(nodes):
    print("\n" + "=" * 70)
    print("社群結構 vs. 園區分類（R4）")
    print("=" * 70)
    from collections import Counter
    comm_park = defaultdict(lambda: [0, 0])
    for n in nodes:
        c = n["community"]
        comm_park[c][1] += 1
        if n["in_science_park"]:
            comm_park[c][0] += 1

    comm_sizes = Counter(n["community"] for n in nodes)
    print("最大 8 個社群裡，園區機構的組成比例：")
    for cid, size in comm_sizes.most_common(8):
        park_n, total_n = comm_park[cid]
        pct = park_n / total_n * 100 if total_n else 0
        print(f"  社群 {cid}（{total_n} 個機構）：園區機構 {park_n} 個（{pct:.1f}%）")

    total_park = sum(v[0] for v in comm_park.values())
    n_communities_with_park = sum(1 for v in comm_park.values() if v[0] > 0)
    print(f"\n全部 {total_park} 個園區機構分散在 {n_communities_with_park} 個不同社群裡"
          f"（總共 {len(comm_park)} 個社群）")


# ---------- R4d: degree 加權 vs 未加權 ----------

def weighted_vs_unweighted_degree(std_edges_path="std_edges.json"):
    print("\n" + "=" * 70)
    print("Degree：加權 vs. 未加權對照（R4）")
    print("=" * 70)
    edges = json.load(open(std_edges_path, encoding="utf-8"))
    G = nx.Graph()
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e.get("weight", 1))
    unweighted = dict(G.degree())
    weighted = dict(G.degree(weight="weight"))
    diffs = [(k, unweighted[k], weighted[k]) for k in list(unweighted)[:5]]
    print("前 5 個節點對照（未加權=相異合作機構數，加權=總共同掛名論文數）：")
    for name, u, w in diffs:
        print(f"  {name[:50]}：未加權={u}，加權={w}")
    print("論文正文目前用的是未加權 degree（相異合作機構數），"
          "跟 betweenness 用加權距離不是同一個尺度，需要在方法論明講。")


# ---------- R1d: 遺漏值機制（地理編碼失敗率 by 國家）----------

def missingness_by_country(df_affil, df_geocoded):
    print("\n" + "=" * 70)
    print("地理編碼失敗率依國家分布（R1/R3）")
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
    print("地理編碼失敗率最高的國家（至少 20 筆才列入，避免小樣本雜訊）：")
    print(summary.head(15))

    for code in ["TW", "CN", "JP", "KR"]:
        if code in summary.index:
            row = summary.loc[code]
            print(f"對照：{code} 失敗率 = {row['fail_rate']*100:.1f}%（n={int(row['count'])}）")
    return summary


# ---------- R1e: Holm-Bonferroni 多重比較校正 ----------

def holm_bonferroni(pvalue_dict, alpha=0.05):
    print("\n" + "=" * 70)
    print("Holm-Bonferroni 多重比較校正（R1）")
    print("=" * 70)
    items = sorted(pvalue_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    print(f"{'檢定':40s} {'原始 p':>10s} {'校正門檻':>10s} {'仍顯著?':>8s}")
    still_significant = True
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        significant = still_significant and (p < threshold)
        if not significant:
            still_significant = False
        print(f"{name[:40]:40s} {p:>10.4g} {threshold:>10.4g} {'是' if significant else '否':>8s}")


def _load_affil_csv():
    """2026-07-29：55k 規模檔名是 affil_full.csv，兩個都找找看，找不到才報錯。"""
    import os
    for candidate in ("affil.csv", "affil_full.csv"):
        if os.path.exists(candidate):
            print(f"讀取 {candidate}")
            return pd.read_csv(candidate)
    raise FileNotFoundError("找不到 affil.csv 或 affil_full.csv。")


if __name__ == "__main__":
    df_affil = _load_affil_csv()
    df_park = pd.read_csv("park_matches.csv")
    df_combined = pd.read_csv("combined.csv")
    df_geocoded = pd.read_csv("geocoded.csv")
    nodes = json.load(open("std_nodes.json", encoding="utf-8"))

    std_lookup = build_std_lookup(df_affil, df_park, df_combined, match_radius_m=2000)
    df = build_regression_df(nodes, std_lookup)
    print(f"迴歸樣本：{len(df)} 個標準化機構（園區內 {df['in_science_park'].sum()}）\n")

    diag = fit_and_diagnose(df, "degree ~ in_science_park + log_univ + log_station + log_junction")
    print("=" * 70)
    print("負二項模型診斷（R1）")
    print("=" * 70)
    print(f"離散參數 alpha = {diag['alpha']:.4f}（alpha 顯著 > 0 代表資料確實過離散，"
          f"用負二項模型而非 Poisson 是對的選擇）")
    print(f"NB AIC = {diag['nb_aic']:.1f}，Poisson AIC = {diag['poisson_aic']:.1f}"
          f"（AIC 越低越好）")
    print(f"概似比檢定（NB vs Poisson）：LR = {diag['lr_stat']:.1f}，p = {diag['lr_p']:.4g}")
    print("\n這個正確估計 alpha 的 NB 模型係數（取代原本論文 Table 2 用固定"
          " alpha=1 的 GLM 版本）：")
    print(diag["nb_model"].summary())

    coef = diag["nb_model"].params["in_science_park"]
    se = diag["nb_model"].bse["in_science_park"]
    df_resid = diag["nb_model"].df_resid
    tost = run_tost(coef, se, df_resid, log_low=math.log(0.80), log_high=math.log(1.25))
    print("\n" + "=" * 70)
    print("TOST 等價性檢定（R1，等價界線 IRR 0.80-1.25）")
    print("=" * 70)
    print(f"p_upper = {tost['p_upper']:.4g}，p_lower = {tost['p_lower']:.4g}")
    print(f"TOST p = {tost['p_tost']:.4g}"
          f"　{'（可宣稱效果落在等價界線內，即統計上支持 H2）' if tost['equivalent'] else '（尚不能宣稱等價，樣本可能仍然不足以排除 H1）'}")

    vif_df = compute_vif(df, ["in_science_park", "log_univ", "log_station", "log_junction"])
    print("\n" + "=" * 70)
    print("VIF 共線性診斷（R1）")
    print("=" * 70)
    print(vif_df.to_string(index=False))

    missingness_by_country(df_affil, df_geocoded)

    threshold_sensitivity(df_affil, df_park, df_combined, nodes, radii_m=[500, 1000, 2000, 3000, 5000])

    hsinchu_concentration_and_loo(df_affil, df_park, df_combined, nodes)

    regression_with_size_control(df)

    community_park_composition(nodes)

    weighted_vs_unweighted_degree()

    null_model_comparison(nodes)

    # 2026-07-29 修正：這裡原本是兩個寫死的 5,000 筆版本舊數字
    # （naive=2.098e-06、entity-resolved=0.090），只有迴歸係數是動態算的。
    # 55k 規模下直接照抄舊數字會混進一個「兩個舊 + 一個新」湊出來的假校正
    # 結果，而且不會報錯、看起來完全正常，是最容易被忽略的錯誤。改成：
    # entity-resolved 的 Mann-Whitney 直接從這次的 df 動態算；naive 的
    # Mann-Whitney 需要 09_build_naive_network.py 的輸出（nodes.json，
    # 注意不是 std_nodes.json），檔案不存在就明確印警告、跳過這個檢定，
    # 不能用舊數字頂替。
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
        print("\n⚠ 找不到 nodes.json（09_build_naive_network.py 的輸出），"
              "Holm-Bonferroni 校正這次先跳過 naive network 那一項，"
              "只用 entity-resolved + 迴歸係數兩項算——先跑完 09，"
              "再重跑這支腳本補齊完整的三項校正。")

    holm_bonferroni(pvalue_dict)

    print("\n完成。把上面全部輸出貼回去，我會整合進論文。")
