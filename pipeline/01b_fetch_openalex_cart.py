"""
CAR-T 複製研究擴充至全量規模：重新抓取 T11491（CAR-T cell therapy research）的完整母體
------------------------------------------------------------------
背景：期刊審稿模擬（5 位審稿人小組，Scientometrics）指出 CAR-T 複製研究目前
還停留在跟 SiC 案例當初一樣的 5,000 篇 cursor 抽樣上限——正是這次修正案為
SiC 徹底排除掉的那個抽樣缺陷（不隨機、可能有年份偏誤）。審稿意見的
Devil's Advocate 給出的 Critical 等級意見明確要求：要嘛把 CAR-T 也擴充到
全量規模、把當初對 SiC 做過的 5 項稽核（ROR 稽核、代表性檢查、
leave-one-park-out、hyper-authorship 排除、園區三角驗證）在 CAR-T 上補做一次，
才能讓論文裡「identical pipeline was re-applied unchanged」這句話成立；
否則就必須在全文明確把 CAR-T 降級為「初步探索」而非「複製研究」。

你已經決定做全量重跑（跟 SiC full_run 走一樣的路）。

跟現有 phase1_cart_fetch.py（5,000 篇上限版本）唯一的差別：max_records
從 5000 改成 100000（T11491 已知全量約 84,431 篇，留緩衝）。其餘抓取邏輯
完全不變，直接複製自 full_run/phase1_openalex_fetch_FULL.py 的模式，只換
TOPIC_ID。

*** 重要：請先備份現有的 cart_run 資料夾 ***
建議整個複製一份 cart_run 到 cart_run_5000_backup（或類似名稱）再開始，
這樣如果全量重跑中途出問題，你手上還有一份完整、已驗證過的 5,000 篇版本
可以退回，不會兩頭空——這正是當初 full_run 之於原本 5,000 筆資料夾的
同一套保險做法。

*** 重要：時間預估，請安排在可以讓電腦連續跑數天的時段 ***
Phase 1（這支腳本，純 API 抓取，不含地理編碼）：約 20-30 分鐘。
Phase 2（02_geocode_and_enrich.py，Nominatim 地理編碼，1 req/sec 限制）：
  這是真正耗時的部分。5,000 篇規模時是 36,892 筆唯一地址、跑了超過 10 小時；
  全量規模論文數增加約 17 倍，即使地址重複率隨規模上升而改善（大部分機構
  會重複出現），唯一地址數保守估計也會落在 10-18 萬筆之間，換算成連續
  執行時間，大約是 3-5 天，可能更長，取決於 Overpass/Nominatim 當時的
  伺服器負載與重試次數。這跟當初 SiC full_run 的 Phase 2 耗時是同一個
  量級的事情，請比照當初的方式安排：讓程式在背景持續跑（可以用工作排程器
  或單純讓電腦保持開機），中途可以隨時 Ctrl+C 中斷，02_geocode_and_enrich.py
  本身有快取機制，重跑會自動接續，不會從頭開始。

完整執行順序（在 cart_run 資料夾裡，建議直接覆蓋現有檔案，因為已經備份過）：
  1. python phase1_cart_fetch_FULL.py
  2. python 02_geocode_and_enrich.py       （最耗時，見上）
  3. python 05_match_parks_distance.py     （沿用同一份 parks_wikidata.csv）
  4. python 06_apply_polygon_refinement.py
  5. python 07_merge_method_a_b.py
  6. python 08_build_entity_resolved_network.py
  7. python 09_build_naive_network.py
  8. python 10_robustness_checks.py        （這裡需要 geocoded.csv 和 nodes.json，
                                             確認 02 和 09 都跑完後再執行）
  9. python 11_institution_size_control.py
  10. python 12_productivity_control_robustness.py

跑完後，把完整主控台輸出貼回來，我會比照這次 v18/v19 的做法，直接對照
你機器上的原始資料獨立重新驗證每一個數字，再把結果寫進論文——連同補做
當初 SiC 做過但 CAR-T 還沒做的那 5 項稽核（ROR 稽核、代表性檢查、
leave-one-park-out、hyper-authorship 排除、園區三角驗證），讓論文裡
「identical pipeline was re-applied unchanged, all nine robustness checks」
這句話真正站得住腳。
"""

import requests
import pandas as pd
import time

MAILTO = "dddsss5419@gmail.com"
BASE_URL = "https://api.openalex.org"
TOPIC_ID = "https://openalex.org/T11491"  # CAR-T cell therapy research


def fetch_works_by_topic(topic_id, year_from=2018, year_to=2024, max_records=100000):
    works = []
    cursor = "*"
    per_page = 100

    while len(works) < max_records:
        params = {
            "filter": f"topics.id:{topic_id},publication_year:{year_from}-{year_to}",
            "per-page": per_page,
            "cursor": cursor,
            "mailto": MAILTO,
        }
        resp = requests.get(f"{BASE_URL}/works", params=params, timeout=30)
        if resp.status_code != 200:
            print(f"API error {resp.status_code}: {resp.text[:200]}")
            break

        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break

        works.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        print(f"已抓取 {len(works)} 篇...")

        if not cursor:
            print("已抓到 cursor 盡頭（母體全部抓完），提前結束。")
            break
        time.sleep(0.2)

    return works[:max_records]


def extract_affiliations(works):
    rows = []
    for w in works:
        paper_id = w.get("id")
        title = w.get("title")
        year = w.get("publication_year")
        cited_by = w.get("cited_by_count")

        for authorship in w.get("authorships", []):
            author = authorship.get("author", {}) or {}
            author_id = author.get("id")
            author_name = author.get("display_name")

            raw_list = authorship.get("raw_affiliation_strings") or []
            raw_single = authorship.get("raw_affiliation_string")
            if raw_single and raw_single not in raw_list:
                raw_list = raw_list + [raw_single]

            institutions = authorship.get("institutions", []) or []
            inst_names = [i.get("display_name") for i in institutions if i]
            inst_countries = [i.get("country_code") for i in institutions if i]

            if not raw_list:
                raw_list = [None]

            for raw_aff in raw_list:
                rows.append({
                    "Paper_ID": paper_id,
                    "Title": title,
                    "Year": year,
                    "Cited_By_Count": cited_by,
                    "Author_ID": author_id,
                    "Author_Name": author_name,
                    "Raw_Affiliation": raw_aff,
                    "Standardized_Institutions": ", ".join(inst_names),
                    "Institution_Countries": ", ".join(filter(None, inst_countries)),
                })
    return pd.DataFrame(rows)


def build_unique_address_list(df):
    unique_addresses = (
        df.dropna(subset=["Raw_Affiliation"])["Raw_Affiliation"]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return unique_addresses.to_frame(name="Raw_Affiliation")


if __name__ == "__main__":
    works_data = fetch_works_by_topic(TOPIC_ID, year_from=2018, year_to=2024, max_records=100000)
    df_affiliations = extract_affiliations(works_data)

    # 檔名沿用 affil_full.csv / addr_uniq_full.csv，讓下游 02/05/06/07/08/09/10/11/12
    # 完全不用改路徑就能直接跑（跟 full_run 資料夾用的是同一套慣例）。
    # *** 執行前務必已經備份好現有的 5,000 篇版本，因為這一步會直接覆蓋 ***
    df_affiliations.to_csv("affil_full.csv", index=False, encoding="utf-8-sig")

    df_unique_addr = build_unique_address_list(df_affiliations)
    df_unique_addr.to_csv("addr_uniq_full.csv", index=False, encoding="utf-8-sig")

    print(f"共 {len(df_affiliations)} 筆作者-機構紀錄, "
          f"去重後 {len(df_unique_addr)} 筆唯一地址待地理編碼。")
    print(f"論文篇數：{df_affiliations['Paper_ID'].nunique()} 篇（預期接近 84,431，"
          f"實際數字以這次抓取當下 OpenAlex 資料庫狀態為準，記得記錄下來，"
          f"論文裡要註明是「population as retrieved」而非預先設定的目標值，"
          f"跟 SiC 全量版本的處理方式一致）。")
