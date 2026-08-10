"""
Phase 1 FULL 版本：抓取 T10361 的完整母體(不設 5,000 篇上限)

跟原本 phase1_openalex_fetch.py 唯一的差別：max_records 從 5000 改成
60000(略高於已知的 55,125 篇，留一點緩衝，OpenAlex 資料量隨時間會微幅
增加)。其餘抓取邏輯完全不變。

輸出檔名刻意跟原本的 affil.csv / addr_uniq.csv 不同（改成 _full 後綴），
避免覆蓋掉你現有已經跑完、已驗證過的 5,000 篇資料——如果全量版跑到一半
出問題，你手上隨時還有原本那份完整能用的資料可以回退，不會兩頭空。

建議：另外開一個新資料夾跑這個版本(例如 full_run/)，跟現有的
repo_build/scpark-network/data/ 完全分開，Phase 2/3 都在新資料夾裡跑，
兩邊互不干擾。

跑法（Windows PowerShell 或 cmd，在新資料夾裡）：
    python phase1_openalex_fetch_FULL.py
預期：因為只是抓取 metadata(不含地理編碼)，OpenAlex 這一步本身很快，
per-page=100、每次請求間隔0.2秒，抓完 55,000+ 篇大約 10-20 分鐘。
真正耗時的是接下來的 Phase 2 geocoding(Nominatim 1 req/sec 限制)，
預估要連續跑數天，請耐心，且可以隨時 Ctrl+C 中斷、之後重跑會自動接續
（見 02_geocode_and_enrich.py 的快取機制）。
"""

import requests
import pandas as pd
import time

MAILTO = "dddsss5419@gmail.com"
BASE_URL = "https://api.openalex.org"
TOPIC_ID = "https://openalex.org/T10361"


def fetch_works_by_topic(topic_id, year_from=2018, year_to=2024, max_records=60000):
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
    works_data = fetch_works_by_topic(TOPIC_ID, year_from=2018, year_to=2024, max_records=60000)
    df_affiliations = extract_affiliations(works_data)

    df_affiliations.to_csv("affil_full.csv", index=False, encoding="utf-8-sig")

    df_unique_addr = build_unique_address_list(df_affiliations)
    df_unique_addr.to_csv("addr_uniq_full.csv", index=False, encoding="utf-8-sig")

    print(f"共 {len(df_affiliations)} 筆作者-機構紀錄, "
          f"去重後 {len(df_unique_addr)} 筆唯一地址待地理編碼。")
    print(f"論文篇數：{df_affiliations['Paper_ID'].nunique()} 篇（預期接近 55,125）。")
