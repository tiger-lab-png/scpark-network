"""
Phase 1: OpenAlex 資料萃取模組
------------------------------------------------
用途：抓取指定新興科技領域的論文,萃取作者、機構與「原始地址字串」,
      作為後續 Nominatim 地理編碼與 OSM 微觀空間分析的輸入。

重要修正（相對於舊版草稿）：
1. OpenAlex 已將 Concepts 標記為 deprecated,官方建議改用 Topics
   (https://docs.openalex.org/api-entities/topics)。本腳本改用
   Topics 搜尋 + topics.id 過濾,而非 concepts.id。
2. authorships 底下的地址欄位已改為清單 `raw_affiliation_strings`
   (一位作者可能同時掛多個機構),舊欄位 `raw_affiliation_string`
   (單數) 仍存在但僅供回溯相容,腳本兩者都抓,避免漏資料。
3. 改用 cursor pagination（OpenAlex 官方建議的大量資料抓取方式）,
   避免傳統 page/per-page 在深分頁時失敗或重複的問題。
4. 加入去重後的機構地址清單輸出,供 Phase 2 Nominatim 使用
   （Nominatim 有嚴格的 1 req/sec 限制且要求結果快取,見腳本末尾說明）。

使用前置作業：
- 先用 find_topic_id() 或直接到 https://api.openalex.org/topics?search=xxx
  查詢你要鎖定的技術節點（例如 "advanced packaging"、"EUV lithography"、
  "generative AI drug discovery"）對應的 Topic ID。
- 把自己的 email 填入 MAILTO,可進入 OpenAlex 的 polite pool,
  取得更穩定、更快的回應。
"""

import requests
import pandas as pd
import time

MAILTO = "dddsss5419@gmail.com"  # 建議填入真實 email 以進入 polite pool
BASE_URL = "https://api.openalex.org"


def find_topic_id(keyword, top_n=5):
    """依關鍵字搜尋 OpenAlex Topics,回傳候選清單供人工確認 ID。"""
    resp = requests.get(
        f"{BASE_URL}/topics",
        params={"search": keyword, "per-page": top_n, "mailto": MAILTO},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    for t in results:
        print(f"{t['id']}  |  {t['display_name']}  |  works_count={t.get('works_count')}")
    return results


def resolve_topic_id(keyword, top_n=5):
    """
    自動用關鍵字搜尋 Topic,選第一個(OpenAlex 判定最相關的)結果直接使用,
    不用你手動複製 ID 貼進來。同時把候選清單印出來,方便你核對選得對不對；
    如果自動選的不是你要的領域,把印出來的候選 ID 複製貼到 TOPIC_ID 覆蓋即可。
    """
    candidates = find_topic_id(keyword, top_n=top_n)
    if not candidates:
        raise ValueError(f"找不到符合關鍵字「{keyword}」的 Topic，換個關鍵字再試一次。")

    chosen = candidates[0]
    print(f"\n>>> 自動選用 Topic：{chosen['display_name']}  ({chosen['id']})")
    print(">>> 如果這不是你要的領域，從上面候選清單挑一個，把 ID 貼到下面 TOPIC_ID 手動覆蓋。\n")
    return chosen["id"]


def fetch_works_by_topic(topic_id, year_from=2018, year_to=2024, max_records=5000):
    """
    用 cursor pagination 抓取指定 Topic 底下的論文,直到達到 max_records
    或資料抓完為止。cursor pagination 是 OpenAlex 官方建議的深分頁作法,
    比傳統 page 參數更穩定,且不受 10,000 筆 offset 上限影響。
    """
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
            break
        time.sleep(0.2)  # OpenAlex 對 polite pool 相對寬鬆,但仍保留節流

    return works[:max_records]


def extract_affiliations(works):
    """
    展開 works -> authorships -> raw affiliation strings。
    同時保留 raw 與 standardized 兩種版本,方便日後比對地理編碼品質。
    """
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

            # 新版欄位（清單）：一位作者可能有多筆原始地址
            raw_list = authorship.get("raw_affiliation_strings") or []
            # 舊版欄位（單一字串）,做為 fallback
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
    """
    輸出去重後的地址清單,這份清單才是要餵給 Phase 2 Nominatim 的東西。
    務必先去重再打 geocoding API—Nominatim 政策明文要求快取結果、
    禁止對同一查詢重複發送請求,且公開服務僅允許 1 request/sec。
    """
    unique_addresses = (
        df.dropna(subset=["Raw_Affiliation"])["Raw_Affiliation"]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return unique_addresses.to_frame(name="Raw_Affiliation")


if __name__ == "__main__":
    # 根據 keyword_scout.py 的實測結果直接鎖定 Topic ID，不再走自動搜尋，
    # 因為這幾個候選已經用「aboutness 分類信心」+「近3年/前3年成長倍率」
    # 量化比較過，直接寫死最有根據。

    # 主選：Silicon Carbide Semiconductor Technologies（第三代半導體）
    #   分類信心 0.886、成長倍率 1.08x——成長不算爆發但確實在成長，
    #   而且「第三代半導體」跟你整套工具的賣點（科學園區微觀地理群聚,
    #   如新竹、Silicon Valley）故事最搭，半導體業本來就高度依賴實體
    #   聚落，這是你原始構想裡最核心的案例。
    TOPIC_ID = "https://openalex.org/T10361"

    # 備選：如果你想走 AI 醫療（GenAI in Drug Discovery）路線，這個候選
    # 成長倍率更高、信心也夠，資料上更能撐「近年爆發式成長」的說法，
    # 只是跟「科學園區」的地理群聚敘事沒有半導體那麼直覺，藥廠/實驗室的
    # 群聚模式比較分散：
    # TOPIC_ID = "https://openalex.org/T10211"  # Computational Drug Discovery Methods

    # 注意：跑出來的 Additive Manufacturing/3D Printing（2.43x）和 Genetics,
    # Bioinformatics, and Biomedical Research（1.35x）雖然成長倍率最高，
    # 但前者是「aboutness 分類器」把 chiplet/heterogeneous integration 誤判連結
    # 到 3D 列印技術（語意上沾到邊但不是同一件事），後者範圍太廣泛、太成熟，
    # 兩個都跟你的研究故事對不起來，已經排除不用。

    works_data = fetch_works_by_topic(TOPIC_ID, year_from=2018, year_to=2024, max_records=5000)
    df_affiliations = extract_affiliations(works_data)
    # 檔名刻意縮短：如果你的 outputs 資料夾路徑本身很深（Windows 常見於各種
    # sandbox/雲端同步資料夾），路徑+檔名超過 260 字元會直接 FileNotFoundError，
    # 這不是程式邏輯錯誤，是 Windows MAX_PATH 限制。檔名越短越保險。
    df_affiliations.to_csv("affil.csv", index=False, encoding="utf-8-sig")

    df_unique_addr = build_unique_address_list(df_affiliations)
    df_unique_addr.to_csv("addr_uniq.csv", index=False, encoding="utf-8-sig")

    print(f"共 {len(df_affiliations)} 筆作者-機構紀錄,"
          f"去重後 {len(df_unique_addr)} 筆唯一地址待地理編碼。")
