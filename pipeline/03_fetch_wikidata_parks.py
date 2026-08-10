"""
權威來源 Ground Truth：從 Wikidata 抓全球科學/技術園區清單
------------------------------------------------------------------
方法論定位：不完全依賴 OSM 志工標記的多邊形（全球品質不均——已開發國家
標得細，很多地方標得很粗糙，這是 OSM 天生的限制，不管用 is_in 還是
reverse geocoding 都繞不過去），改用 Wikidata 這個有審核機制的權威資料源
當 ground truth，OSM/Nominatim 只是輔助定位、補充 Wikidata 沒收錄的新興
園區。這樣論文處理的是「怎麼應對 OSM 資料品質不均」這個已知痛點，而不是
單純「用哪個 API 查詢」，審稿人會更買單。

實作：用 Wikidata Query Service（SPARQL）查詢所有「科學園區/技術園區」
類別（含子類別）且有座標資料的條目，一次抓完存成本地檔案，之後比對完全
不用再打任何外部 API，沒有 rate limit 問題。

已知風險：Wikidata 上「科學園區」這個概念存在好幾個容易搞混、可能互相
重複/合併過的 QID。已經用 wd_test.py 實測驗證：新竹科學園區（Q717461）
的 P31 直接指向 Q1976594，所以這個 QID 確定是對的，先收斂到它 +
Q1281153（technology park）兩個已驗證/明確的類別，用 wdt:P279* 往下抓
子類別涵蓋命名變體。跑完務必看腳本印出的自我檢查結果，沒通過代表類別
還要再擴充。

效能筆記（第一輪）：第一版查詢把 4 個候選 QID + P279* 遞迴 + SERVICE
wikibase:label（對 item 和 country 都查標籤）全部塞在同一個 SPARQL 查詢
裡，結果在 Wikidata 公用 endpoint 上直接卡死、連續逾時（實測基本連線跟
小查詢都只要 1.3 秒，問題不在網路，在查詢太重）。

效能筆記（第二輪）：拿掉 label service 後改善，但 504 Gateway Timeout
還是發生——問題出在 `wdt:P31/wdt:P279*` 這種「屬性路徑」跟其他 join
（P625 座標、OPTIONAL P402/P17）湊在同一個查詢裡，查詢引擎很難最佳化，
是 WDQS 公用服務常見的效能陷阱。改成三階段：(1) 先用小查詢把候選類別的
子類別展開成一份有限的 QID 清單（P279* 單獨跑，沒有其他 join，快）；
(2) 用 VALUES + 直接 `wdt:P31`（不是屬性路徑，是索引過的直接比對）去
抓機構條目，主查詢改用 POST 避免 QID 清單一長 URL 超長；(3) 最後
批次補標籤。
"""

import json
import os
import time

import pandas as pd
import requests

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

# Wikidata 政策要求識別性 User-Agent，否則容易被限流/拒絕
HEADERS = {
    "User-Agent": "micro-geo-innovation-mapper/0.1 (dddsss5419@gmail.com)",
    "Accept": "application/sparql-results+json",
}

# 已驗證/明確的「科學園區/技術園區」候選類別 QID，用 UNION 納入，
# 加上 wdt:P279* 往下抓子類別（涵蓋各種命名變體：science park、technology
# park、research park、technopark...）。Q1976594 已用新竹科學園區
# （Q717461）實測驗證是正確的 P31 目標。
CANDIDATE_CLASS_QIDS = [
    "Q1976594",  # science park（已驗證：新竹科學園區的直接 P31 類別）
    "Q1281153",  # technology park
]

SUBCLASS_QUERY = """
SELECT DISTINCT ?class WHERE {{
  VALUES ?seed {{ {seeds} }}
  ?class wdt:P279* ?seed .
}}
""".format(seeds=" ".join(f"wd:{q}" for q in CANDIDATE_CLASS_QIDS))


def sparql_request(query, timeout=60, max_retries=3, use_post=False):
    """
    共用的 SPARQL 請求函式。query 短的用 GET，query 長（VALUES 塞很多 QID
    時）改用 POST，避免 URL 長度限制。
    """
    for attempt in range(max_retries):
        try:
            if use_post:
                resp = requests.post(
                    WIKIDATA_SPARQL_URL,
                    data={"query": query, "format": "json"},
                    headers=HEADERS,
                    timeout=timeout,
                )
            else:
                resp = requests.get(
                    WIKIDATA_SPARQL_URL,
                    params={"query": query, "format": "json"},
                    headers=HEADERS,
                    timeout=timeout,
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[wikidata 重試 {attempt + 1}/{max_retries}] {e}")
            time.sleep(wait)
    raise RuntimeError("Wikidata SPARQL 查詢重試多次仍失敗。")


def fetch_subclasses():
    """
    第一階段：單獨展開 CANDIDATE_CLASS_QIDS 底下所有子類別，query 裡只有
    P279* 這一個 join，沒有摻雜 P31/P625/OPTIONAL，執行速度快很多。
    """
    data = sparql_request(SUBCLASS_QUERY, timeout=60)
    class_qids = [
        b["class"]["value"].rsplit("/", 1)[-1]
        for b in data["results"]["bindings"]
    ]
    class_qids = sorted(set(class_qids))
    print(f"展開子類別，共 {len(class_qids)} 個類別 QID")
    if len(class_qids) > 500:
        print("  ⚠ 類別數量偏多，可能表示種子 QID 掛在某個很大的階層下"
              "（例如被歸類到過於一般的上位概念），建議之後人工抽查幾個"
              "類別 QID 確認語意合理。")
    return class_qids


def fetch_instances_for_classes(class_qids, timeout=90):
    """
    第二階段：用 VALUES + 直接 wdt:P31（索引過的直接比對，不是屬性路徑）
    去抓所有屬於這些類別的機構條目。query 用 POST，避免 QID 清單一長
    URL 超過長度限制。
    """
    query = """
    SELECT DISTINCT ?item ?coord ?osmRelation ?country WHERE {{
      VALUES ?class {{ {classes} }}
      ?item wdt:P31 ?class .
      ?item wdt:P625 ?coord .
      OPTIONAL {{ ?item wdt:P402 ?osmRelation . }}
      OPTIONAL {{ ?item wdt:P17 ?country . }}
    }}
    """.format(classes=" ".join(f"wd:{q}" for q in class_qids))
    return sparql_request(query, timeout=timeout, use_post=True)


LABEL_CACHE_FILE = "wikidata_labels_cache.json"


def _load_label_cache():
    if os.path.exists(LABEL_CACHE_FILE):
        with open(LABEL_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_label_cache(cache):
    # 跟 02 腳本的地理編碼快取一樣，先寫暫存檔再原子性換檔，避免中途中斷
    # 留下寫一半的壞檔。
    tmp_path = LABEL_CACHE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, LABEL_CACHE_FILE)


def fetch_labels_batch(qids, batch_size=50, max_retries=3):
    """
    用 MediaWiki wbgetentities API 批次補標籤，取代 SPARQL 的
    SERVICE wikibase:label——同樣的資訊，但不用讓 query.wikidata.org
    去 join 一堆語言 fallback，快很多也穩很多。

    2026-07-29 補上重試 + 本地快取：原本這裡沒有重試、也沒有寫入快取，
    只要中間任何一批網路暫時性失敗就直接整支腳本噴例外，前面兩階段
    SPARQL 查詢的結果（可能花了不少時間）就白費了，還得整個重跑。改成
    跟地理編碼快取一樣的邏輯：已經查到的標籤立刻寫進 LABEL_CACHE_FILE，
    重跑只會補還沒查到的部分。
    """
    cache = _load_label_cache()
    qids = list(dict.fromkeys(qids))  # 去重且保留順序
    missing = [q for q in qids if q not in cache]

    if missing:
        print(f"標籤快取已有 {len(cache)} 筆，還需要查 {len(missing)} 筆...")
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            entities = None
            for attempt in range(max_retries):
                try:
                    resp = requests.get(
                        WIKIDATA_API_URL,
                        params={
                            "action": "wbgetentities",
                            "ids": "|".join(chunk),
                            "props": "labels",
                            # 優先英文，沒有的話退而求其次抓任何一種語言的
                            # 標籤，總比顯示裸 QID 好讀（例如 Q760517 只有
                            # 德文標籤的情況）
                            "languages": "en|de|fr|ja|zh",
                            "format": "json",
                        },
                        headers=HEADERS,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    entities = resp.json().get("entities", {})
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  [標籤查詢重試 {attempt + 1}/{max_retries}] {e}")
                    time.sleep(wait)

            if entities is None:
                raise RuntimeError(
                    f"補標籤重試 {max_retries} 次仍失敗（第 {i}~{i + len(chunk)} 筆）。"
                    f"已完成的 {len(cache)} 筆標籤都存在 {LABEL_CACHE_FILE} 裡，"
                    f"直接重跑這支腳本會跳過這些、只補還沒查到的部分，不用整批重來。"
                )

            for qid, ent in entities.items():
                ent_labels = ent.get("labels", {})
                label = None
                for lang in ("en", "de", "fr", "ja", "zh"):
                    if lang in ent_labels:
                        label = ent_labels[lang]["value"]
                        break
                cache[qid] = label or qid
            _save_label_cache(cache)
            print(f"  已補標籤 {min(i + batch_size, len(missing))}/{len(missing)}"
                  f"（快取累計 {len(cache)} 筆）")

    return {q: cache.get(q, q) for q in qids}


PRELABEL_CHECKPOINT_FILE = "parks_prelabel_checkpoint.csv"
PRELABEL_CHECKPOINT_META_FILE = "parks_prelabel_checkpoint.meta.json"


def _checkpoint_is_valid():
    """
    檢查點只有在跟目前的 CANDIDATE_CLASS_QIDS 完全一致時才能重用。這點
    很重要：如果之後 self_check() 沒過、你調整了 CANDIDATE_CLASS_QIDS
    再重跑，不能誤用舊 QID 清單查出來的檢查點，否則調整等於白調。
    """
    if not (os.path.exists(PRELABEL_CHECKPOINT_FILE)
            and os.path.exists(PRELABEL_CHECKPOINT_META_FILE)):
        return False
    with open(PRELABEL_CHECKPOINT_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("candidate_class_qids") == sorted(CANDIDATE_CLASS_QIDS)


def _save_checkpoint(df):
    df.to_csv(PRELABEL_CHECKPOINT_FILE, index=False, encoding="utf-8-sig")
    with open(PRELABEL_CHECKPOINT_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"candidate_class_qids": sorted(CANDIDATE_CLASS_QIDS)}, f)


def fetch_wikidata_parks():
    """
    三階段流程：(1) 展開子類別 (2) 用 VALUES + 直接 P31 抓機構條目
    (3) 批次補標籤。這只是一次性、量不大的查詢（全球科學園區類條目大概
    幾百到幾千筆），不像 Nominatim/Overpass 那樣需要對每一筆機構地址
    各打一次，完全沒有我們前面碰到的 rate limit / 節流問題。

    2026-07-29 補上檢查點：(1)(2) 兩階段查完的結果先落地存檔
    （PRELABEL_CHECKPOINT_FILE），(3) 補標籤失敗時重跑不用重打 SPARQL；
    (3) 本身也改成有重試、有本地快取（見 fetch_labels_batch）。
    """
    if _checkpoint_is_valid():
        print(f"發現與目前 CANDIDATE_CLASS_QIDS 相符的檢查點 "
              f"{PRELABEL_CHECKPOINT_FILE}，跳過 SPARQL 查詢、直接接續補標籤...")
        df = pd.read_csv(PRELABEL_CHECKPOINT_FILE)
    else:
        class_qids = fetch_subclasses()
        data = fetch_instances_for_classes(class_qids)

        rows = []
        for b in data["results"]["bindings"]:
            coord_str = b.get("coord", {}).get("value", "")
            # Wikidata 座標格式是 WKT: "Point(經度 緯度)"，注意順序是 lon 在前
            lat, lon = None, None
            if coord_str.startswith("Point("):
                try:
                    lon_str, lat_str = coord_str[6:-1].split(" ")
                    lat, lon = float(lat_str), float(lon_str)
                except ValueError:
                    item_id = b.get("item", {}).get("value", "?")
                    print(f"  ⚠ 座標格式無法解析，略過這筆: {item_id} -> {coord_str!r}")

            country_uri = b.get("country", {}).get("value")
            country_qid = country_uri.rsplit("/", 1)[-1] if country_uri else None

            rows.append({
                "wikidata_id": b["item"]["value"].rsplit("/", 1)[-1],
                "lat": lat,
                "lon": lon,
                "osm_relation_id": b.get("osmRelation", {}).get("value"),
                "country_qid": country_qid,
            })

        df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])
        df = df.drop_duplicates(subset=["wikidata_id"])
        _save_checkpoint(df)
        print(f"已存檢查點 {PRELABEL_CHECKPOINT_FILE}（{len(df)} 筆，尚未補標籤）")

    country_qids = set(df["country_qid"].dropna().unique())
    print(f"第一階段完成，共 {len(df)} 筆，開始補標籤...")
    all_qids = list(df["wikidata_id"]) + list(country_qids)
    labels = fetch_labels_batch(all_qids)

    df["name"] = df["wikidata_id"].map(labels)
    df["country"] = df["country_qid"].map(labels)
    df = df.drop(columns=["country_qid"])
    return df


def self_check(df):
    """
    已知案例自我驗證：新竹科學園區（Wikidata Q717461）如果沒出現在清單裡，
    代表上面的 CANDIDATE_CLASS_QIDS 需要調整，不能直接信任這份清單。

    回傳 bool，讓呼叫端決定要不要把結果當成可信的 parks_wikidata.csv
    寫出去——這份清單是後面 05/06/07 整條 H1/H2 分類邏輯的 ground truth，
    驗證沒過卻悄悄寫出一份看起來正常的 CSV，是比腳本直接報錯更危險的
    結果，所以不能只印警告字串就算了。
    """
    hit = df[df["wikidata_id"] == "Q717461"]
    if len(hit) > 0:
        print(f"✓ 自我驗證通過：新竹科學園區有抓到 -> {hit.iloc[0].to_dict()}")
        return True
    else:
        print("⚠ 自我驗證失敗：新竹科學園區（Q717461）不在清單裡，"
              "代表 CANDIDATE_CLASS_QIDS 需要調整，這份清單目前不能直接信任。"
              "建議去 https://www.wikidata.org/wiki/Q717461 手動確認它實際"
              "掛在哪個 P31 類別下，把正確的 QID 加進 CANDIDATE_CLASS_QIDS。")
        return False


if __name__ == "__main__":
    print("查詢 Wikidata 全球科學/技術園區清單...")
    df_parks = fetch_wikidata_parks()
    print(f"共取得 {len(df_parks)} 筆有座標的園區條目")

    has_osm = df_parks["osm_relation_id"].notna().sum()
    print(f"其中 {has_osm} 筆有連結 OSM relation ID（未來要做精確多邊形比對可以優先用這批）")

    if self_check(df_parks):
        df_parks.to_csv("parks_wikidata.csv", index=False, encoding="utf-8-sig")
        print("已存成 parks_wikidata.csv")
    else:
        df_parks.to_csv("parks_wikidata_UNVERIFIED.csv", index=False, encoding="utf-8-sig")
        print("⚠ 自我驗證沒過，這次「沒有」寫入 parks_wikidata.csv，避免 05/06/07"
              "誤用一份錯的 ground truth 而不自知。結果先存成"
              "parks_wikidata_UNVERIFIED.csv 給你檢查、調整 CANDIDATE_CLASS_QIDS，"
              "確認調整後的驗證有過，重跑這支腳本才會正式寫出 parks_wikidata.csv。")
        raise SystemExit(1)
