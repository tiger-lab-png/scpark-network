"""
Phase 2: Nominatim 地理編碼 + OSM 微觀空間特徵萃取（Hybrid A/B）
------------------------------------------------------------------
輸入：Phase 1 產出的 addresses_unique_for_geocoding.csv（去重後的機構地址）
輸出：enriched_geo_features.csv，每筆地址附上經緯度、是否落在既有科學園區
      多邊形內（方法 A）、若無多邊形則以密度指標估計創新熱區強度（方法 B）。

合規與穩健性設計重點：
1. Nominatim 使用政策明文規定：公開服務上限 1 request/sec、必須設定有意義的
   User-Agent、結果必須在本地端快取、禁止對同一查詢重複發送請求。本腳本用
   本地 JSON 快取檔案 + 1.1 秒節流落實這些要求；長期或大規模使用應改自架
   Nominatim 或改用 Photon/Geoapify 分流。
2. 方法 A 用 Overpass 的 `is_in` 查詢，而不是 `around`。`around` 只能找出
   「附近」的地物，不能保證該點真的落在多邊形「內部」；`is_in` 才是正確
   的點在多邊形內（point-in-polygon）查詢方式。
3. 方法 B（密度＋距離特徵）作為方法 A 查無多邊形時的 fallback，兩者合併
   即為 hybrid 邏輯，也是這篇論文方法論上的一個小貢獻點。
"""

import json
import math
import os
import re
import time

import pandas as pd
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# 2026-07 重新核對過 OSM Wiki「Public Overpass API instances」表格後更新：
# - api.openstreetmap.fr、overpass.osm.rambler.ru：現在 DNS 直接解析失敗
#   （不是忙碌，是網域已經不在官方清單上了），整組拿掉。
# - overpass.kumi.systems：官方已正式更名遷移到 overpass.private.coffee，
#   舊網域大量 429 很可能就是因為新東家沒在舊網域上分配資源，改用新網址。
#   Wiki 上明寫 private.coffee 這個實例 "no rate limit in place"。
# - 新增 VK Maps（maps.mail.ru）：Wiki 上明寫沒有請求限制。
# - overpass-api.de 官方主站留著當備援：政策是每天 <10,000 次查詢、<1GB
#   都算安全，我們總共只需要 3,249 個不同座標，遠低於這個門檻。
OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

# Nominatim 政策要求提供可辨識的 User-Agent/Referer，並留聯絡方式
HEADERS = {
    "User-Agent": "micro-geo-innovation-mapper/0.1 (dddsss5419@gmail.com)"
}

GEOCODE_CACHE_FILE = "geo_cache.json"  # 檔名故意縮短,避免深層路徑超過 Windows 260 字元限制
NOMINATIM_MIN_INTERVAL = 1.1  # 秒，略高於官方 1 req/sec 上限,留緩衝

# 用來判斷「科學園區/創新特區」的關鍵字（可依研究領域擴充，建議多語系）
PARK_KEYWORDS = [
    "science park", "technology park", "tech park", "innovation district",
    "innovation park", "research park", "high-tech zone", "hi-tech zone",
    "科學園區", "科技園區", "產業園區", "創新園區", "研究園區",
]


# ---------- 快取工具 ----------

def load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache, path):
    """
    無人看管長時間執行時，中途斷電/被強制關機是真實風險：如果直接寫
    原檔，寫到一半被打斷會留下截斷、壞掉的 JSON，下次 load_cache() 會
    直接噴例外，等於前面幾小時的快取全毀。改成先寫到暫存檔，寫完整個
    成功了才用 os.replace() 原子性地換掉正式檔案——os.replace 在同一個
    磁碟分割上是原子操作，不會有「寫一半」的中間狀態，最壞情況就是這次
    的暫存檔沒寫完、正式檔案還是上一次寫完的完整版本，不會壞掉。
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# ---------- Phase 2a: Nominatim geocoding ----------

# 用來辨認「這一段是機構名稱」的關鍵字，而不是系所/實驗室層級的細節。
# 找到「最後一個」符合的片段，從那裡開始保留到字串結尾——這樣能丟掉更前面
# 的系所/中心細節，但保留機構名稱本身（不會像單純「留最後 N 段」那樣，
# 連 University of XXX 這種真正決定地理位置的關鍵詞都一起被砍掉）。
INSTITUTION_KEYWORDS = re.compile(
    r'\b(university|institute|college|hospital|school|center|centre|'
    r'corp|inc|ltd|laboratory|lab|academy|foundation|clinic|company)\b',
    re.IGNORECASE,
)


def clean_address_variants(address):
    """
    從「完整原始地址」到「越來越精簡」產生一系列候選查詢字串。

    背景：原始的 raw_affiliation_strings 常常包含系所/實驗室/中心層級的
    細節，甚至夾雜 email、註腳編號這類跟地理位置完全無關的雜訊，例如
    "6Department of Biomedical Data Science, Stanford University, Stanford,
    California"——開頭那個 "6" 是文獻擷取時黏到單字前面的上標註腳編號。
    Nominatim 的自由文字搜尋對這種結構複雜的長字串常常直接查無結果，但
    同一個地址去掉系所細節、只留「機構, 城市, 州/國家」就找得到。
    這不是地址本身有問題，是查詢字串太雜訊，屬於可以救回來的資料，不該
    直接放棄。

    策略（由嚴格到寬鬆，依序嘗試，第一個查到結果就採用）：
      1. 原始字串。
      2. 去掉開頭黏著的註腳數字（如 "6Department..." -> "Department..."）。
      3. 去掉 email 地址、"Electronic address:" 這類雜訊片段。
      4. 用逗號切開，從「最後一個含機構關鍵字的片段」開始保留到結尾——
         優先保留機構名稱本身，這對微觀地理定位很重要（如果直接砍到只剩
         「City, Country」，座標會退化成城市中心點，跟你研究要的校園級
         精度不符）。
      5. 保底：上面找不到機構關鍵字，或還是查無結果，才退到最後 3 段、
         最後 2 段（City, State, Country 這種純地名層級）。
    """
    variants = [address]

    no_footnote = re.sub(r'^\d+(?=[A-Za-z])', '', address).strip()
    if no_footnote and no_footnote != address:
        variants.append(no_footnote)

    no_email = re.sub(r'Electronic address:\s*\S+', '', no_footnote, flags=re.IGNORECASE)
    no_email = re.sub(r'\S+@\S+', '', no_email).strip(' .;,')
    if no_email and no_email != no_footnote:
        variants.append(no_email)

    parts = [p.strip() for p in no_email.split(',') if p.strip()]

    inst_idx = None
    for i, p in enumerate(parts):
        if INSTITUTION_KEYWORDS.search(p):
            inst_idx = i  # 保留最後一個符合的（涵蓋機構名稱出現在較後面片段的情況）
    if inst_idx is not None and inst_idx > 0:
        variants.append(', '.join(parts[inst_idx:]))

    for n in (3, 2):
        if len(parts) > n:
            variants.append(', '.join(parts[-n:]))

    seen = set()
    unique_variants = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            unique_variants.append(v)
    return unique_variants


def _nominatim_call(query_text, max_retries=3):
    """
    對單一查詢字串打 Nominatim，回傳 (results, network_ok)。
    network_ok=False 代表重試後仍是網路層級的失敗（不是查無結果），
    呼叫端要分開處理，不能把這種情況跟「合法查無資料」混為一談。
    """
    params = {"q": query_text, "format": "jsonv2", "limit": 1}
    for attempt in range(max_retries):
        try:
            resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return resp.json(), True
        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s 指數退避
            print(f"[geocode 重試 {attempt + 1}/{max_retries}] {query_text[:60]}...: {e}")
            time.sleep(wait)
    return None, False


def geocode_address(address, cache, max_retries=3):
    """
    將原始地址字串轉成 (lat, lon)。先查本地快取，沒有才打 API，
    打完立刻寫回快取檔，避免程式中斷後重複消耗 rate limit。

    重要修正 1（網路例外 vs 合法查無資料）：DNS 解析失敗、連線中斷這類
    「暫時性網路問題」跟「Nominatim 真的查不到這個地址」是兩回事。網路
    例外重試 max_retries 次仍失敗，這筆地址「不寫入快取」，下次重跑會
    自動重新嘗試；只有 API 真的回應成功、但查無結果，才是合法的查無資料，
    才會寫入快取。

    重要修正 2（地址簡化 fallback）：原始地址字串查無結果，不代表這個
    機構真的地理編碼失敗——常常是字串包含系所/email/註腳編號這類雜訊，
    讓 Nominatim 的解析器卡住。改成依序嘗試 clean_address_variants() 產生
    的一系列簡化版本，任何一個版本查到結果就採用，全部版本都查無結果
    才算真正的合法查無資料。查詢字串永遠用「原始地址」當快取 key，
    不管實際上是哪個簡化版本查到的。
    """
    if address in cache:
        return cache[address]

    variants = clean_address_variants(address)

    for variant_idx, variant in enumerate(variants):
        results, network_ok = _nominatim_call(variant, max_retries=max_retries)

        if not network_ok:
            # 這是網路層級失敗，不是「這個簡化版本查無結果」，整筆地址先放棄，
            # 不寫入快取，下次重跑會從頭（含所有簡化版本）重新嘗試。
            print(f"[geocode 放棄，下次重跑會自動補] {address[:60]}...")
            time.sleep(NOMINATIM_MIN_INTERVAL)
            return [None, None]

        time.sleep(NOMINATIM_MIN_INTERVAL)  # 落實 1 req/sec 節流，每次 HTTP 呼叫都要等

        if results:
            lat, lon = float(results[0]["lat"]), float(results[0]["lon"])
            if variant_idx > 0:
                print(f"[geocode 簡化後找到] 原始地址查無結果，改用簡化版本"
                      f"「{variant[:60]}」找到座標")
            cache[address] = [lat, lon]
            save_cache(cache, GEOCODE_CACHE_FILE)
            return [lat, lon]

    # 所有簡化版本都查無結果，才算真正合法的查無資料，寫入快取避免重複查詢。
    cache[address] = [None, None]
    save_cache(cache, GEOCODE_CACHE_FILE)
    return [None, None]


def geocode_all(addresses, cache_path=GEOCODE_CACHE_FILE, failed_log_path="geo_failed.txt"):
    """
    可安全中斷、可續跑：cache 在每筆地址處理完就立刻寫回磁碟（見
    geocode_address），所以不管是程式自然跑完、當機、還是你自己按
    Ctrl+C 中斷，已經成功的部分都不會遺失。重新執行這支腳本時，
    這個 for 迴圈還是會把 10404 筆地址從頭跑一次，但已經在快取裡的
    地址只是查一次本地字典（幾乎不耗時），不會重打 Nominatim，
    所以「重跑」實際上等於「從中斷的地方繼續」，不用擔心要整批重來。

    另外加了：
    - Ctrl+C 中斷時印出目前進度，不會噴一堆 traceback 嚇人。
    - 每 50 筆印一次預估剩餘時間，讓你知道還要等多久。
    - 這次執行裡「重試 3 次仍失敗」的地址，另外寫進 geo_failed.txt，
      方便你事後直接看清單，不用在 console 裡往上滾。
    """
    cache = load_cache(cache_path)
    records = []
    start_time = time.time()
    failed_this_run = []
    consecutive_empty = 0  # 連續「合法查無結果」的計數，用來偵測疑似軟性節流

    try:
        for i, addr in enumerate(addresses):
            was_cached = addr in cache
            lat, lon = geocode_address(addr, cache)

            if not was_cached and lat is None and lon is None:
                # 這次執行才處理、而且失敗的（跟「原本快取裡就是合法查無結果」的情況分開記）
                failed_this_run.append(addr)
                consecutive_empty += 1
                if consecutive_empty >= 15:
                    print(f"\n⚠ 連續 {consecutive_empty} 筆地址都查無結果（含簡化版本都試過），"
                          f"不太可能是巧合，比較像是被暫時軟性節流。強制暫停 60 秒再繼續，"
                          f"如果之後還是連續失敗，建議先停下來手動確認幾筆地址。\n")
                    time.sleep(60)
                    consecutive_empty = 0
            elif not was_cached:
                consecutive_empty = 0  # 這次執行有成功查到，重置連續失敗計數

            records.append({"Raw_Affiliation": addr, "Latitude": lat, "Longitude": lon})

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed  # 筆/秒
                remaining = (len(addresses) - (i + 1)) / rate if rate > 0 else float("nan")
                print(f"已地理編碼 {i + 1}/{len(addresses)}　"
                      f"預估剩餘時間：約 {remaining / 60:.1f} 分鐘")

    except KeyboardInterrupt:
        print(f"\n手動中斷：已完成 {len(records)}/{len(addresses)} 筆。"
              f"已成功的部分都存在 {cache_path} 裡了，直接重跑這支腳本會從這裡繼續，"
              f"不用整批重來。")
        raise

    if failed_this_run:
        with open(failed_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed_this_run))
        print(f"\n這次執行有 {len(failed_this_run)} 筆重試 3 次後仍失敗，"
              f"清單存在 {failed_log_path}，這些地址沒有寫入快取，"
              f"下次重跑會自動再試一次。")

    return pd.DataFrame(records)


# ---------- Phase 2b/2c 共用：Overpass 查詢 + 重試 + 座標層級快取 ----------
#
# 這裡改動比較大，原因是實測遇到兩個問題：
# 1. 真正的 bug：classify_location() 原本用 `lat is None or lon is None` 判斷
#    有沒有座標，但 geocode_all() 回傳的 DataFrame 一旦欄位裡混了失敗的 None
#    跟成功的浮點數，pandas 會把整欄轉成 float64，None 會被靜默轉成
#    float('nan')——而 `float('nan') is None` 是 False，判斷失效，於是
#    「查無座標」的地址還是被送去問 Overpass，查詢字串裡直接嵌進去變成
#    `is_in(nan,nan)`，這是無效語法，Overpass 才會一直回 406。這是這次
#    log 裡看到「這麼多 (nan,nan) 406 錯誤」的直接原因，已修正成用
#    pd.isna() 判斷。
# 2. 就算是合法座標（像 42.35,-71.09 波士頓那種），也一樣出現 406 跟
#    connect timeout——這代表公開的 overpass-api.de 服務本身在長時間高頻
#    查詢下會限流/暫時性拒絕，不是你的程式邏輯錯。對策是：加重試退避、
#    降低頻率，並且用「座標」當 key 做本地快取——很多機構其實是同一個
#    校園、同一組座標（例如同一所大學不同系所），快取後同一個座標只會
#    真的打一次 Overpass，其餘直接讀本地結果，大幅減少實際請求量。

OVERPASS_CACHE_FILE = "overpass_cache.json"
OVERPASS_MIN_INTERVAL = 3.0  # 實測 kumi.systems 對高頻查詢比預期敏感，拉長間隔降低 429 機率


def _coord_key(lat, lon, precision=4):
    """座標四捨五入到小數 4 位（約 11 公尺）當快取 key，同校區的機構會命中同一筆。"""
    return f"{round(lat, precision)}_{round(lon, precision)}"


def overpass_query(query, max_node_failures=len(OVERPASS_ENDPOINTS), max_rate_limit_waits=2):
    """
    對 Overpass 送出查詢。實測發現兩種失敗性質完全不同，分開處理：

    - 429（Too Many Requests）：節點本身是活的，只是嫌你問太快。這種情況
      應該「等完 Retry-After 之後繼續打同一個節點」，不該馬上換節點——
      換到的節點如果是掛掉的，只是白白浪費 30 秒 timeout，完全沒有幫助。
      同一節點最多讓它 429 max_rate_limit_waits 次（現在有 4 個鏡像可以
      輪，沒必要在同一個過載節點上死等太久，改成最多等 2 輪就換節點，
      把時間分給還沒試過的鏡像）。
    - 其他錯誤（連線逾時、406 等）：代表這個節點本身有問題，換下一個節點，
      最多換 max_node_failures 次（預設等於鏡像站數量，每個都試過一輪）。

    回傳 elements 清單；所有節點都試過還是失敗才回傳 None（呼叫端要自行
    處理「沒查到」跟「查詢失敗」的差別，不要混為一談）。
    """
    for node_attempt in range(max_node_failures):
        endpoint = OVERPASS_ENDPOINTS[node_attempt % len(OVERPASS_ENDPOINTS)]

        for rate_limit_attempt in range(max_rate_limit_waits):
            try:
                resp = requests.post(endpoint, data={"data": query}, headers=HEADERS, timeout=30)
            except Exception as e:
                print(f"[overpass 節點錯誤 {node_attempt + 1}/{max_node_failures}，換節點] {endpoint}: {e}")
                break  # 換節點，不是這個節點的 429 問題

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else 15
                print(f"[overpass 429，{endpoint} 忙碌中，等 {wait} 秒後重試同一節點"
                      f"（{rate_limit_attempt + 1}/{max_rate_limit_waits}）]")
                time.sleep(wait)
                continue  # 同一節點再試一次，不換節點也不算進 node_attempt

            try:
                resp.raise_for_status()
                return resp.json().get("elements", [])
            except Exception as e:
                print(f"[overpass 節點錯誤 {node_attempt + 1}/{max_node_failures}，換節點] {endpoint}: {e}")
                break  # 換節點
        else:
            # 429 等滿 max_rate_limit_waits 次還沒放行，放棄這個節點換下一個
            print(f"[overpass {endpoint} 429 太多次，放棄這個節點]")

        time.sleep(2)  # 換節點前稍微停一下

    return None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 公尺
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def classify_location_combined(lat, lon, radius_m=2000):
    """
    把方法 A（is_in 園區多邊形）跟方法 B（around 密度特徵）合併成「一次」
    Overpass 查詢，而不是分開打兩次——這是把總請求量直接砍半的關鍵優化，
    在公開服務被限流成這樣的情況下差異很大。
    兩種查詢回傳的 element 彼此不會混淆：園區多邊形只看 name 有沒有園區
    關鍵字，密度特徵只看 amenity/railway/highway 這幾個 tag，同一批
    elements 分開判斷完全沒問題，不需要靠查詢語法本身區分來源。
    回傳 None 代表這次查詢失敗（重試後仍失敗），呼叫端不該當成「合法查無資料」。
    """
    query = f"""
    [out:json][timeout:30];
    is_in({lat},{lon})->.park;
    (way(pivot.park); relation(pivot.park);) -> .parkAreas;
    .parkAreas out tags;
    (
      node["amenity"~"university|research_institute"](around:{radius_m},{lat},{lon});
      way["amenity"~"university|research_institute"](around:{radius_m},{lat},{lon});
      node["railway"="station"](around:{radius_m * 2},{lat},{lon});
      node["highway"="motorway_junction"](around:{radius_m * 2},{lat},{lon});
    ) -> .nearby;
    .nearby out center;
    """
    elements = overpass_query(query)
    if elements is None:
        return None

    in_park, park_name = False, None
    univ_count = 0
    station_dists, junction_dists = [], []

    for el in elements:
        tags = el.get("tags", {})
        name = (tags.get("name") or "").lower()
        if not in_park and any(kw in name for kw in PARK_KEYWORDS):
            in_park, park_name = True, tags.get("name")

        elat = el.get("lat") or (el.get("center") or {}).get("lat")
        elon = el.get("lon") or (el.get("center") or {}).get("lon")
        if elat is not None and elon is not None:
            d = haversine(lat, lon, elat, elon)
            if tags.get("amenity") in ("university", "research_institute"):
                univ_count += 1
            if tags.get("railway") == "station":
                station_dists.append(d)
            if tags.get("highway") == "motorway_junction":
                junction_dists.append(d)

    return {
        "in_science_park": in_park,
        "park_name": park_name if in_park else None,
        "univ_research_count": univ_count,
        "nearest_station_m": min(station_dists) if station_dists else None,
        "nearest_junction_m": min(junction_dists) if junction_dists else None,
        "method_used": "A_polygon" if in_park else "B_density_fallback",
    }


# ---------- Hybrid：A 優先，查無多邊形才 fallback 到 B ----------

def classify_location(lat, lon, cache):
    """
    以座標為單位做 hybrid A/B 分類（合併成一次 Overpass 查詢），結果寫入
    cache（同座標下次直接讀取，不重打 Overpass）。lat/lon 用 pd.isna() 判斷，
    避免 NaN 被漏掉導致送出無效查詢（這是先前 406 洗版的根因）。
    """
    if pd.isna(lat) or pd.isna(lon):
        return {
            "in_science_park": None, "park_name": None,
            "univ_research_count": None, "nearest_station_m": None,
            "nearest_junction_m": None, "method_used": "no_coordinates",
        }

    key = _coord_key(lat, lon)
    if key in cache:
        return cache[key]

    result = classify_location_combined(lat, lon)
    time.sleep(OVERPASS_MIN_INTERVAL)

    if result is None:
        # Overpass 這次查詢重試後仍失敗，不寫入快取，下次重跑會自動再試，
        # 跟「查詢成功但沒有落在任何園區」的合法結果分開處理。
        return {
            "in_science_park": None, "park_name": None,
            "univ_research_count": None, "nearest_station_m": None,
            "nearest_junction_m": None, "method_used": "overpass_failed",
        }

    cache[key] = result
    save_cache(cache, OVERPASS_CACHE_FILE)
    return result


def enrich_addresses(df_geocoded, cache_path=OVERPASS_CACHE_FILE):
    """
    對每一筆已地理編碼的地址跑 hybrid A/B 分類，回傳合併後的 DataFrame。
    跟 geocode_all() 一樣：可安全 Ctrl+C 中斷、有 ETA 顯示、透過座標快取
    自動跳過已經處理過的座標（同校區的多個機構只會真的查一次 Overpass）。
    """
    cache = load_cache(cache_path)

    # 開跑前先算「實際需要查幾個不同座標」，不是地址筆數。很多機構其實是
    # 同一個校區、同一組座標（同一所大學不同系所），這個數字通常會比你想
    # 的小很多，讓你知道真正的工作量，不用被 10404 這個總筆數嚇到。
    valid_coords = df_geocoded.dropna(subset=["Latitude", "Longitude"])
    unique_keys = {_coord_key(r.Latitude, r.Longitude) for r in valid_coords.itertuples()}
    already_done = sum(1 for k in unique_keys if k in cache)
    print(f"共 {len(df_geocoded)} 筆地址，有效座標 {len(valid_coords)} 筆，"
          f"去重後實際需要查詢的不同座標數：{len(unique_keys)}"
          f"（已完成 {already_done}，剩下 {len(unique_keys) - already_done}）")

    feature_rows = []
    start_time = time.time()

    try:
        for i, row in df_geocoded.iterrows():
            feats = classify_location(row["Latitude"], row["Longitude"], cache)
            feature_rows.append(feats)

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(df_geocoded) - (i + 1)) / rate if rate > 0 else float("nan")
                print(f"已完成微觀空間特徵萃取 {i + 1}/{len(df_geocoded)}　"
                      f"預估剩餘時間：約 {remaining / 60:.1f} 分鐘")

    except KeyboardInterrupt:
        print(f"\n手動中斷：已完成 {len(feature_rows)}/{len(df_geocoded)} 筆。"
              f"座標快取存在 {cache_path}，重跑會自動跳過已經處理過的座標，"
              f"不用整批重來。")
        raise

    df_features = pd.DataFrame(feature_rows)
    return pd.concat([df_geocoded.reset_index(drop=True), df_features], axis=1)


if __name__ == "__main__":
    # 讀取 Phase 1 產出的去重地址清單（檔名已縮短為 addr_uniq.csv，
    # 理由見 phase1_openalex_fetch.py 開頭註解：Windows 深層路徑 + 長檔名
    # 容易超過 260 字元導致 FileNotFoundError）
    addr_df = pd.read_csv("addr_uniq.csv")
    addresses = addr_df["Raw_Affiliation"].dropna().tolist()

    print(f"共 {len(addresses)} 筆地址待處理（Nominatim 1 req/sec,請耐心等待）...")
    df_geocoded = geocode_all(addresses)
    df_geocoded.to_csv("geocoded.csv", index=False, encoding="utf-8-sig")

    df_enriched = enrich_addresses(df_geocoded)
    df_enriched.to_csv("enriched.csv", index=False, encoding="utf-8-sig")

    hit_a = (df_enriched["method_used"] == "A_polygon").sum()
    hit_b = (df_enriched["method_used"] == "B_density_fallback").sum()
    no_coord = (df_enriched["method_used"] == "no_coordinates").sum()
    overpass_failed = (df_enriched["method_used"] == "overpass_failed").sum()
    print(f"方法 A 命中（既有園區多邊形）：{hit_a} 筆")
    print(f"方法 B fallback（密度指標估計）：{hit_b} 筆")
    print(f"沒有座標，無法分類（geocoding 失敗）：{no_coord} 筆")
    print(f"有座標但 Overpass 查詢失敗（重跑會自動補）：{overpass_failed} 筆")
