"""
13_fetch_density_5000m.py — 半徑脫鉤檢定（radius-decoupling test）的資料抓取

背景（v21 審查、編輯決定信 Roadmap 第 1 項，Devil's Advocate Critical 1）：
Table 3 的園區 IRR 在 1,000m 以上四個門檻幾乎持平（1.15/1.11/1.13/1.17），
只有 p 值隨處理組樣本數移動；而主要規格（2,000m）的處理變數跟密度共變量
剛好定義在「同一個 2,000m 圓盤」上。兩位審稿人獨立提出同一個決定性檢定：
把密度共變量的半徑跟處理半徑脫鉤重跑——
  (a) 處理 2,000m × 密度 5,000m
  (b) 處理 5,000m × 密度 5,000m（對角線對照）
  （「處理 5,000m × 密度 2,000m」就是現有 Table 3 的 5,000m 列，不用重算。）
這需要每個機構座標「5,000m 半徑內的大學/研究機構數」——現有資料只存了
2,000m 的計數，所以要重新查 Overpass。這支腳本只做資料抓取，不做分析；
抓完把 density_5000m.csv 跟 density5000_provenance.json 交回即可。

跟 02_geocode_and_enrich.py 的一致性：
- 查的是同一個 OSM 特徵（amenity=university|research_institute 的 node+way）。
- 02 用寬鬆 regex 抓回來後再用「完全等於 university/research_institute」過濾，
  這裡直接用錨定 regex ^(university|research_institute)$，集合等價。
- 沿用同一份端點清單與冷卻/重試邏輯（簡化版），單一端點失敗自動輪替。

效率設計：
- 只查 std_nodes.json 裡 7,886 個機構去重後的 5,098 個座標。
- 一次請求批次 10 個座標（每座標一個 out count，回傳順序=語句順序），
  約 510 次請求；含間隔預估 1.5〜3.5 小時。可隨時 Ctrl+C，快取會自動接續。

審稿人要求的凍結資訊：腳本會把每一批的查詢時間與實際使用的端點寫進
density5000_provenance.json——投稿時 Methodology 引用的「Overpass 查詢日期
與 instance」直接從這個檔案取，不要用記憶。

跑法（在 full_run 資料夾裡）：
    python 13_fetch_density_5000m.py
"""

import json
import os
import time
import datetime

import requests

OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]
ENDPOINT_COOLDOWN_SECONDS = 300
MIN_INTERVAL = 6.0
BATCH_SIZE = 8
RADIUS_M = 5000

CACHE_FILE = "density5000_cache.json"
PROVENANCE_FILE = "density5000_provenance.json"
OUTPUT_FILE = "density_5000m.csv"
LOWERBOUND_FILE = "density2000_lowerbound.csv"  # 每座標已知的 2,000m 計數（5km 計數的物理下界）

_cooldown_until = {}


def pick_endpoint():
    now = time.time()
    healthy = [e for e in OVERPASS_ENDPOINTS if _cooldown_until.get(e, 0) <= now]
    pool = healthy if healthy else OVERPASS_ENDPOINTS
    return pool[0]


def mark_failed(ep):
    _cooldown_until[ep] = time.time() + ENDPOINT_COOLDOWN_SECONDS
    # 輪到清單尾端
    OVERPASS_ENDPOINTS.append(OVERPASS_ENDPOINTS.pop(OVERPASS_ENDPOINTS.index(ep)))


def build_batch_query(coords):
    parts = ["[out:json][timeout:120];"]
    for i, (lat, lon) in enumerate(coords):
        parts.append(
            f'node["amenity"~"^(university|research_institute)$"]'
            f"(around:{RADIUS_M},{lat},{lon})->.n{i};"
        )
        parts.append(
            f'way["amenity"~"^(university|research_institute)$"]'
            f"(around:{RADIUS_M},{lat},{lon})->.w{i};"
        )
        parts.append(f"(.n{i}; .w{i};)->.a{i};")
        parts.append(f".a{i} out count;")
    return "\n".join(parts)


HEADERS = {"User-Agent": "scpark-network-research/1.0 (academic research; dddsss5419@gmail.com)"}


def run_batch(coords, bounds, max_attempts=12):
    query = build_batch_query(coords)
    consecutive_429 = 0
    for attempt in range(max_attempts):
        ep = pick_endpoint()
        try:
            r = requests.post(ep, data={"data": query}, timeout=150, headers=HEADERS)
            if r.status_code == 429:
                consecutive_429 += 1
                wait = min(30 * (2 ** (consecutive_429 - 1)), 240)
                print(f"  [429 @ {ep}，等 {wait} 秒後換端點]")
                mark_failed(ep)  # 429 也輪替端點，不再硬撞同一個
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"  [{r.status_code} @ {ep}，換端點]")
                mark_failed(ep)
                continue
            elements = r.json().get("elements", [])
            counts = [el for el in elements if el.get("type") == "count"]
            if len(counts) != len(coords):
                print(f"  [回傳 {len(counts)} 個 count，預期 {len(coords)}，換端點重試]")
                mark_failed(ep)
                continue
            totals = [int(c.get("tags", {}).get("total", 0)) for c in counts]
            # 資料層驗證：5km 圓盤包含 2km 圓盤，計數不可能小於已知的 2km 計數。
            # 違反 = 這個端點的資料庫覆蓋不完整（例如只涵蓋單一區域的 instance），
            # 整批丟棄、端點進冷卻。這正是 overpass.osm.ch（僅瑞士）造成
            # 第一次抓取汙染的原因。
            bad = [i for i, ((lat, lon), t) in enumerate(zip(coords, totals))
                   if t < bounds.get(f"{lat},{lon}", 0)]
            if bad:
                print(f"  [資料驗證失敗：{len(bad)} 個座標的 5km 計數 < 已知 2km 計數，"
                      f"@ {ep} 資料庫疑似區域性覆蓋，換端點]")
                mark_failed(ep)
                continue
            return totals, ep
        except Exception as e:
            print(f"  [{type(e).__name__} @ {ep}，換端點] {str(e)[:100]}")
            mark_failed(ep)
    return None, None


def main():
    nodes = json.load(open("std_nodes.json", encoding="utf-8"))
    coords = sorted(set((round(n["lat"], 7), round(n["lon"], 7)) for n in nodes))
    print(f"{len(nodes)} 個機構，去重後 {len(coords)} 個座標")

    bounds = {}
    if os.path.exists(LOWERBOUND_FILE):
        import csv
        with open(LOWERBOUND_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bounds[row["key"]] = float(row["count2000"])
        print(f"已載入 {len(bounds)} 筆 2,000m 下界驗證資料")
    else:
        print(f"⚠ 找不到 {LOWERBOUND_FILE}，資料層驗證停用（不建議）")

    cache = {}
    if os.path.exists(CACHE_FILE):
        cache = json.load(open(CACHE_FILE, encoding="utf-8"))
        n0 = len(cache)
        cache = {k: v for k, v in cache.items() if v >= bounds.get(k, 0)}
        if len(cache) < n0:
            print(f"清除 {n0 - len(cache)} 筆違反 2km 下界的中毒快取")
        print(f"快取已有 {len(cache)} 筆，接續執行")

    provenance = []
    if os.path.exists(PROVENANCE_FILE):
        provenance = json.load(open(PROVENANCE_FILE, encoding="utf-8"))

    todo = [c for c in coords if f"{c[0]},{c[1]}" not in cache]
    print(f"待查 {len(todo)} 個座標，批次大小 {BATCH_SIZE}，"
          f"預估 {len(todo) / BATCH_SIZE * (MIN_INTERVAL + 12) / 60:.0f} 分鐘上下")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
        totals, ep = run_batch(batch, bounds)
        round_failures = 0
        while totals is None:
            round_failures += 1
            if round_failures > 6:
                print("  [連續 6 輪全端點失敗（約半小時），存檔停止；稍後重跑會自動接續]")
                break
            wait = 300
            print(f"  [這一批全部端點都失敗，等 {wait//60} 分鐘後重試（第 {round_failures}/6 輪），進度已存檔]")
            time.sleep(wait)
            _cooldown_until.clear()  # 冷卻全部重置，重新輪一遍
            totals, ep = run_batch(batch, bounds)
        if totals is None:
            break
        for (lat, lon), cnt in zip(batch, totals):
            cache[f"{lat},{lon}"] = cnt
        provenance.append({
            "utc_time": t0, "endpoint": ep,
            "n_coords": len(batch), "radius_m": RADIUS_M,
        })
        json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"))
        json.dump(provenance, open(PROVENANCE_FILE, "w", encoding="utf-8"), indent=1)
        done = len(cache)
        print(f"{done}/{len(coords)} 座標完成")
        time.sleep(MIN_INTERVAL)

    # 輸出（就算沒跑完也輸出目前進度，方便檢查）
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("lat,lon,univ_research_count_5000m\n")
        for c in coords:
            key = f"{c[0]},{c[1]}"
            if key in cache:
                f.write(f"{c[0]},{c[1]},{cache[key]}\n")
    print(f"已寫出 {OUTPUT_FILE}（{sum(1 for c in coords if f'{c[0]},{c[1]}' in cache)}"
          f"/{len(coords)} 筆）與 {PROVENANCE_FILE}")
    if all(f"{c[0]},{c[1]}" in cache for c in coords):
        print("全部完成。把 density_5000m.csv 跟 density5000_provenance.json 交回即可。")


if __name__ == "__main__":
    main()
