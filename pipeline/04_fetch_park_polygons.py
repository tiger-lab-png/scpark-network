"""
針對 parks_wikidata.csv 裡「有連結 OSM relation ID」的園區（目前 10 個，
包括新竹科學園區 relation 4125319），各打一次很小、很明確的 Overpass
查詢，抓這個已知 relation 的實際邊界多邊形，取代「距離某個參考點多近」
的門檻判定，改成真正的 point-in-polygon 判斷。

這跟之前卡住的自架 Overpass / 公開鏡像站限流是完全不同量級的工作：只查
10 個已知、範圍很小的特定 relation，不是對全世界資料做 is_in()，公開
鏡像站應該扛得住，不需要等自架的那個跑完。

輸出：park_polygons.json，key 是 wikidata_id，value 是一個 **list of rings**
（MultiPolygon，不是單一多邊形）——每個 ring 是 [[lon,lat], ...] 座標環。

改版原因：新竹科學園區這種案例，relation 底下的 outer way 銜接不起來，
不是資料壞掉，是因為它本來就是「新竹本部 + 竹南 + 銅鑼」等好幾塊不相連
的園區組成、只是行政上算同一個 relation。第一版把所有銜接不起來的碎片
硬湊成一個 convex hull，結果凸包把分散基地「中間的空白地帶」也包進去，
導致像苗栗的大學被誤判成「在園區裡」（實測發現的真實 bug）。改版邏輯：
盡量把 outer way 個別銜接封閉成獨立的 ring，銜接不起來的單一 way 各自
獨立成一個 ring，不再把不相干的碎片湊在一起算凸包——point-in-polygon
判斷時只要落在「任一個」ring 裡就算 True，正確模擬 MultiPolygon 的語意。
inner（洞）不處理，科學園區形狀通常不是甜甜圈狀，這個簡化影響很小。
"""

import json
import time

import requests

# 2026-07-29 更新：overpass.kumi.systems 已正式遷移到 overpass.private.coffee
# （見 02_geocode_and_enrich.py 的說明），舊網域留著只會白白浪費一次重試。
# 換成跟 02 一致、目前確認還在運作的節點池（只查 10 個左右的已知 relation，
# 量體很小，這裡不需要像 02 那樣做冷卻/優先順序機制，單純多幾個候選輪流試）。
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

HEADERS = {
    "User-Agent": "micro-geo-innovation-mapper/0.1 (dddsss5419@gmail.com)"
}


def fetch_relation_geometry(relation_id, max_retries=3):
    """對單一 relation 打 Overpass 查詢，回傳 out geom 的原始 JSON。"""
    query = f"[out:json][timeout:60];relation({relation_id});out geom;"

    for attempt in range(max_retries):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            resp = requests.post(endpoint, data={"data": query}, headers=HEADERS, timeout=70)
            if resp.status_code == 200:
                return resp.json()
            print(f"  relation {relation_id}: HTTP {resp.status_code}（{endpoint}），重試...")
        except Exception as e:
            print(f"  relation {relation_id}: {e}（{endpoint}），重試...")
        time.sleep(5)
    print(f"  relation {relation_id}: 重試多次仍失敗，放棄，之後會 fallback 用距離門檻。")
    return None


def _chain_one_ring(remaining):
    """
    從 remaining（list of ways）裡挑一條開始，盡量端點銜接延伸，直到閉合
    或再也接不上為止。回傳 (ring_or_None, still_remaining)。

    重要：銜接不起來就回傳 None（丟棄這段，不強制閉合）。強制把頭尾硬接
    起來會湊出一個「看起來像多邊形、但形狀不保證正確」的東西——之前這樣
    做，結果把交大所在的區域錯誤地切在外面，比原本距離門檻的判定還糟。
    資料不完整（way 缺一段）時，寧可承認「這塊沒有可靠資料」，也不要用
    一個可能是錯的形狀去下判斷。
    """
    chain = remaining.pop(0)
    changed = True
    while remaining and changed:
        changed = False
        for i, seg in enumerate(remaining):
            if chain[-1] == seg[0]:
                chain = chain + seg[1:]
                remaining.pop(i)
                changed = True
                break
            if chain[-1] == seg[-1]:
                chain = chain + seg[::-1][1:]
                remaining.pop(i)
                changed = True
                break
            if chain[0] == seg[-1]:
                chain = seg[:-1] + chain
                remaining.pop(i)
                changed = True
                break
            if chain[0] == seg[0]:
                chain = seg[::-1][:-1] + chain
                remaining.pop(i)
                changed = True
                break
        if chain[0] == chain[-1]:
            break

    if chain[0] != chain[-1]:
        return None, remaining  # 銜接不起來，丟棄，不強制閉合

    return chain, remaining


def assemble_polygon(relation_data):
    """
    把 Overpass out geom 回傳的 relation members 組成一個(或多個)獨立的
    封閉環，回傳 list of rings（MultiPolygon 語意，不是單一多邊形）。
    只用 outer/空字串角色的 way，忽略 inner（洞）。只保留真正銜接成功、
    首尾相接的環；銜接不起來的碎片直接丟棄（不強制閉合、不算凸包），
    這個園區可用的精確資料就會相應減少，缺的部分回頭由呼叫端 fallback
    用距離門檻判定，比硬猜一個形狀可能錯的答案安全。
    """
    elements = relation_data.get("elements", [])
    rel = next((e for e in elements if e["type"] == "relation"), None)
    if rel is None:
        return None

    outer_ways = []
    for member in rel.get("members", []):
        # type=multipolygon 關聯用 outer/inner 角色；很多 type=boundary
        # 關聯（新竹科學園區這種常見）用空字串角色標記主環，不是 outer。
        # 兩種都收，只排除明確標記 inner（洞）的 way。
        role = member.get("role", "")
        if member.get("type") == "way" and role in ("outer", "") and "geometry" in member:
            coords = [(pt["lon"], pt["lat"]) for pt in member["geometry"]]
            if len(coords) >= 2:
                outer_ways.append(coords)

    if not outer_ways:
        return None

    rings = []
    dropped_fragments = 0
    remaining = outer_ways[:]
    while remaining:
        before = len(remaining)
        ring, remaining = _chain_one_ring(remaining)
        if ring is not None and len(ring) >= 4:  # 至少 3 個不同點 + 閉合點
            rings.append(ring)
        else:
            dropped_fragments += before - len(remaining) or 1

    if dropped_fragments:
        print(f"  ⓘ {dropped_fragments} 段 way 銜接不起來，已丟棄（不影響其他"
              "有效銜接成功的部分）")

    if not rings:
        return None

    n_pieces = len(rings)
    if n_pieces > 1:
        print(f"  ⓘ 這個園區的邊界由 {n_pieces} 塊不相連的區域組成"
              "（例如分散的多個園區基地），各自獨立判斷，不會硬湊成一個凸包。")

    return rings


if __name__ == "__main__":
    import pandas as pd

    df_parks = pd.read_csv("parks_wikidata.csv")
    has_osm = df_parks.dropna(subset=["osm_relation_id"])
    print(f"共 {len(has_osm)} 個園區有連結 OSM relation ID，開始逐一抓取邊界...")

    polygons = {}
    for row in has_osm.itertuples():
        rel_id = int(row.osm_relation_id)
        print(f"抓取 {row.name}（relation {rel_id}）...")
        data = fetch_relation_geometry(rel_id)
        if data is None:
            continue
        rings = assemble_polygon(data)
        if rings is None:
            print(f"  {row.wikidata_id}: 抓不到有效的 outer way 幾何，跳過")
            continue
        polygons[row.wikidata_id] = rings
        total_pts = sum(len(r) for r in rings)
        print(f"  成功，{len(rings)} 塊區域，共 {total_pts} 個頂點")
        time.sleep(3)  # 只有 10 個左右的請求，禮貌性節流即可

    with open("park_polygons.json", "w", encoding="utf-8") as f:
        json.dump(polygons, f, ensure_ascii=False)

    print(f"\n完成，成功取得 {len(polygons)}/{len(has_osm)} 個園區的精確邊界多邊形")
    print("輸出：park_polygons.json")
