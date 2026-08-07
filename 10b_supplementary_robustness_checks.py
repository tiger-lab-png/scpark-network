"""
Re-retrieve the university/research-institute count within 2,000 m of each institution.

The 2,000 m counts already exist in enriched.csv, but they were collected on a
different date from the 5,000 m counts. OpenStreetMap coverage grows over time,
so the two radii must be measured against the same snapshot to be comparable;
this script fetches the 2,000 m counts again in the same session as
13_fetch_density_5000m.py.

Reads:  std_nodes.json (institution coordinates) and density_5000m.csv, whose
        counts are the upper bound used for validation.
Writes: density_2000m_v2.csv, density2000v2_provenance.json (UTC timestamp,
        endpoint, batch size and radius per request) and density2000v2_cache.json
        for resuming.
Set before running: MAILTO, RADIUS_M, BATCH_SIZE and MIN_INTERVAL.
This takes hours; Ctrl+C is safe because the cache is written after every batch.
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
RADIUS_M = 2000

CACHE_FILE = "density2000v2_cache.json"
PROVENANCE_FILE = "density2000v2_provenance.json"
OUTPUT_FILE = "density_2000m_v2.csv"
UPPERBOUND_FILE = "density_5000m.csv"  # a 2 km count may never exceed the 5 km count at the same point

_cooldown_until = {}


def pick_endpoint():
    now = time.time()
    healthy = [e for e in OVERPASS_ENDPOINTS if _cooldown_until.get(e, 0) <= now]
    pool = healthy if healthy else OVERPASS_ENDPOINTS
    return pool[0]


def mark_failed(ep):
    _cooldown_until[ep] = time.time() + ENDPOINT_COOLDOWN_SECONDS
    # rotate the failed endpoint to the back of the queue
    OVERPASS_ENDPOINTS.append(OVERPASS_ENDPOINTS.pop(OVERPASS_ENDPOINTS.index(ep)))


def build_batch_query(coords):
    """
    One request per BATCH_SIZE coordinates, each ending in `out count`; Overpass
    returns the counts in statement order, so results map back positionally.

    The anchored regex ^(university|research_institute)$ selects the same feature
    set as 02_geocode_and_enrich.py, which uses a loose regex and then filters on
    exact tag equality.
    """
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


# OpenAlex, Nominatim and Overpass all ask for a contact address so they can
# reach the operator of a script that misbehaves.
MAILTO = "your-email@example.com"

HEADERS = {"User-Agent": f"scpark-network-research/1.0 (academic research; {MAILTO})"}


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
                print(f"  [429 from {ep}, waiting {wait} s and switching endpoint]")
                mark_failed(ep)  # rotate on 429 too, rather than hammering one instance
                time.sleep(wait)
                continue
            if r.status_code != 200:
                print(f"  [{r.status_code} from {ep}, switching endpoint]")
                mark_failed(ep)
                continue
            elements = r.json().get("elements", [])
            counts = [el for el in elements if el.get("type") == "count"]
            if len(counts) != len(coords):
                print(f"  [got {len(counts)} counts, expected {len(coords)}; switching endpoint]")
                mark_failed(ep)
                continue
            totals = [int(c.get("tags", {}).get("total", 0)) for c in counts]
            # Validation against an upper bound: the 2 km disc sits inside the
            # 5 km disc, so its count can never be larger. `bounds` holds the
            # negated 5 km counts, which lets the same comparison direction as
            # the lower-bound check in 13_fetch_density_5000m.py be reused. A
            # violation means the endpoint's database is regionally limited, so
            # the batch is discarded and the endpoint put into cooldown.
            bad = [i for i, ((lat, lon), t) in enumerate(zip(coords, totals))
                   if -t < bounds.get(f"{lat},{lon}", -10**9)]
            if bad:
                print(f"  [validation failed: {len(bad)} coordinates have a 2 km count "
                      f"above their 5 km count; {ep} looks regionally limited, "
                      f"switching endpoint]")
                mark_failed(ep)
                continue
            return totals, ep
        except Exception as e:
            print(f"  [{type(e).__name__} from {ep}, switching endpoint] {str(e)[:100]}")
            mark_failed(ep)
    return None, None


def main():
    nodes = json.load(open("std_nodes.json", encoding="utf-8"))
    coords = sorted(set((round(n["lat"], 7), round(n["lon"], 7)) for n in nodes))
    print(f"{len(nodes)} institutions, {len(coords)} distinct coordinates")

    bounds = {}
    if os.path.exists(UPPERBOUND_FILE):
        import csv
        with open(UPPERBOUND_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bounds[f"{row['lat']},{row['lon']}"] = -float(row["univ_research_count_5000m"])
        print(f"loaded {len(bounds)} upper-bound values for validation (stored negated)")
    else:
        print(f"{UPPERBOUND_FILE} not found; validation disabled (not recommended)")

    cache = {}
    if os.path.exists(CACHE_FILE):
        cache = json.load(open(CACHE_FILE, encoding="utf-8"))
        n0 = len(cache)
        # drop any cached value that a regionally limited endpoint may have poisoned
        cache = {k: v for k, v in cache.items() if -v >= bounds.get(k, -10**9)}
        if len(cache) < n0:
            print(f"discarded {n0 - len(cache)} cached values violating the 5 km upper bound")
        print(f"{len(cache)} values cached, resuming")

    provenance = []
    if os.path.exists(PROVENANCE_FILE):
        provenance = json.load(open(PROVENANCE_FILE, encoding="utf-8"))

    todo = [c for c in coords if f"{c[0]},{c[1]}" not in cache]
    print(f"{len(todo)} coordinates to query, batch size {BATCH_SIZE}, "
          f"roughly {len(todo) / BATCH_SIZE * (MIN_INTERVAL + 12) / 60:.0f} minutes")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
        totals, ep = run_batch(batch, bounds)
        round_failures = 0
        while totals is None:
            round_failures += 1
            if round_failures > 6:
                print("  [every endpoint failed 6 rounds running; stopping, progress saved, "
                      "re-run later to resume]")
                break
            wait = 300
            print(f"  [every endpoint failed for this batch; waiting {wait//60} min "
                  f"(round {round_failures}/6), progress saved]")
            time.sleep(wait)
            _cooldown_until.clear()  # reset all cooldowns and try the pool again
            totals, ep = run_batch(batch, bounds)
        if totals is None:
            break
        for (lat, lon), cnt in zip(batch, totals):
            cache[f"{lat},{lon}"] = cnt
        # Record when each batch was retrieved and from which instance: OSM
        # coverage grows over time, so the query date and instance belong in the
        # methods section and should be read from this file, not from memory.
        provenance.append({
            "utc_time": t0, "endpoint": ep,
            "n_coords": len(batch), "radius_m": RADIUS_M,
        })
        json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"))
        json.dump(provenance, open(PROVENANCE_FILE, "w", encoding="utf-8"), indent=1)
        done = len(cache)
        print(f"{done}/{len(coords)} coordinates done")
        time.sleep(MIN_INTERVAL)

    # Write whatever is finished, so partial progress can be inspected. Note that
    # the header below is emitted verbatim as in the original run; the column
    # holds the 2,000 m counts despite its name.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("lat,lon,univ_research_count_5000m\n")
        for c in coords:
            key = f"{c[0]},{c[1]}"
            if key in cache:
                f.write(f"{c[0]},{c[1]},{cache[key]}\n")
    print(f"wrote {OUTPUT_FILE} ({sum(1 for c in coords if f'{c[0]},{c[1]}' in cache)}"
          f"/{len(coords)} rows) and {PROVENANCE_FILE}")
    if all(f"{c[0]},{c[1]}" in cache for c in coords):
        print("All coordinates retrieved.")


if __name__ == "__main__":
    main()
