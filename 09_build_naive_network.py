"""
Refine the distance-based park classification with exact boundaries.

Reads:  park_polygons.json (step 04) and park_matches.csv (step 05).
Writes: park_matches.csv, updated in place with two extra columns:
        in_park_polygon_precise -- point-in-polygon verdict where the nearest
        park has boundary data, otherwise empty;
        in_park_best_guess      -- the verdict used downstream.

The polygon rule is deliberately asymmetric. Step 04 keeps only rings that
actually close and discards unclosed fragments, so the boundary data can
understate a park's extent but never overstate it. A True from the polygon test
is therefore trusted and overrides the distance rule, while a False (or missing
boundary) falls back to the distance rule rather than overturning it.
Point-in-polygon uses plain ray casting; at city scale a planar approximation is
sufficient and shapely is not needed.
"""

import json

import pandas as pd


def point_in_ring(lon, lat, ring):
    """Standard ray casting. `ring` is a closed [(lon, lat), ...] sequence."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > lat) != (yj > lat):
            x_intersect = (xj - xi) * (lat - yi) / (yj - yi) + xi
            if lon < x_intersect:
                inside = not inside
        j = i
    return inside


def point_in_multipolygon(lon, lat, rings):
    """True when the point falls inside any ring (parks can have detached sites)."""
    return any(point_in_ring(lon, lat, ring) for ring in rings)


if __name__ == "__main__":
    with open("park_polygons.json", "r", encoding="utf-8") as f:
        polygons = json.load(f)
    print(f"loaded exact boundaries for {len(polygons)} parks")

    df = pd.read_csv("park_matches.csv")

    precise_results = []
    changed_count = 0
    for row in df.itertuples():
        wikidata_id = row.nearest_park_wikidata_id
        if pd.isna(wikidata_id) or wikidata_id not in polygons:
            precise_results.append(None)
            continue

        rings = polygons[wikidata_id]
        result = point_in_multipolygon(row.Longitude, row.Latitude, rings)
        precise_results.append(result)

        if result != row.in_science_park_gt:
            changed_count += 1

    df["in_park_polygon_precise"] = precise_results
    # asymmetric trust: only a polygon True overrides the distance threshold
    df["in_park_best_guess"] = df["in_park_polygon_precise"].where(
        df["in_park_polygon_precise"] == True, df["in_science_park_gt"]
    )
    df.to_csv("park_matches.csv", index=False, encoding="utf-8-sig")

    n_precise = sum(1 for r in precise_results if r is not None)
    n_precise_true = sum(1 for r in precise_results if r is True)
    print(f"\n{n_precise} institutions have boundary data for their nearest park")
    print(f"inside the boundary: {n_precise_true}")
    print(f"disagreements with the distance rule: {changed_count} "
          f"(only the polygon Trues change the final verdict; see "
          f"in_park_best_guess)")
    print("park_matches.csv updated with in_park_polygon_precise / in_park_best_guess")

    # spot-check output for two campuses adjacent to a park boundary
    check = df[df["Raw_Affiliation"].str.contains("Chiao Tung|Tsing Hua", case=False, na=False)]
    if len(check) > 0:
        print("\nspot check:")
        print(check[["Raw_Affiliation", "distance_to_park_m", "in_science_park_gt",
                      "in_park_polygon_precise", "in_park_best_guess"]]
              .drop_duplicates(subset=["Raw_Affiliation"]).to_string(index=False))
