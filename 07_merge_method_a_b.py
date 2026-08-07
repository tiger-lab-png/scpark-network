"""
Match institution coordinates to the nearest registry park (method A).

Reads:  geocoded.csv (step 02) and parks_wikidata.csv (step 03).
Writes: park_matches.csv -- for every geocoded affiliation, the nearest park,
        its Haversine distance in metres, and whether that distance is within
        MATCH_RADIUS_M.
Set before running: MATCH_RADIUS_M (default 2,000 m, matching the density search
radius used in step 02 so the two methods stay comparable).
This is a point-to-point distance rule, refined into a true point-in-polygon
test for parks with boundary data in step 06. It runs entirely offline.
"""

import math

import numpy as np
import pandas as pd

MATCH_RADIUS_M = 2000  # same radius as the method B density search


def haversine(lat1, lon1, lat2, lon2):
    """Scalar version, kept for single-point use; match_all() vectorises the same formula."""
    R = 6371000  # metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def match_all(df_geocoded, df_parks):
    """
    Compute all institution-to-park distances as one (n_institutions x n_parks)
    numpy broadcast. Mathematically identical to calling haversine() per pair,
    but a per-row pandas loop over tens of thousands of coordinates takes tens of
    minutes where this takes seconds.
    """
    valid = df_geocoded.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    R = 6371000.0
    lat1 = np.radians(valid["Latitude"].to_numpy())[:, None]      # (N, 1)
    lon1 = np.radians(valid["Longitude"].to_numpy())[:, None]     # (N, 1)
    lat2 = np.radians(df_parks["lat"].to_numpy())[None, :]        # (1, M)
    lon2 = np.radians(df_parks["lon"].to_numpy())[None, :]        # (1, M)

    dphi = lat2 - lat1
    dlambda = lon2 - lon1
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2) ** 2
    dist_matrix = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))    # (N, M)

    nearest_idx = np.argmin(dist_matrix, axis=1)
    nearest_dist = dist_matrix[np.arange(len(valid)), nearest_idx]

    results = pd.DataFrame({
        "Raw_Affiliation": valid["Raw_Affiliation"],
        "Latitude": valid["Latitude"],
        "Longitude": valid["Longitude"],
        "nearest_park_name": df_parks["name"].to_numpy()[nearest_idx],
        "nearest_park_wikidata_id": df_parks["wikidata_id"].to_numpy()[nearest_idx],
        "distance_to_park_m": np.round(nearest_dist, 1),
        "in_science_park_gt": nearest_dist <= MATCH_RADIUS_M,
    })
    return results


if __name__ == "__main__":
    df_geocoded = pd.read_csv("geocoded.csv")
    df_parks = pd.read_csv("parks_wikidata.csv")
    print(f"{len(df_geocoded)} institution coordinates, {len(df_parks)} registry parks")

    df_matches = match_all(df_geocoded, df_parks)
    df_matches.to_csv("park_matches.csv", index=False, encoding="utf-8-sig")

    hit_count = df_matches["in_science_park_gt"].sum()
    print(f"\nwithin {MATCH_RADIUS_M} m of a registry park: {hit_count} "
          f"({hit_count / len(df_matches) * 100:.1f}%)")
    print("output: park_matches.csv")
