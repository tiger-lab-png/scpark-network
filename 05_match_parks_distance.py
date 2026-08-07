# data/

Two small reference files that the classification steps depend on. Both are
tracked in the repository so the pipeline can be run without first re-querying
Wikidata and OpenStreetMap.

## `parks_wikidata.csv`

The science-park registry, produced by `pipeline/03_fetch_wikidata_parks.py`.
One row per Wikidata item that is an instance of "science park" or "technology
park" (or a subclass of either) and carries coordinates.

| column | meaning |
| --- | --- |
| `wikidata_id` | Wikidata item id, e.g. `Q717461` |
| `lat`, `lon` | the item's coordinate (WGS84) |
| `osm_relation_id` | linked OpenStreetMap relation, where the item records one |
| `name` | English label, falling back to de/fr/ja/zh when no English label exists |
| `country` | label of the item's country |

Used by `05_match_parks_distance.py` as the reference point set for the
nearest-park distance rule.

## `park_polygons.json`

Exact park boundaries, produced by `pipeline/04_fetch_park_polygons.py` for the
registry entries that link an OSM relation. A JSON object keyed by
`wikidata_id`; each value is a **list of closed rings**, and each ring is a list
of `[lon, lat]` pairs. Several rings mean the park consists of detached sites, in
which case a point inside any ring counts as inside the park. Holes are not
represented. Ways that could not be chained into a closed ring were discarded
rather than force-closed, so these boundaries can understate a park's extent but
never overstate it — which is why `06_apply_polygon_refinement.py` trusts a
polygon "inside" verdict but falls back to the distance rule on "outside".

## Not included here

The intermediate and output files (`affil_full.csv`, `geocoded.csv`,
`enriched.csv`, `park_matches.csv`, `combined.csv`, `nodes.json`, `edges.json`,
`std_nodes.json`, `std_edges.json`, the geocoding and Overpass caches and the
density covariate files) are far too large for a code repository and are
archived separately; see the Zenodo release referenced in the top-level
`README.md`. Running the pipeline from step 01 regenerates all of them.
