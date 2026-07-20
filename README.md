# scpark-network

Reproducible pipeline for *Micro-Geographic Proximity and Global Academic Collaboration Networks
in an Emerging Semiconductor Field: An Open Pipeline Using OpenAlex, Wikidata, and OpenStreetMap*.

This repository accompanies the paper's H1/H2 test of whether academic institutions located inside
a formally designated science/technology park show higher co-affiliation network degree centrality
than comparable institutions outside such parks, and whether any such association survives
controlling for local infrastructure density and institution productivity.

Case study: OpenAlex Topic **T10361** ("Silicon Carbide Semiconductor Technologies"), 2018–2024,
5,000 works, 10,404 raw author-institution affiliation strings.

## Pipeline overview

Scripts in `pipeline/` are numbered in the order they must be run. Each stage reads the CSV/JSON
files produced by the previous stage(s); all intermediate and final data files are provided in
`data/` so the full pipeline does not need to be re-run to reproduce the analysis in `analysis/`.

| Step | Script | Input | Output | Purpose |
|---|---|---|---|---|
| 1 | `01_fetch_openalex.py` | OpenAlex API (Topic T10361) | `affil.csv`, `addr_uniq.csv` | Extract works, authorships, and raw affiliation strings via cursor pagination |
| 2 | `02_geocode_and_enrich.py` | `addr_uniq.csv` | `geocoded.csv`, `enriched.csv` | Nominatim geocoding (with address-simplification cascade) + Method B: OSM `is_in()` park tagging and infrastructure-density covariates via Overpass |
| 2b | `02b_rerun_enrich_only.py` | `geocoded.csv` | `enriched.csv` | Lightweight entry point to re-run only the Method B / Overpass step without re-geocoding |
| 3 | `03_fetch_wikidata_parks.py` | Wikidata Query Service (SPARQL) | `parks_wikidata.csv` | Authoritative ground-truth registry of 231 science/technology parks worldwide |
| 4 | `04_fetch_park_polygons.py` | `parks_wikidata.csv` (P402 links) | `park_polygons.json` | Exact OSM polygon boundaries for the 10 parks with a linked OSM relation |
| 5 | `05_match_parks_distance.py` | `geocoded.csv`, `parks_wikidata.csv` | `park_matches.csv` | Baseline haversine-distance park-proximity classification (Method A, distance-only) |
| 6 | `06_apply_polygon_refinement.py` | `park_polygons.json` | `park_matches.csv` (updated) | Refines Method A with exact point-in-polygon containment where available (asymmetric confidence rule) |
| 7 | `07_merge_method_a_b.py` | `park_matches.csv`, `enriched.csv` | `combined.csv` | Cross-validates Method A (Wikidata/polygon) against Method B (OSM keyword tagging); carries density covariates forward |
| 8 | `08_build_entity_resolved_network.py` | `affil.csv`, `park_matches.csv` | `std_nodes.json`, `std_edges.json` | Institution-level co-affiliation network using OpenAlex's standardized institution field (primary analysis network) |
| 9 | `09_build_naive_network.py` | `affil.csv`, `park_matches.csv` | `nodes.json`, `edges.json` | Same network built from raw, unresolved affiliation strings (methodological comparison network) |

Scripts in `analysis/` reproduce all statistics reported in the paper's Results section:

| Script | Reproduces |
|---|---|
| `10_robustness_checks.py` | Mann-Whitney tests, negative binomial regression (Table 2), TOST equivalence test, VIF, distance-threshold sensitivity (Table 3), Hsinchu leave-one-park-out, label-permutation test, community composition, geocoding-failure-by-country, Holm-Bonferroni correction |
| `11_institution_size_control.py` | Institution-productivity-controlled negative binomial regression (Table 4) with correctly MLE-estimated dispersion parameter and 5-predictor VIF diagnostics. Imports `10_robustness_checks.py` at runtime via `importlib` (its filename starts with a digit, so it cannot be imported with a plain `import` statement) — keep both files in the same directory. |

## Setup

```bash
pip install -r requirements.txt
```

Set your own contact email in the `MAILTO` constant near the top of `01_fetch_openalex.py` before
running Step 1 (required to enter OpenAlex's "polite pool" for faster, more reliable API access).
Nominatim (Step 2) and the public Overpass API (Steps 2, 4, 6) are rate-limited, third-party,
volunteer-run services; re-running Steps 1–7 from scratch against live APIs can take several hours
and is not required to reproduce the statistical results, since all intermediate outputs are
included in `data/`.

**Path convention:** every script reads/writes bare filenames (e.g. `pd.read_csv("affil.csv")`)
rather than `data/affil.csv`, because they were developed and run with all data files and scripts
in one flat working directory. To reproduce the analysis without editing paths, run scripts with
`data/` as your working directory, e.g.:

```bash
cd data
python ../analysis/10_robustness_checks.py
python ../analysis/11_institution_size_control.py
```

New output files (e.g. `ror_audit_mismatches.csv`) will be written into `data/` alongside the
inputs. If you'd rather keep the folder split strictly read-only, copy the scripts you want to run
into `data/` instead and run them from there.

## Data

See `data/README.md` for a description of every file, its provenance, and known limitations
(most importantly: geocoding failure rate, ~9.3%; and OSM park-boundary polygon coverage limited
to 10 of 231 candidate parks).

## Known limitations / audit notes

- `Standardized_Institutions` in `affil.csv` is OpenAlex's own automated entity-resolution output,
  not a manually verified ground truth. See the paper's Limitations section and any accompanying
  ROR cross-validation audit for a quantified assessment of resid