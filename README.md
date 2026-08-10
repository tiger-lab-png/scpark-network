# scpark-network

Reproducible pipeline for **"Pseudo-Replication in Affiliation-String Networks:
A Reporting Protocol for Small Spatial Effects in Bibliometrics"**
(Hung-Chi Chang, submitted to *Scientometrics*).

The paper shows that building institution-level collaboration networks from raw
affiliation strings commits pseudo-replication in Hurlbert's (1984) sense: the
effect estimate is left essentially unchanged while the *p* value is inflated by
twenty orders of magnitude. It packages the diagnostic into a fourteen-item
reporting protocol and stress-tests that protocol on a substantive question —
whether research institutions inside designated science parks occupy more central
positions in the global co-authorship network.

Everything reported in the manuscript can be re-executed from this repository.

---

## Quick start

```bash
pip install -r requirements.txt
python prepare_data.py     # decompress the deposited inputs (once)
python verify.py           # ~1 min: checks every deposited figure against the paper
```

`verify.py` exits non-zero if any deposited input is missing or if the
re-estimated headline specification does not match the manuscript. Add `--full`
to additionally re-run the 5,000-refit permutation analyses (1–2 hours).

---

## Which data belongs to which result

The manuscript reports **three** runs. They are deposited separately, and mixing
them will not reproduce anything.

| Directory | Contents | Where it appears in the paper |
|---|---|---|
| `data/full_run/` | **Full silicon-carbide population.** 54,671 works, 80,176 unique raw affiliation strings, 69,455 geocoded, 8,382 standardized institutions of which **7,886** carry valid coordinates. | **All inferential results.** Abstract, Tables 2–5, Figs. 1 and 3, Online Resource 1 (S1–S11). |
| `data/cart_run/` | CAR-T cell-therapy cross-field probe (capped sample). | Table 6 and the cross-field probe section. |
| `data/pilot_5000/` | The initial 5,000-work cursor-paginated sample: 10,404 raw strings, 2,663 standardized institutions (2,598 with coordinates). | Retained as the *designed sample-size comparison* described in Methodology, which separates effect-size stability from sample-size-driven changes in significance. It is **not** the analysis frame. |

Scripts read bare filenames, so run them from the directory holding the inputs:

```bash
cd data/full_run
python ../../analysis/15_reviewer_response_analyses.py
```

---

## Pipeline

Run in order. All intermediate outputs are deposited, so Steps 1–9 do not need to
be re-executed to reproduce any reported statistic.

| Step | Script | Purpose |
|---|---|---|
| 1 | `pipeline/01_fetch_openalex.py` | Cursor-paginated extraction of OpenAlex Topic **T10361** (Silicon Carbide Semiconductor Technologies), 2018–2024 → `affil_full.csv`, `addr_uniq_full.csv` |
| 1b | `pipeline/01b_fetch_openalex_cart.py` | Same extraction for the CAR-T cross-field probe |
| 2 | `pipeline/02_geocode_and_enrich.py` | Nominatim geocoding with the address-simplification cascade; Overpass `is_in()` park tagging and infrastructure-density covariates |
| 2b | `pipeline/02b_rerun_enrich_only.py` | Re-run the Overpass step without re-geocoding |
| 3 | `pipeline/03_fetch_wikidata_parks.py` | Wikidata SPARQL retrieval of 231 candidate science/technology parks |
| 4 | `pipeline/04_fetch_park_polygons.py` | Exact OSM polygon boundaries for the parks with a linked OSM relation |
| 5 | `pipeline/05_match_parks_distance.py` | Haversine distance classification (Method A) |
| 6 | `pipeline/06_apply_polygon_refinement.py` | Point-in-polygon refinement under the asymmetric confidence rule |
| 7 | `pipeline/07_merge_method_a_b.py` | Cross-validate Method A against Method B (OSM keyword tagging) |
| 8 | `pipeline/08_build_entity_resolved_network.py` | Entity-resolved network → `std_nodes.json`, `std_edges.json` (**primary analysis network**) |
| 9 | `pipeline/09_build_naive_network.py` | Naive raw-string network → `nodes.json`, `edges.json` (methodological comparison only) |

## Analysis

| Script | Reproduces |
|---|---|
| `analysis/10_robustness_checks.py` | Mann–Whitney tests, negative binomial regression, TOST equivalence, VIF, distance-threshold sweep, leave-one-park-out, label permutation, community composition, geocoding failure by country, Holm–Bonferroni |
| `analysis/11_institution_size_control.py` | Institution-productivity-controlled regression with MLE dispersion and 5-predictor VIF |
| `analysis/12_productivity_control_robustness.py` | Productivity-control robustness |
| `analysis/13_fetch_density_5000m.py` | 5,000 m density covariate retrieval (August 2026 OSM vintage) |
| `analysis/14_refetch_density_2000m.py` | Vintage-matched 2,000 m re-retrieval — establishes that the apparent radius effect was retrieval vintage, not spatial scale |
| `analysis/15_reviewer_response_analyses.py` | Hyper-authorship exclusion caps, bootstrap CIs and TOST, power grid, fractional counting, country fixed effects, betweenness seed stability, rank-biserial effect sizes, country-stratified permutation, affiliation-multiplicity audit |

### Manuscript location → script

| Manuscript location | Run from | Script |
|---|---|---|
| Table 2 — node-identity comparison, effect sizes | `data/full_run/` | `analysis/15_reviewer_response_analyses.py` |
| Tables 3–5 — regressions, threshold sweep, productivity control | `data/full_run/` | `analysis/10_robustness_checks.py`, `analysis/12_productivity_control_robustness.py` |
| Table 6 — CAR-T cross-field probe | `data/cart_run/` | `analysis/10_robustness_checks.py`, `analysis/15_reviewer_response_analyses.py` |
| Exclusion caps, equivalence tests, power grid, fractional counting, country fixed effects, betweenness seed stability, stratified permutation, affiliation-multiplicity audit | `data/full_run/` | `analysis/15_reviewer_response_analyses.py` |
| Online Resource 1, Tables S4–S10 | `data/full_run/` | `analysis/15_reviewer_response_analyses.py` |
| Fig. 1 — analysis pipeline | — | `figures/fig1_analysis_pipeline.dot` (`dot -Tpng -Gdpi=300`) |
| Fig. 3 — degree-centrality distributions | `data/full_run/` | `figures/fig3_degree_distributions.py` |

All random seeds are module constants at the top of
`analysis/15_reviewer_response_analyses.py` and are listed in Online Resource 1,
Table S10. They are arbitrary integers fixed for reproducibility; the eight-digit
form is a naming convention and encodes no execution date.

---

## Environment

```bash
pip install -r requirements.txt
```

Developed on Python 3.13.3 with the versions pinned in `requirements.txt`
(numpy 2.2.6, pandas 2.2.3, SciPy 1.15.2, statsmodels 0.14.6, NetworkX 3.4.2).
Reported estimates were independently reproduced under Python 3.11.15 with
numpy 2.3.5, pandas 3.0.2, SciPy 1.17.1 and NetworkX 3.6.1, so they are not
sensitive to numerical- or graph-library versions.

Files larger than 3 MB are stored gzipped to stay well below the 100 MB
per-file hosting limit; `prepare_data.py` restores them in place.

---

## Known limitations and audit notes

These are stated at length in the manuscript; the operative ones for anyone
re-using this code are:

- `Standardized_Institutions` is OpenAlex's own automated entity-resolution
  output, not a manually verified ground truth. Residual fragmentation was
  audited against ROR identifiers on a 5,000-work pilot extraction
  (Online Resource 1, S1); the full-population re-audit is out of scope and the
  claims in the paper are qualified accordingly.
- Geocoding succeeds for 86.6% of the 80,176 raw affiliation strings in the full
  population. Failure is not geographically random; rates are tabulated by
  institution country in the Results.
- Wikidata park coverage (231 registry entries) is the design's weakest
  component, and OSM polygon boundaries exist for only a minority of them.
  The distance proxy's internally measured false-positive rate is 41.5% at
  2,000 m and 54.1% at 5,000 m against exact geometry.
- Nominatim and the public Overpass API are volunteer-run and rate-limited.
  Re-running Steps 1–7 against live services takes several hours and is not
  required to reproduce any reported statistic.

## Supplementary material

`docs/` contains Online Resource 1, the supplementary tables accompanying the
manuscript.

## Licence and citation

MIT (see `LICENSE`). Cite via `CITATION.cff` or:

> Chang, H.-C. (2026). *scpark-network: Analysis pipeline for science-park
> proximity and academic co-authorship networks* [Computer software]. Zenodo.
