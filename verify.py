#!/usr/bin/env python3
"""verify.py - check that this deposit reproduces the manuscript.

The failure this script exists to prevent is a deposit whose scripts are present
and correct but whose inputs are absent, incomplete, or drawn from a different
run. Such a deposit passes visual inspection and fails on execution. Run:

    python prepare_data.py      # once, to decompress
    python verify.py            # fast checks   (~1 minute)
    python verify.py --full     # + permutation tests (~1-2 hours)

Exit status is 0 only if every check passes.
"""
import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FULL = ROOT / "data" / "full_run"
CART = ROOT / "data" / "cart_run"
PILOT = ROOT / "data" / "pilot_5000"

# Values as printed in the manuscript. Tolerances are absolute.
TOL = 5e-4

REQUIRED = {
    FULL: ["affil_full.csv", "park_matches.csv", "combined.csv", "geocoded.csv",
           "std_nodes.json", "std_edges.json", "nodes.json", "edges.json",
           "density_5000m.csv", "density_2000m_v2.csv", "addr_uniq_full.csv",
           "enriched.csv", "parks_wikidata.csv", "park_polygons.json"],
    CART: ["affil_full.csv", "park_matches.csv", "combined.csv", "geocoded.csv",
           "std_nodes.json", "density_5000m.csv"],
    PILOT: ["affil.csv", "park_matches.csv", "combined.csv", "geocoded.csv",
            "std_nodes.json", "nodes.json"],
}

results = []


def check(name, got, want, tol=None):
    if tol is None:
        ok = got == want
    else:
        ok = got is not None and abs(float(got) - float(want)) <= tol
    results.append((ok, name, got, want))
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<46} got {got!s:<14} expected {want}")
    return ok


def section(title):
    print(f"\n{title}\n" + "-" * len(title))


# --------------------------------------------------------------------------- #
# 1. Presence
# --------------------------------------------------------------------------- #
def check_presence():
    section("1. Required inputs present")
    ok = True
    for d, files in REQUIRED.items():
        for f in files:
            p = d / f
            present = p.is_file() and p.stat().st_size > 0
            results.append((present, f"{d.name}/{f}", "present" if present else "MISSING", "present"))
            if not present:
                print(f"  [FAIL] {d.name}/{f} missing "
                      f"(run `python prepare_data.py` first if a .gz is there)")
                ok = False
    if ok:
        print(f"  [ok ] all {sum(len(v) for v in REQUIRED.values())} required input files present")
    return ok


# --------------------------------------------------------------------------- #
# 2. Corpus dimensions (Methodology; Abstract)
# --------------------------------------------------------------------------- #
def check_dimensions():
    section("2. Corpus dimensions against the manuscript")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    papers, insts = set(), set()
    with open(FULL / "affil_full.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            papers.add(row["Paper_ID"])
            for s in (row.get("Standardized_Institutions") or "").split(","):
                s = s.strip()
                if s:
                    insts.add(s)
    check("works retrieved (unique Paper_ID)", len(papers), 54671)
    check("standardized institutions (all)", len(insts), 8382)

    with open(FULL / "addr_uniq_full.csv", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)
        addrs = {r[0].strip() for r in rdr if r and r[0].strip()}
    check("unique raw affiliation strings", len(addrs), 80176)

    with open(FULL / "geocoded.csv", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    geo = sum(1 for r in rows if (r.get("Latitude") or "").strip() not in ("", "nan"))
    check("geocoded rows", len(rows), 80176)
    check("rows with valid coordinates", geo, 69455)
    check("geocoding success rate (%)", round(100 * geo / len(rows), 1), 86.6, tol=0.05)

    for fname, want, label in [("std_nodes.json", 7886, "entity-resolved nodes (with coords)"),
                               ("nodes.json", 69455, "naive nodes (with coords)")]:
        with open(FULL / fname, encoding="utf-8") as fh:
            check(label, len(json.load(fh)), want)

    with open(PILOT / "std_nodes.json", encoding="utf-8") as fh:
        check("pilot: entity-resolved nodes with coords", len(json.load(fh)), 2598)
    pilot_insts = set()
    with open(PILOT / "affil.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            for s in (row.get("Standardized_Institutions") or "").split(","):
                s = s.strip()
                if s:
                    pilot_insts.add(s)
    check("pilot: standardized institutions (all)", len(pilot_insts), 2663)

    return all(r[0] for r in results)


# --------------------------------------------------------------------------- #
# 3. Headline estimate (Abstract; Results, Table 3/5)
# --------------------------------------------------------------------------- #
def check_headline():
    section("3. Headline specification re-estimated from the deposited data")
    spec = importlib.util.spec_from_file_location(
        "rr", ROOT / "analysis" / "15_reviewer_response_analyses.py")
    rr = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(FULL)
    try:
        spec.loader.exec_module(rr)
        affil, park, combined, geocoded, nodes, density = rr.load_inputs()
        dmap = rr.density_by_affiliation(geocoded, density)
        lookup = rr.build_institution_lookup(
            affil, park, combined, rr.TREATMENT_RADIUS_M, dmap)
        frame = rr.build_frame(nodes, lookup)
        s = rr.summarise(frame, label="headline")
    finally:
        os.chdir(cwd)

    check("N (institutions in regression)", int(s["n"]), 7886)
    check("treated at 5,000 m", int(s["treated"]), 504)
    check("park IRR", round(s["irr"], 4), 1.2348, tol=TOL)
    check("95% CI lower", round(s["ci95_low"], 3), 1.109, tol=TOL)
    check("95% CI upper", round(s["ci95_high"], 3), 1.375, tol=TOL)
    check("model-based p", float(f"{s['model_p']:.5f}"), 0.00012, tol=1e-5)
    return all(r[0] for r in results)


# --------------------------------------------------------------------------- #
# 4. Optional: the permutation-based figures of record
# --------------------------------------------------------------------------- #
def check_full():
    section("4. Full re-run of 15_reviewer_response_analyses.py")
    print("  running with fixed seeds; this takes 1-2 hours ...")
    script = ROOT / "analysis" / "15_reviewer_response_analyses.py"
    proc = subprocess.run([sys.executable, str(script)], cwd=FULL)
    if proc.returncode != 0:
        results.append((False, "script 15 exit status", proc.returncode, 0))
        print("  [FAIL] script exited non-zero")
        return False

    with open(FULL / "reviewer_response_results.json", encoding="utf-8") as fh:
        out = json.load(fh)
    flat = json.dumps(out)
    print("  wrote reviewer_response_results.json "
          f"({len(flat):,} chars) and reviewer_response_tables.csv")
    print("  compare reviewer_response_tables.csv against Tables 2, 5 and S4-S10 "
          "of the manuscript.")
    results.append((True, "script 15 completed", "ok", "ok"))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also re-run the 5,000-refit permutation analyses")
    args = ap.parse_args()

    print("scpark-network - deposit verification")
    print("=" * 60)

    if not check_presence():
        print("\nRESULT: FAIL - inputs missing; nothing further was attempted.")
        return 1
    check_dimensions()
    check_headline()
    if args.full:
        check_full()

    failed = [r for r in results if not r[0]]
    print("\n" + "=" * 60)
    if failed:
        print(f"RESULT: FAIL - {len(failed)} of {len(results)} checks did not pass:")
        for _, name, got, want in failed:
            print(f"  - {name}: got {got}, expected {want}")
        return 1
    print(f"RESULT: PASS - all {len(results)} checks reproduce the manuscript.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
