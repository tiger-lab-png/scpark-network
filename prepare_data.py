#!/usr/bin/env python3
"""Decompress the deposited data files in place.

Large inputs are stored gzipped so that no single file approaches the 100 MB
per-file ceiling of the hosting platform. The analysis scripts read plain
filenames, so run this once after cloning or unpacking the archive:

    python prepare_data.py

Re-running is safe: files that already exist are skipped unless --force.
"""
import argparse
import gzip
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that already exist")
    args = ap.parse_args()

    if not ROOT.is_dir():
        print(f"error: {ROOT} not found", file=sys.stderr)
        return 1

    archives = sorted(ROOT.rglob("*.gz"))
    if not archives:
        print("nothing to do: no .gz files under data/")
        return 0

    written = skipped = 0
    for src in archives:
        dst = src.with_suffix("")
        if dst.exists() and not args.force:
            skipped += 1
            continue
        with gzip.open(src, "rb") as fh_in, open(dst, "wb") as fh_out:
            shutil.copyfileobj(fh_in, fh_out, length=1 << 20)
        written += 1
        print(f"  {dst.relative_to(ROOT.parent)}  ({dst.stat().st_size:,} bytes)")

    print(f"\ndecompressed {written} file(s), skipped {skipped} already present")
    print("next: python verify.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
