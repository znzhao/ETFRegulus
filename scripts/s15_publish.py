"""Stage 15 -- the web app's data set (WEBAPP_PLAN.md Step 3).

    python -m scripts.s15_publish --out site_data
    python -m scripts.s15_publish --out site_data --only spy_tlt_60_40,momentum --ceilings 5

Reads the curated prices (Stages 1-2), runs every strategy x ceiling under
`config/experiments/webapp.yaml`, writes the JSON files of the data contract, and checks
them with `src.publish.contract`. Exits non-zero, leaving the output directory untouched,
if anything fails: the new data is built in a temporary directory and swapped in only once
it is valid.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from src.publish import contract, pipeline


def write(docs: dict[str, dict], out: Path) -> None:
    for rel, doc in docs.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="data directory to (re)write")
    ap.add_argument("--config", default=pipeline.Settings.config)
    ap.add_argument("--start", default=pipeline.Settings.start)
    ap.add_argument("--end", default=None, help="last session to use (default: all curated data)")
    ap.add_argument("--only", default=None, help="comma-separated strategies (default: all six)")
    ap.add_argument("--ceilings", default=",".join(map(str, pipeline.CEILINGS)))
    a = ap.parse_args(argv)

    settings = pipeline.Settings(
        config=a.config, start=a.start, end=a.end,
        ceilings=tuple(int(c) for c in a.ceilings.split(",")),
        strategies=tuple(a.only.split(",")) if a.only else pipeline.Settings.strategies)
    t0 = time.time()
    docs = pipeline.build_site(settings)

    tmp = a.out.with_name(a.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    write(docs, tmp)
    errs = contract.validate_site(tmp)
    if errs:
        for e in errs:
            print(e)
        print(f"{len(errs)} contract problem(s); {a.out} left unchanged, invalid output in {tmp}")
        return 1
    if a.out.exists():
        shutil.rmtree(a.out)
    tmp.rename(a.out)
    size = sum(f.stat().st_size for f in a.out.rglob("*.json"))
    print(f"wrote {a.out} ({size / 1e6:.1f} MB, {len(docs)} files) in {time.time() - t0:.0f}s: valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
