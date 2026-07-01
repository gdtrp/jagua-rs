#!/usr/bin/env python3
"""Generate one prod-test case folder per CSV row.

Reads nesting_data_*.csv (columns: request_data JSON, bin_width, bin_height, rotations, spacing)
and writes jagua-sqs-processor/tests/testdata/prod-tests/case-NNN/{data.json, <itemId>.svg}.
SVGs are downloaded from the request's private S3 url via `aws s3 cp` (cached by url).

Usage: python3 scripts/gen_prod_cases.py jagua-sqs-processor/tests/testdata/nesting_data_*.csv
"""
import csv, json, sys, os, glob, shutil, subprocess, urllib.parse
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROD = os.path.join(ROOT, "jagua-sqs-processor/tests/testdata/prod-tests")


def s3_fetch(url, dst):
    u = urllib.parse.urlparse(url)
    bucket, key = u.netloc.split(".")[0], u.path.lstrip("/")
    r = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", dst, "--region", "eu-north-1"],
        capture_output=True, text=True)
    return url, r.returncode, r.stderr.strip()[:120]


def valid_svg(path):
    if not os.path.exists(path):
        return False
    head = open(path, "rb").read(64).lstrip()
    return head.startswith(b"<?xml") or head.startswith(b"<svg")


def main(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    # clean previous generated cases
    for d in glob.glob(os.path.join(PROD, "case-*")):
        shutil.rmtree(d, ignore_errors=True)

    # collect unique urls, download once into a cache
    cache_dir = os.path.join(PROD, ".svg_cache")
    os.makedirs(cache_dir, exist_ok=True)
    urls = sorted({it["url"] for r in rows for it in json.loads(r["request_data"]).get("items", [])})
    cache = {u: os.path.join(cache_dir, urllib.parse.quote(u, safe="")) for u in urls}
    todo = [(u, p) for u, p in cache.items() if not valid_svg(p)]
    print(f"{len(urls)} unique SVGs, downloading {len(todo)} (cached {len(urls)-len(todo)}) ...")
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(lambda a: s3_fetch(*a), todo))
    bad_urls = {u for u, p in cache.items() if not valid_svg(p)}
    if bad_urls:
        print(f"WARNING: {len(bad_urls)} SVGs could not be downloaded; cases using them are skipped")

    seen = set()
    written = skipped_dup = skipped_dl = 0
    for i, r in enumerate(rows, 1):
        req = json.loads(r["request_data"])
        items = req.get("items", [])
        # dedupe identical requests (same items + bin/rotation/spacing)
        sig = (
            tuple(sorted((it["url"], it["count"], tuple(it.get("allowedRotations") or []))
                         for it in items)),
            r["bin_width"], r["bin_height"], r["rotations"], r["spacing"],
        )
        if sig in seen:
            skipped_dup += 1
            continue
        # skip cases that reference an SVG we couldn't fetch
        if any(it["url"] in bad_urls for it in items):
            skipped_dl += 1
            continue
        seen.add(sig)

        name = f"case-{len(seen):03d}"
        d = os.path.join(PROD, name)
        os.makedirs(d, exist_ok=True)
        json.dump({
            "description": f"prod row {i}",
            "csvRow": i,
            "binWidth": float(r["bin_width"]),
            "binHeight": float(r["bin_height"]),
            "spacing": float(r["spacing"]),
            "amountOfRotations": int(float(r["rotations"])),
            "items": items,
        }, open(os.path.join(d, "data.json"), "w"), indent=2, ensure_ascii=False)
        for it in items:
            shutil.copyfile(cache[it["url"]], os.path.join(d, f'{it["itemId"]}.svg'))
        written += 1
    print(f"generated {written} cases (skipped {skipped_dup} duplicates, "
          f"{skipped_dl} with undownloadable SVGs) under {PROD}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else glob.glob(os.path.join(PROD, "..", "nesting_data_*.csv"))[0])
