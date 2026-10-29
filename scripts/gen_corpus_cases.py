#!/usr/bin/env python3
"""Add production nesting requests to the RFC-030 oracle corpus (dxf-processor N0).

Unlike gen_prod_cases.py this never deletes a case: it appends case-NNN folders after the highest
existing one, skips requests that duplicate an existing case, and keeps the columns the newer
nesting_data rows carry (max_fit, fill_direction, start_corner). Each case gets data.json, the
part SVGs (the request's `url`) and the part DXFs (`dxfUrl`, under dxf/<key>).

Input: a CSV export of nesting_data with the header
  id,request_data,bin_width,bin_height,rotations,spacing,max_fit,fill_direction,start_corner,created

Objects come from VK object storage through cutl-infra's vk-s3.sh, one at a time. Keys are the
same as in the AWS-era URLs; the bucket is cutl-production-data.

Usage: python3 scripts/gen_corpus_cases.py <export.csv> <corpus>/cases <corpus>/dxf <path/to/vk-s3.sh>
"""
import csv
import glob
import json
import os
import subprocess
import sys
import urllib.parse

BUCKET = "cutl-production-data"
# The backend's constant (cutl-backend NestingRequestFactory.globalDefaultPolicy).
OFFCUT_POLICY = {"minOffcutWidthMm": 100.0, "minOffcutHeightMm": 100.0,
                 "shape": "RECTANGLE_MERGED", "kerfMm": 0.0}


def key_of(url):
    """The object key of an AWS-era URL, a VK URL or a bare key."""
    if not url:
        return None
    u = urllib.parse.urlparse(url)
    if not u.scheme:
        return url.lstrip("/")
    path = urllib.parse.unquote(u.path.lstrip("/"))
    if "vkcs.cloud" in u.netloc:  # path-style: /<bucket>/<key>
        return path.split("/", 1)[1] if "/" in path else None
    return path  # virtual-host style (amazonaws.com, data.cutl.online)


def fetch(vk, key, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = subprocess.run([vk, "cp", f"s3://{BUCKET}/{key}", dst], capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0


def signature(binw, binh, rot, spacing, max_fit, fill, corner, items):
    return (
        tuple(sorted((key_of(it["url"]), it["count"], tuple(it.get("allowedRotations") or []))
                     for it in items)),
        float(binw), float(binh), int(float(rot)), float(spacing),
        bool(max_fit), fill or "STAIRCASE", corner or "TOP_LEFT",
    )


def main(csv_path, cases_dir, dxf_dir, vk):
    csv.field_size_limit(sys.maxsize)  # request_data of a large job exceeds the 128 KB default
    existing = sorted(glob.glob(os.path.join(cases_dir, "case-*")))
    seen = set()
    for d in existing:
        v = json.load(open(os.path.join(d, "data.json")))
        seen.add(signature(v["binWidth"], v["binHeight"], v["amountOfRotations"], v["spacing"],
                           v.get("maxFit"), v.get("fillDirection"), v.get("startCorner"),
                           v["items"]))
    n = max((int(os.path.basename(d)[5:]) for d in existing), default=0)
    written = dup = missing = 0
    for row in csv.DictReader(open(csv_path)):
        items = json.loads(row["request_data"]).get("items", [])
        max_fit = row.get("max_fit", "").lower() in ("t", "true", "1")
        sig = signature(row["bin_width"], row["bin_height"], row["rotations"], row["spacing"],
                        max_fit, row.get("fill_direction") or None,
                        row.get("start_corner") or None, items)
        if sig in seen:
            dup += 1
            continue
        name = f"case-{n + 1:03d}"
        d = os.path.join(cases_dir, name)
        ok = True
        for it in items:
            svg_key = key_of(it.get("url"))
            if not svg_key or not fetch(vk, svg_key, os.path.join(d, f'{it["itemId"]}.svg')):
                ok = False
                break
            dxf_key = key_of(it.get("dxfUrl"))
            if dxf_key and not fetch(vk, dxf_key, os.path.join(dxf_dir, dxf_key)):
                print(f"{name}: DXF missing {dxf_key}", file=sys.stderr)
        if not ok:
            missing += 1
            subprocess.run(["rm", "-rf", d])
            continue
        n += 1
        seen.add(sig)
        doc = {
            "description": f"prod nesting {row['id']}",
            "nestingId": row["id"],
            "created": row.get("created"),
            "binWidth": float(row["bin_width"]),
            "binHeight": float(row["bin_height"]),
            "spacing": float(row["spacing"]),
            "amountOfRotations": int(float(row["rotations"])),
            "maxFit": max_fit,
            "offcutPolicy": OFFCUT_POLICY,
            "items": items,
        }
        if row.get("fill_direction"):
            doc["fillDirection"] = row["fill_direction"]
        if row.get("start_corner"):
            doc["startCorner"] = row["start_corner"]
        json.dump(doc, open(os.path.join(d, "data.json"), "w"), indent=2, ensure_ascii=False)
        written += 1
    print(f"added {written} cases, skipped {dup} duplicates, {missing} with a missing SVG")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
