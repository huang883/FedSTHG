"""Public ABIDE PCP ROI download with checksums and explicit cohort provenance."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import time
import numpy as np
import requests

META_API = ("https://api.github.com/repos/preprocessed-connectomes-project/abide/"
            "contents/Phenotypic_V1_0b_preprocessed1.csv")
BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"


def request(url, direct=False):
    last = None
    for attempt in range(3):
        try:
            with requests.Session() as session:
                session.trust_env = not direct
                r = session.get(url, timeout=(15, 60))
                r.raise_for_status()
                return r
        except requests.RequestException as exc:
            last = exc
            if getattr(exc.response, "status_code", None) in {403, 404}:
                break
            time.sleep(attempt + 1)
    raise last


def download_abide(out, direct=False, workers=6, limit=0, metadata=None):
    """limit: max subjects per site; 0 means all available rows, no QC selection.

    UM_1/UM_2 and UCLA_1/UCLA_2 are merged by institution (explicit B07).
    No label-dependent filtering to manufacture the paper's sample counts.
    """
    out = Path(out)
    (out / "roi").mkdir(parents=True, exist_ok=True)
    if metadata:
        raw = Path(metadata).read_bytes()
        meta_source = str(Path(metadata).resolve())
    else:
        payload = request(META_API, direct).json()
        raw = base64.b64decode(payload["content"])
        meta_source = META_API + " git_blob=" + payload["sha"]
    (out / "phenotypes.csv").write_bytes(raw)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    selected = []
    for site in ["NYU", "UM", "USM", "UCLA"]:
        candidates = sorted([r for r in rows if r["SITE_ID"].split("_")[0] == site],
                            key=lambda r: int(r["SUB_ID"]))
        for r in (candidates[:limit] if limit else candidates):
            selected.append((site, r))

    def one(item):
        site, r = item
        record = {"subject_id": r["SUB_ID"], "site": site, "source_site": r["SITE_ID"],
                  "label": 1 if r["DX_GROUP"] == "1" else 0, "file_id": r["FILE_ID"]}
        if r["DX_GROUP"] not in {"1", "2"} or r["FILE_ID"] in {"", "no_filename"}:
            return {**record, "status": "excluded", "reason": "missing derivative ID or diagnosis"}
        filename = r["FILE_ID"] + "_rois_aal.1D"
        url = f"{BASE}/cpac/filt_global/rois_aal/{filename}"
        path = out / "roi" / filename
        record.update(url=url, path="roi/" + filename)
        try:
            if not path.exists():
                content = request(url, direct).content
                part = path.with_suffix(".part")
                part.write_bytes(content)
                part.replace(path)
            content = path.read_bytes()
            x = np.loadtxt(io.BytesIO(content), comments="#")
            if x.ndim != 2 or x.shape[1] != 116 or x.shape[0] < 2 or not np.isfinite(x).all():
                raise ValueError(f"Invalid ROI array: {x.shape}")
            return {**record, "status": "ok", "sha256": sha256(content).hexdigest(),
                    "bytes": len(content), "shape": list(x.shape),
                    "constant_rois": int((np.std(x, axis=0) == 0).sum())}
        except Exception as exc:
            return {**record, "status": "excluded", "reason": str(exc)}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, record in enumerate(pool.map(one, selected), 1):
            records.append(record)
            if i % 25 == 0:
                print(f"Downloaded/checked {i}/{len(selected)}", flush=True)
    manifest = {"dataset": "ABIDE_PCP_cpac_filt_global", "synthetic": False,
                "metadata_source": meta_source, "metadata_sha256": sha256(raw).hexdigest(),
                "policy": "all listed NYU/UM/USM/UCLA with downloadable finite AAL116; no author QC list",
                "records": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counts = {}
    for r in records:
        key = r["site"] + "/" + r["status"]
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(counts, indent=2), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--direct", action="store_true", help="Ignore process proxy variables")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--metadata", help="Optional previously downloaded official phenotype CSV")
    args = p.parse_args()
    download_abide(**vars(args))


if __name__ == "__main__":
    main()
