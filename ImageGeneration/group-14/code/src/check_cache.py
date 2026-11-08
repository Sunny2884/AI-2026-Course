"""Validate an existing raw-feature cache without loading any visual encoder."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from model import FEATURE_SCHEMA, FEATURE_SHAPES
from prepare_features import digest, source_rows, cache_valid
from cache_io import validate_cached_row, read_features


def validate_cache(directory, csv_path, data_root, pairs=None, limit=None, deep=False):
    directory = Path(directory)
    for name in ("config.json", "complete.json", "conditions.jsonl"):
        if not (directory/name).is_file():
            raise FileNotFoundError("Feature cache incomplete: {}. Run scripts/prepare.sh first.".format(directory/name))
    config = json.loads((directory/"config.json").read_text())
    if limit is None:
        limit = config["limit"]
    complete = json.loads((directory/"complete.json").read_text())
    if config["schema"] != FEATURE_SCHEMA or config["csv_sha256"] != digest(csv_path):
        raise ValueError("Feature schema or split CSV changed")
    if config["pairs_sha256"] != (digest(pairs) if pairs else None) or config["limit"] != limit:
        raise ValueError("Feature cache pairing or extraction LIMIT differs; use the matching cache")
    if complete["fingerprint"] != config["fingerprint"] or complete["manifest_sha256"] != digest(directory/"conditions.jsonl"):
        raise ValueError("Feature cache completion/manifest mismatch")
    expected = source_rows(Path(csv_path), Path(data_root), pairs, limit)
    cached = [json.loads(line) for line in (directory/"conditions.jsonl").read_text().splitlines() if line.strip()]
    if len(cached) != len(expected) or len(cached) != complete["count"]:
        raise ValueError("Feature cache source count mismatch")
    for row, reference in zip(cached, expected):
        for key in ("id", "source_path", "target_path"):
            if row.get(key) != reference.get(key):
                raise ValueError("Cached source/supervision mismatch: "+row["id"])
        if row["feature_schema"] != FEATURE_SCHEMA or row["feature_fingerprint"] != config["fingerprint"]:
            raise ValueError("Mixed cache configurations")
        validate_cached_row(row)
        with safe_open(row["clip_feature_path"], framework="pt") as handle:
            metadata = handle.metadata()
        if deep:
            if metadata["source_sha256"] != digest(row["source_path"]) or not cache_valid(Path(row["clip_feature_path"]), metadata):
                raise ValueError("Source changed or cached feature corrupted")
            read_features(row)
    return config, cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "csv", "data-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--requested-limit", type=int, help="Require coverage for this inference prefix; zero means full split")
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    _, rows = validate_cache(args.cache, args.csv, args.data_root, args.pairs, args.limit, args.deep)
    if args.requested_limit is not None:
        requested = source_rows(args.csv, args.data_root, limit=args.requested_limit)
        if not {row["id"] for row in requested}.issubset({row["id"] for row in rows}):
            raise ValueError("Feature cache does not cover requested evaluation IDs")
    print("Reusing {} cached sources; no visual encoders loaded".format(len(rows)), flush=True)


if __name__ == "__main__":
    main()
