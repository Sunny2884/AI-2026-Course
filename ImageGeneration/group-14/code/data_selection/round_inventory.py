"""Validate completed teacher rounds and report gaps without treating IDs as K."""

import hashlib
import json


def inspect_rounds(round_root, csv_path, expected_stems, start, end):
    expected = set(expected_stems)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    entries = []
    for number in range(start, end+1):
        directory = round_root/("round_{:03d}".format(number))
        entry = {"round": number, "path": str(directory), "status": "missing"}
        if directory.is_dir():
            files = {p.stem for p in directory.iterdir() if p.suffix.lower() == ".png"}
            entry.update(images=len(files), missing_images=len(expected-files), extra_images=len(files-expected))
            complete = directory/"complete.json"
            config = directory/"config.json"
            if not complete.is_file() or not config.is_file():
                entry["status"] = "incomplete"
            else:
                info, settings = json.loads(complete.read_text()), json.loads(config.read_text())
                valid = (files == expected and info.get("images") == len(expected)
                         and settings.get("round") == number and settings.get("csv_sha256") == digest)
                entry["status"] = "complete" if valid else "inconsistent"
                entry["seed"] = settings.get("seed")
        entries.append(entry)
    return {"requested_start": start, "requested_end": end,
            "complete_rounds": [e["round"] for e in entries if e["status"] == "complete"],
            "unavailable_rounds": [e["round"] for e in entries if e["status"] != "complete"],
            "expected_sources_per_round": len(expected), "entries": entries}
