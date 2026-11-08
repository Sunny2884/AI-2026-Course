"""Generate timestamped collision-resistant run IDs, or recover one for resume."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo


def new_run_id():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12]


def resume_run_id(checkpoint):
    config = json.loads((checkpoint/"config.json").read_text())
    return config.get("run_id") or checkpoint.resolve().parent.name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(resume_run_id(args.resume) if args.resume else new_run_id())
