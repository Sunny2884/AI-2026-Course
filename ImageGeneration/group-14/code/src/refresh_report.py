"""Refresh saved checkpoint plots, baseline references and fixed-image comparisons."""

import argparse
import json
from pathlib import Path

from evaluate_checkpoints import report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from torch.utils.tensorboard import SummaryWriter
    config = json.loads((args.output/"config.json").read_text())
    results = json.loads((args.output/"metrics.json").read_text())
    sources = [json.loads(line) for line in (args.output/"conditions/conditions.jsonl").read_text().splitlines() if line.strip()]
    coverage = json.loads((args.output/"coverage.json").read_text())
    for row in results:
        if not (args.output/("epoch_{:03d}".format(row["epoch"]))/"metrics/summary.json").is_file():
            raise FileNotFoundError("Missing metric summary for epoch "+str(row["epoch"]))
    with SummaryWriter(str(args.output/"tensorboard")) as writer:
        report(args.output, results, config["split"], writer, sources[:5], coverage["requested"])


if __name__ == "__main__":
    main()
