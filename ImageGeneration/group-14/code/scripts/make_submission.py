"""Pack 2,000 generated face images into a Kaggle submission CSV.

Images must be named after test IDs, for example face_00001.png or
face_00001.jpg. This is a packaging step, not an image evaluator.
"""

import argparse
import base64
import csv
import io
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="directory of generated images")
    parser.add_argument("--test", type=Path, default=Path(__file__).with_name("test.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    with args.test.open(newline="", encoding="utf-8-sig") as file:
        ids = [row["id"] for row in csv.DictReader(file)]
    if len(ids) != 2000 or len(set(ids)) != 2000:
        raise ValueError("test.csv must contain 2,000 unique IDs")
    if not args.images.is_dir():
        raise ValueError(f"Image directory does not exist: {args.images}")

    allowed = {".jpg", ".jpeg", ".png"}
    files = [p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    by_id = {}
    for path in files:
        if path.stem in by_id:
            raise ValueError(f"More than one image for {path.stem}")
        by_id[path.stem] = path
    missing = sorted(set(ids) - by_id.keys())
    extra = sorted(by_id.keys() - set(ids))
    if missing or extra:
        raise ValueError(f"Missing {len(missing)} IDs {missing[:3]}; extra {len(extra)} IDs {extra[:3]}")

    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "image_b64"])
        for image_id in ids:
            with Image.open(by_id[image_id]) as image:
                if image.width < 64 or image.height < 64:
                    raise ValueError(f"Image too small: {image_id}")
                output = io.BytesIO()
                image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS).save(
                    output, format="JPEG", quality=75, optimize=True
                )
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            if len(encoded) > 80_000 or len(output.getvalue()) > 60_000:
                raise ValueError(f"Image too large after encoding: {image_id}")
            writer.writerow([image_id, encoded])
    print(f"Wrote {len(ids)} rows to {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
