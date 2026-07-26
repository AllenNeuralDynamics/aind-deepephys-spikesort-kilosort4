"""Inventory a mounted Nextflow task folder and extract staged S3 paths."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


DATA = Path("../data")
RESULTS = Path("../results")
S3_PATTERN = re.compile(r"s3://[^\s'\"()]+")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in DATA.rglob("*") if path.is_file())
    inventory = []
    s3_paths = set()
    for path in files:
        relative = path.relative_to(DATA)
        inventory.append({"path": str(relative), "size": path.stat().st_size})
        if path.stat().st_size <= 1_000_000:
            text = path.read_text(errors="replace")
            s3_paths.update(S3_PATTERN.findall(text))
            if path.name.startswith(".command"):
                target = RESULTS / relative.name
                shutil.copyfile(path, target)
    output = {"files": inventory, "s3_paths": sorted(s3_paths)}
    (RESULTS / "task_inventory.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()