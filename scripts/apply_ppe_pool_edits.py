"""Apply the three reviewed label edits to the shared YOLO files, with backups.

Usage: python3 scripts/apply_ppe_pool_edits.py [--apply]
Dry-run is the default. Andromeda's database and prompt bank are not modified.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
POOL = Path("/opt/mount/project/data")
REPORT = ROOT / "reports/ppe-preparation-2026-11-08"
SOURCES = {81: "cap-pavithrap", 85: "safety-working"}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def matching_rows(lines, names, size, target_class, target_box):
    width, height = size
    found = []
    for i, line in enumerate(lines):
        fields = line.split()
        if not fields:
            continue
        assert len(fields) == 5, f"invalid YOLO line {i + 1}"
        cls = int(fields[0])
        cx, cy, w, h = map(float, fields[1:])
        assert all(math.isfinite(v) for v in (cx, cy, w, h))
        assert cls in names and 0 < w <= 1 and 0 < h <= 1
        box = ((cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height)
        if names[cls] == target_class and all(
            abs(a - b) < 1e-5 for a, b in zip(box, target_box)
        ):
            found.append(i)
    return found


def prepare():
    changes = json.loads((REPORT / "proposed-label-edits.json").read_text())
    assert len(changes) == 3, "this script is scoped to the three reviewed changes"
    planned = []
    for change in changes:
        dataset = SOURCES[change["project_id"]]
        root = POOL / dataset
        name = change["file_name"]
        assert Path(name).name == name
        names = yaml.safe_load((root / "data.yaml").read_text())["names"]
        names = dict(enumerate(names)) if isinstance(names, list) else names
        path = root / "labels" / Path(name).with_suffix(".txt")
        before = path.read_bytes()
        lines = before.decode().splitlines(keepends=True)
        with Image.open(root / "images" / name) as image:
            size = image.size
        old = change.get("before_bbox_xywh", change.get("bbox_xywh"))
        found = matching_rows(lines, names, size, change["class"], old)
        if "proposed_bbox_xywh" in change:
            assert len(found) == 1, f"source changed; expected one old box: {path}"
            x, y, w, h = change["proposed_bbox_xywh"]
            width, height = size
            assert (
                min(x, y) >= 0 and min(w, h) > 0 and x + w <= width and y + h <= height
            )
            cls = lines[found[0]].split()[0]
            new_line = f"{cls} {(x + w / 2) / width:.15g} {(y + h / 2) / height:.15g} {w / width:.15g} {h / height:.15g}\n"
            lines[found[0]] = new_line
            assert (
                len(
                    matching_rows(
                        lines,
                        names,
                        size,
                        change["class"],
                        change["proposed_bbox_xywh"],
                    )
                )
                == 1
            )
        else:
            assert len(found) == 2, (
                f"source changed; expected two duplicate boxes: {path}"
            )
            del lines[found[1]]
            assert len(matching_rows(lines, names, size, change["class"], old)) == 1
        after = "".join(lines).encode()
        assert before != after
        planned.append(
            {
                "dataset": dataset,
                "path": str(path),
                "image": name,
                "before_sha256": sha(before),
                "after_sha256": sha(after),
                "before_lines": len(before.splitlines()),
                "after_lines": len(after.splitlines()),
                "before_bytes": before,
                "after_bytes": after,
                "status": "planned",
            }
        )
    assert len({x["path"] for x in planned}) == 3
    return planned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    planned = prepare()
    public = lambda: [
        {k: v for k, v in r.items() if not k.endswith("_bytes")} for r in planned
    ]
    if not args.apply:
        print(json.dumps(public(), indent=2))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = REPORT / "pool-edit-backups" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    journal = backup / "journal.json"
    # Save every original before writing any shared file.
    for record in planned:
        dest = backup / record["dataset"] / Path(record["path"]).name
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(record["before_bytes"])
        assert sha(dest.read_bytes()) == record["before_sha256"]
        record["backup_path"] = str(dest)
    journal.write_text(json.dumps(public(), indent=2) + "\n")
    for record in planned:
        path = Path(record["path"])
        # Refuse to overwrite another teammate's edit after preflight.
        assert sha(path.read_bytes()) == record["before_sha256"], (
            f"concurrent source edit: {path}"
        )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".ppe-edit-", suffix=".tmp", delete=False
            ) as f:
                temporary = Path(f.name)
                f.write(record["after_bytes"])
            assert sha(temporary.read_bytes()) == record["after_sha256"]
            assert sha(path.read_bytes()) == record["before_sha256"], (
                f"concurrent source edit: {path}"
            )
            os.replace(temporary, path)
            temporary = None
            assert sha(path.read_bytes()) == record["after_sha256"]
            record["status"] = "applied_and_readback_verified"
        except Exception:
            record["status"] = "failed_check_journal_before_retry"
            raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            journal.write_text(json.dumps(public(), indent=2) + "\n")
    print(json.dumps({"journal": str(journal), "edits": public()}, indent=2))


if __name__ == "__main__":
    main()
