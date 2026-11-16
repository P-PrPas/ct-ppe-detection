"""Freeze an explicitly split source manifest into a portable YOLO release."""

import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from core.validation import CLASSES, digest, read_rows, require


def build_release(config):
    """Copy source inputs only after checking split isolation and benchmark exclusion."""
    source = Path(config["manifest"])
    output = Path(config["out"])
    require(not output.exists(), f"Release already exists: {output}")
    source_hash = digest(source)
    rows = read_rows(source)
    require(
        len(rows) == config["expected_images"], "Source image count differs from config"
    )
    protected = set()
    for path in config["protected_exams"]:
        exam = json.loads(Path(path).read_text())
        protected.update(member["sha256"] for member in exam["members"])
    groups, hashes = defaultdict(set), defaultdict(set)
    prepared = []
    for index, row in enumerate(rows):
        split, group = row["split"], row["group"]
        require(
            split in {"train", "val", "test"}
            and isinstance(group, str)
            and bool(group),
            "Each input needs a split and nonempty family group",
        )
        image = (source.parent / row["image"]).resolve()
        label = (source.parent / row["label"]).resolve()
        image_hash = digest(image)
        label_hash = digest(label)
        for key, actual in [("image_sha256", image_hash), ("label_sha256", label_hash)]:
            if key in row:
                require(row[key] == actual, f"Source hash changed: {image}")
        require(
            image_hash not in protected or split == "test",
            f"Benchmark image in train/val: {image}",
        )
        groups[group].add(split)
        hashes[image_hash].add(split)
        with Image.open(image) as opened:
            opened.verify()
        names = row.get("names", CLASSES)
        require(len(set(names)) == len(names), "Duplicate source class names")
        lines = []
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split()
            require(len(fields) == 5, f"Invalid YOLO label: {label}")
            class_id = int(fields[0])
            require(0 <= class_id < len(names), f"Unknown class ID: {label}")
            name = names[class_id]
            require(
                name in CLASSES, f"Explicitly review unsupported source class: {name}"
            )
            x, y, width, height = map(float, fields[1:])
            require(
                all(math.isfinite(v) for v in (x, y, width, height))
                and 0 <= x <= 1
                and 0 <= y <= 1
                and 0 < width <= 1.00001
                and 0 < height <= 1.00001,
                f"Invalid box: {label}",
            )
            lines.append(f"{CLASSES.index(name)} " + " ".join(fields[1:]))
        filename = f"{index:08d}{image.suffix.lower()}"
        prepared.append((row, image, label, image_hash, label_hash, filename, lines))
    require(all(len(s) == 1 for s in groups.values()), "Family crosses splits")
    require(all(len(s) == 1 for s in hashes.values()), "Exact duplicate crosses splits")
    counts = Counter(row["split"] for row in rows)
    require(
        set(counts) == {"train", "val", "test"}, "All three splits must contain images"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ppe-release-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "release"
        manifest = []
        for row, image, label, image_hash, label_hash, filename, lines in prepared:
            relative_image = Path("images") / row["split"] / filename
            relative_label = (
                Path("labels") / row["split"] / Path(filename).with_suffix(".txt")
            )
            copied = staging / relative_image
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, copied)
            require(
                digest(copied) == image_hash and digest(label) == label_hash,
                f"Source changed during build: {image}",
            )
            copied_label = staging / relative_label
            copied_label.parent.mkdir(parents=True, exist_ok=True)
            copied_label.write_text("\n".join(lines) + "\n" if lines else "")
            manifest.append(
                dict(
                    image=str(output / relative_image),
                    label=str(output / relative_label),
                    image_sha256=image_hash,
                    label_sha256=digest(copied_label),
                    source_image=str(image),
                    source_label=str(label),
                    source_label_sha256=label_hash,
                    group=row["group"],
                    split=row["split"],
                    benchmark_exact_match=image_hash in protected,
                )
            )
        (staging / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in manifest)
        )
        (staging / "data.yaml").write_text(
            yaml.safe_dump(
                dict(
                    path=str(output),
                    train="images/train",
                    val="images/val",
                    test="images/test",
                    names=CLASSES,
                ),
                sort_keys=False,
            )
        )
        summary = dict(
            total_images=len(rows),
            images=dict(counts),
            classes=CLASSES,
            group_leakage=0,
            exact_hash_split_leakage=0,
            benchmark_exact_leakage_train_val=0,
            protected_exams=config["protected_exams"],
            source_manifest_sha256=source_hash,
        )
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (staging / "build-config.json").write_text(json.dumps(config, indent=2) + "\n")
        require(not output.exists(), f"Release appeared during build: {output}")
        require(digest(source) == source_hash, "Source manifest changed during build")
        staging.rename(output)
    return summary
