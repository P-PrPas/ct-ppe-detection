"""Verify frozen releases before training; never alter images, labels or splits."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CLASSES = [
    "Helmet",
    "Vest",
    "Mask_Gas",
    "Goggles",
    "Coverall",
    "Gloves",
    "Mask_Face",
    "Shoes",
    "Safety_Belt",
    "Cap",
    "Ear_Protection",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_release(data, policy, root):
    """Return verified input hashes, enforcing the selected release's extra guards."""
    require(
        yaml.safe_load((data / "data.yaml").read_text())["names"] == CLASSES,
        "PPE class order changed",
    )
    rows = read_rows(data / "manifest.jsonl")
    require(len(rows) == policy["images"], "Unexpected manifest image count")
    reference = policy.get("reference")
    reference_rows = (
        read_rows(root / reference / "input-hashes.jsonl") if reference else []
    )
    if reference:
        require(len(reference_rows) == len(rows), "Reference image count changed")
    frozen = {row["image"]: row for row in reference_rows}
    groups, hashes = defaultdict(set), defaultdict(set)
    counts, support = Counter(), defaultdict(set)
    snapshots = []
    for row in rows:
        image, label = Path(row["image"]), Path(row["label"])
        image_hash, label_hash = digest(image), digest(label)
        expected_image = row.get("image_sha256")
        expected_label = row.get("release_label_sha256", row.get("label_sha256"))
        if expected_image:
            require(image_hash == expected_image, f"Image changed: {image}")
        if expected_label:
            require(label_hash == expected_label, f"Label changed: {label}")
        if reference:
            previous = frozen.get(str(image))
            require(
                previous is not None
                and previous["label_sha256"] == label_hash
                and previous["image_sha256"] == image_hash,
                f"Reference input changed: {image}",
            )
        split = row["split"]
        counts[split] += 1
        groups[row["group"]].add(split)
        hashes[image_hash].add(split)
        for line in label.read_text().splitlines():
            if line.strip():
                fields = line.split()
                require(
                    len(fields) == 5 and 0 <= int(fields[0]) < len(CLASSES),
                    f"Invalid label: {label}",
                )
                support[split].add(int(fields[0]))
        snapshots.append(
            dict(
                image=str(image),
                label=str(label),
                image_sha256=image_hash,
                label_sha256=label_hash,
                group=row["group"],
                split=split,
            )
        )
    require(
        all(len(splits) == 1 for splits in groups.values()), "Family crosses splits"
    )
    require(
        all(len(splits) == 1 for splits in hashes.values()),
        "Exact image hash crosses splits",
    )
    if "counts" in policy:
        require(dict(counts) == policy["counts"], "Split counts changed")
        for split, count in counts.items():
            require(
                len(list((data / "images" / split).glob("*"))) == count,
                f"Image count changed: {split}",
            )
            require(
                len(list((data / "labels" / split).glob("*.txt"))) == count,
                f"Label count changed: {split}",
            )
    if policy.get("all_classes"):
        require(
            all(
                support[split] == set(range(len(CLASSES)))
                for split in ("train", "val", "test")
            ),
            "Missing class support",
        )
    if policy.get("baseline"):
        old = read_rows(root / policy["baseline"] / "manifest.jsonl")
        members = {(str(r["project_id"]), r["file_name"]): r["split"] for r in old}
        old_groups = {r["group"]: r["split"] for r in old}
        for row in rows:
            require(
                members.get((str(row["project_id"]), row["file_name"])) == row["split"],
                "Reviewed split membership changed",
            )
            require(
                old_groups.get(row["group"]) == row["split"],
                "Reviewed group crosses baseline split",
            )
    if policy.get("expanded"):
        summary = json.loads((data / "summary.json").read_text())
        require(
            summary["total_images"] == len(rows)
            and summary["group_leakage"] == 0
            and summary["benchmark_exact_leakage_train_val"] == 0,
            "Expansion integrity checks failed",
        )
        require(
            all(
                not r.get("benchmark_exact_match") or r["split"] == "test" for r in rows
            ),
            "Benchmark image in train/val",
        )
        if policy.get("core_evaluation"):
            require(
                summary["core_eval_hashes_and_labels_unchanged"]
                and summary["core_validation_images"] == 1337
                and summary["core_test_images"] == 2544,
                "Core evaluation changed",
            )
    return snapshots
