"""Copy the frozen 2,250-image reviewed snapshot into a separate training dataset."""

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from legacy.apply_verified_ppe_labels import validate_labels

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/ppe-verified-20261109"
DEST = ROOT / "data/ppe-reviewed-2250"
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
CORE = [
    "Helmet",
    "Vest",
    "Mask_Gas",
    "Goggles",
    "Coverall",
    "Gloves",
    "Mask_Face",
    "Shoes",
    "Safety_Belt",
    "Ear_Protection",
    "no_Helmet",
    "no_Goggles",
    "no_Gloves",
    "no_Ear_Protection",
    "no_Mask_Face",
    "Cap",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = [
        json.loads(line)
        for line in (SNAPSHOT / "selected-source-manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    original = [
        json.loads(line)
        for line in (ROOT / "data/ppe-full-current-20261109/manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    membership = {(str(row["project_id"]), row["file_name"]): row for row in original}
    assert (
        len(rows) == 2250
        and len({(r["project_id"], r["image_id"]) for r in rows}) == 2250
    )
    assert not DEST.exists(), f"Destination already exists: {DEST}"
    prepared = []
    for row in rows:
        image, source_label = (
            Path(row["copied_image"]),
            Path(row["copied_source_label"]),
        )
        assert SNAPSHOT.resolve() in image.resolve().parents
        assert SNAPSHOT.resolve() in source_label.resolve().parents
        assert digest(image) == row["image_sha256"], f"Image snapshot changed: {image}"
        assert digest(source_label) == row["label_sha256"], (
            f"Label snapshot changed: {source_label}"
        )
        names = ["Cap"] if str(row["project_id"]) == "81" else CORE
        validate_labels(source_label.read_bytes(), set(range(len(names))))
        lines, boxes = [], []
        for line in source_label.read_text().splitlines():
            cls, *coordinates = line.split()
            name = names[int(cls)]
            if name.startswith("no_"):
                continue
            target = CLASSES.index(name)
            lines.append(str(target) + " " + " ".join(coordinates))
            boxes.append([target, *map(float, coordinates)])
        assert boxes == row["positive_boxes"], (
            f"Snapshot/selection boxes disagree: {source_label}"
        )
        content = ("\n".join(lines) + "\n").encode() if lines else b""
        validate_labels(content, set(range(11)))
        with Image.open(image) as decoded:
            decoded.verify()
        old = membership[(str(row["project_id"]), row["file_name"])]
        assert old["group"] == row["group"]
        prepared.append((row, image, source_label, content, old["split"]))
    totals, negatives = Counter(), Counter()
    support = {split: Counter() for split in ["train", "val", "test"]}
    group_splits, hash_splits = defaultdict(set), defaultdict(set)
    manifest = []
    for row, source_image, source_label, content, split in prepared:
        name = f"{row['project_id']}_{row['file_name']}"
        image = DEST / "images" / split / name
        label = DEST / "labels" / split / Path(name).with_suffix(".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image)
        label.write_bytes(content)
        assert digest(image) == row["image_sha256"] and label.read_bytes() == content
        totals[split] += 1
        negatives[split] += not bool(row["positive_boxes"])
        support[split].update(int(box[0]) for box in row["positive_boxes"])
        group_splits[row["group"]].add(split)
        hash_splits[row["image_sha256"]].add(split)
        manifest.append(
            {
                **row,
                "split": split,
                "image": str(image),
                "label": str(label),
                "release_label_sha256": digest(label),
                "snapshot_image": str(source_image),
                "snapshot_source_label": str(source_label),
            }
        )
    assert sum(totals.values()) == 2250
    assert all(len(splits) == 1 for splits in group_splits.values()), (
        "Original group leakage"
    )
    assert all(len(splits) == 1 for splits in hash_splits.values()), (
        "Exact-image leakage"
    )
    assert all(support[split][cls] > 0 for split in support for cls in range(11)), (
        "Missing class in a split"
    )
    assert (
        len(list((DEST / "images").glob("*/*")))
        == len(list((DEST / "labels").glob("*/*.txt")))
        == 2250
    )
    config = {
        "path": str(DEST),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 11,
        "names": CLASSES,
    }
    (DEST / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (DEST / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )
    summary = {
        "images": dict(totals),
        "total_images": 2250,
        "classes": CLASSES,
        "groups": len(group_splits),
        "group_leakage": 0,
        "exact_image_hash_leakage": 0,
        "empty_positive_labels": dict(negatives),
        "boxes_by_split": {
            split: {name: support[split][i] for i, name in enumerate(CLASSES)}
            for split in support
        },
        "selection_reasons": dict(Counter(row["selection_reason"] for row in rows)),
        "source": str(SNAPSHOT),
        "split_policy": "Preserved membership of the full-current first training release for the selected images.",
        "checks": [
            "All frozen image and label SHA256 hashes match selection manifest",
            "Positive boxes match the reviewed selection exactly",
            "All images decode successfully",
            "YOLO rows/classes/finite coordinates/bounds validated",
            "Destination image bytes verified",
            "All 11 classes present in each split",
            "No recorded original group or exact image hash crosses splits",
        ],
        "limitations": [
            "Uses historical complete-review selection; no new full visual re-audit or guarantee of error-free labels.",
            "Known groups and exact image hashes checked; arbitrary near-duplicate scenes are not exhaustively proven separate.",
            "Includes full frames and equipment product imagery from the reviewed selection; no new person-crop filtering.",
        ],
        "pool_modified": False,
        "training_started": False,
    }
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
