"""Build native-resolution person crops and leakage-checked grouped PPE splits."""

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/ppe-verified-20261109"
CLASSES = json.loads((OUT / "selection-summary.json").read_text())["classes"]


def phash(image):
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(
        np.float32
    )
    values = cv2.dct(small)[:8, :8].flatten()[1:]
    return sum(int(v > np.median(values)) << i for i, v in enumerate(values))


def main():
    rows = [
        json.loads(s)
        for s in (OUT / "selected-source-manifest.jsonl").read_text().splitlines()
    ]
    metadata = {str(Path(r["copied_image"]).resolve()): r for r in rows}
    parent = {r["group"]: r["group"] for r in rows}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    hashes = []
    for row in rows:
        im = cv2.imread(row["copied_image"])
        assert im is not None
        h, w = im.shape[:2]
        value = phash(im)
        for previous, aspect, group in hashes:
            if abs(w / h / aspect - 1) < 0.05 and (previous ^ value).bit_count() <= 4:
                parent[find(row["group"])] = find(group)
        hashes.append((value, w / h, row["group"]))
    for row in rows:
        row["group"] = find(row["group"])
    detector = YOLO(str(ROOT / "models/yolo11n.pt"))
    images = OUT / "person-crops/images"
    labels = OUT / "person-crops/labels"
    images.mkdir(parents=True, exist_ok=False)
    labels.mkdir(parents=True)
    crops, exclusions = [], Counter()
    results = detector.predict(
        [r["copied_image"] for r in rows],
        classes=[0],
        conf=0.3,
        imgsz=960,
        batch=16,
        device=1,
        stream=True,
        verbose=False,
    )
    processed = 0
    for result in results:
        row = metadata[str(Path(result.path).resolve())]
        im = result.orig_img
        height, width = im.shape[:2]
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            exclusions["no_person_detected"] += 1
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            pw, ph = x2 - x1, y2 - y1
            if pw < 32 or ph < 48:
                exclusions["person_too_small"] += 1
                continue
            left, top = max(0, int(x1 - 0.12 * pw)), max(0, int(y1 - 0.10 * ph))
            right, bottom = (
                min(width, int(np.ceil(x2 + 0.12 * pw))),
                min(height, int(np.ceil(y2 + 0.10 * ph))),
            )
            crop_w, crop_h = right - left, bottom - top
            target = []
            for cls, cx, cy, bw, bh in row["positive_boxes"]:
                bx1, by1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                bx2, by2 = (cx + bw / 2) * width, (cy + bh / 2) * height
                l, t, r, b = (
                    max(left, bx1),
                    max(top, by1),
                    min(right, bx2),
                    min(bottom, by2),
                )
                if r - l < 2 or b - t < 2:
                    continue
                target.append(
                    [
                        cls,
                        ((l + r) / 2 - left) / crop_w,
                        ((t + b) / 2 - top) / crop_h,
                        (r - l) / crop_w,
                        (b - t) / crop_h,
                    ]
                )
            name = f"{row['project_id']}_{row['image_id']}_p{index:02d}"
            path = images / (name + ".jpg")
            assert cv2.imwrite(
                str(path), im[top:bottom, left:right], [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            label = labels / (name + ".txt")
            label.write_text(
                "".join(
                    f"{int(v[0])} " + " ".join(f"{x:.10f}" for x in v[1:]) + "\n"
                    for v in target
                )
            )
            crops.append(
                {
                    "image": str(path),
                    "label": str(label),
                    "group": row["group"],
                    "source_key": [row["project_id"], row["image_id"]],
                    "source_image": row["copied_image"],
                    "native_crop_xyxy": [left, top, right, bottom],
                    "classes": [int(v[0]) for v in target],
                }
            )
        processed += 1
        if processed % 100 == 0:
            print(
                json.dumps(
                    {"sources_processed": processed, "person_crops": len(crops)}
                ),
                flush=True,
            )
    assert crops, "No person crops"
    groups = sorted({r["group"] for r in crops})
    index = {g: i for i, g in enumerate(groups)}
    matrix = np.zeros((len(groups), len(CLASSES)), dtype=np.int64)
    sizes = np.zeros(len(groups), dtype=np.int64)
    for row in crops:
        i = index[row["group"]]
        sizes[i] += 1
        for cls in row["classes"]:
            matrix[i, cls] += 1
    support = (matrix > 0).sum(0)
    assert (support > 0).all(), (
        f"Missing classes in person crops: {[CLASSES[i] for i in np.flatnonzero(support == 0)]}"
    )
    rng = np.random.default_rng(20261109)
    best, best_score = None, float("inf")
    for _ in range(20000):
        assignment = rng.choice(3, len(groups), p=[0.8, 0.1, 0.1])
        counts = np.array([matrix[assignment == s].sum(0) for s in range(3)])
        split_sizes = np.array([sizes[assignment == s].sum() for s in range(3)])
        missing = (counts[0] == 0).sum() + (counts[1:, support >= 3] == 0).sum()
        score = (
            1000 * missing + np.abs(split_sizes / len(crops) - [0.8, 0.1, 0.1]).sum()
        )
        if score < best_score:
            best, best_score = assignment.copy(), score
        if missing == 0 and score < 0.04:
            break
    assignments = dict(
        zip(groups, ["train" if s == 0 else "val" if s == 1 else "test" for s in best])
    )
    stats = {}
    for split in ["train", "val", "test"]:
        subset = [r for r in crops if assignments[r["group"]] == split]
        assert subset
        (OUT / f"{split}.txt").write_text("".join(r["image"] + "\n" for r in subset))
        counts = Counter(c for r in subset for c in r["classes"])
        stats[split] = {
            "images": len(subset),
            "groups": len({r["group"] for r in subset}),
            "boxes_by_class": {c: counts[i] for i, c in enumerate(CLASSES)},
        }
        for row in subset:
            row["split"] = split
    assert all(stats["train"]["boxes_by_class"][c] > 0 for c in CLASSES)
    group_sets = [
        {r["group"] for r in crops if r["split"] == split}
        for split in ["train", "val", "test"]
    ]
    assert not (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    )
    (OUT / "crop-manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in crops)
    )
    yaml.safe_dump(
        {
            "path": str(OUT),
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
            "nc": 11,
            "names": dict(enumerate(CLASSES)),
        },
        (OUT / "data.yaml").open("w"),
        sort_keys=False,
    )
    summary = {
        "selected_source_images": len(rows),
        "person_crops": len(crops),
        "excluded_at_crop_stage": dict(exclusions),
        "splits": stats,
        "group_leakage": False,
        "grouping": "Known original/augmentation families plus conservative 63-bit perceptual-hash near duplicates (distance<=4 and aspect ratio within5%).",
        "class_group_support": dict(zip(CLASSES, map(int, support))),
        "all_11_classes_in_train": True,
        "split_seed": 20261109,
        "test_not_used_for_model_selection": True,
    }
    (OUT / "release-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
