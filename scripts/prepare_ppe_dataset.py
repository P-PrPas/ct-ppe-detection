"""Audit Andromeda COCO snapshots and prepare a draft; never edit source labels.

Usage: python3 scripts/prepare_ppe_dataset.py
The draft is deliberately gated until central annotation edits and a visual
completeness review have been verified. It is not a training release.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ppe-preparation-2026-11-08"
NAMES = list(yaml.safe_load((ROOT / "dataset/classes.yaml").read_text())["names"].values())[:11]
SOURCES = {85: "safety-working", 81: "cap-pavithrap"}


def family(project_id, name):
    return f"{project_id}:" + re.sub(r"\.rf\.[^.]+", "", name)


def split_for(group):
    # Fixed assignment by source family, independent of image enumeration.
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def representative_candidates(records):
    groups = collections.defaultdict(list)
    for record in records:
        if not record["baseline_exclusion"]:
            groups[record["group"]].append(record)
    # Resolution breaks ties; it is not a semantic quality score.
    chosen = [sorted(rows, key=lambda r: (-r["width"] * r["height"], r["file_name"]))[0]
              for rows in groups.values()]
    counts = collections.Counter(b["class"] for r in chosen for b in r["positive_boxes"])
    for record in chosen:
        record["review_priority"] = round(sum(1 / counts[n] for n in
                                             {b["class"] for b in record["positive_boxes"]}), 8)
    return sorted(chosen, key=lambda r: (-r["review_priority"], r["group"])), counts


def self_check():
    assert len(NAMES) == 11 and len(set(NAMES)) == 11
    assert family(85, "x_jpg.rf.abc.jpg") == family(85, "x_jpg.rf.def.jpg")
    assert family(85, "x.jpg") != family(81, "x.jpg")
    assert split_for("a") == split_for("a")
    assert {split_for(str(i)) for i in range(1000)} == {"train", "val", "test"}


def main():
    self_check()
    edit_result = OUT / "pool-edit-result.json"
    if edit_result.exists() and not json.loads(edit_result.read_text())["andromeda_auto_updated"]:
        raise SystemExit("Shared pool labels were edited after these Andromeda snapshots. "
                         "Refresh the imports/snapshots and sync report before rebuilding; "
                         "refusing to overwrite the draft with stale labels.")
    records, invalid, count_all = [], [], collections.Counter()
    snapshot_hashes = {}
    screening = {}
    excluded_families = {}
    for name in ("cap-screening.json", "core-screening.json"):
        for item in json.loads((OUT / name).read_text()):
            screening[item["project_id"], item["image_id"]] = item["screening"]
            if item["screening"].startswith(("exclude_", "needs_")):
                excluded_families[family(item["project_id"], item["file_name"])] = item["screening"]
    for pid, source in SOURCES.items():
        snap = OUT / "snapshots" / f"project-{pid}.coco.json"
        snapshot_hashes[pid] = hashlib.sha256(snap.read_bytes()).hexdigest()
        data = json.loads(snap.read_text())
        images = {x["id"]: x for x in data["images"]}
        categories = {x["id"]: x["name"] for x in data["categories"]}
        assert len(images) == len(data["images"]), "duplicate image IDs"
        by_image = collections.defaultdict(list)
        for a in data["annotations"]:
            assert a["image_id"] in images and a["category_id"] in categories
            z = images[a["image_id"]]
            x, y, w, h = a["bbox"]
            if (not all(math.isfinite(v) for v in (x, y, w, h)) or
                    min(w, h) <= 0 or min(x, y) < 0 or
                    x + w > z["width"] + 1e-6 or y + h > z["height"] + 1e-6):
                invalid.append({"project_id": pid, "annotation": a})
            count_all[categories[a["category_id"]]] += 1
            by_image[a["image_id"]].append(a)
        for number, z in enumerate(data["images"]):
            name = z["file_name"]
            assert Path(name).name == name, "unexpected image path; resolve explicitly"
            path = ROOT / "dataset" / source / "images" / name
            blob = path.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            with Image.open(path) as image:
                assert image.size == (z["width"], z["height"]), f"dimension mismatch: {path}"
                # Decode pixels to detect a truncated or corrupt local mirror.
                image.load()
            boxes = [{"class": categories[a["category_id"]], "bbox_xywh": a["bbox"],
                      "source_annotation_id": a["id"]}
                     for a in by_image[z["id"]] if categories[a["category_id"]] in NAMES]
            records.append({"project_id": pid, "image_id": z["id"], "file_name": name,
                            "local_mirror": str(path), "pool_path": f"/opt/mount/project/data/{source}/images/{name}",
                            "width": z["width"], "height": z["height"], "sha256": digest,
                            "family": family(pid, name), "positive_boxes": boxes,
                            "screening": screening.get((pid, z["id"]), "not_visually_screened"),
                            "baseline_exclusion": excluded_families.get(family(pid, name)),
                            "semantic_review": "pending", "central_edits_verified": False})
            if (number + 1) % 1000 == 0:
                print(f"#{pid}: decoded and hashed {number + 1} images", flush=True)

    assert not invalid, f"invalid source boxes: {len(invalid)}; do not prepare a draft"
    # Connect filename variants and byte-identical files across all sources.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    first_hash = {}
    for r in records:
        find(r["family"])
        union(r["family"], first_hash.setdefault(r["sha256"], r["family"]))
    cap_images = {r["image_id"]: r for r in records if r["project_id"] == 81}
    for group in json.loads((OUT / "manual-cap-groups.json").read_text()):
        for image_id in group[1:]:
            union(cap_images[group[0]]["family"], cap_images[image_id]["family"])
    seen, duplicates, conflicts = {}, [], []
    for r in records:
        r["group"] = find(r["family"])
        r["provisional_split"] = split_for(r["group"])
        if r["sha256"] in seen:
            other = seen[r["sha256"]]
            r["byte_duplicate_of"] = f"{other['project_id']}:{other['image_id']}"
            duplicates.append({"image": f"{r['project_id']}:{r['image_id']}", "duplicate_of": r["byte_duplicate_of"]})
            signature = lambda v: sorted((b["class"], tuple(b["bbox_xywh"])) for b in v)
            if signature(r["positive_boxes"]) != signature(other["positive_boxes"]):
                conflicts.append({"image": f"{r['project_id']}:{r['image_id']}", "other": r["byte_duplicate_of"]})
        else:
            seen[r["sha256"]] = r
    groups = collections.defaultdict(set)
    for r in records:
        groups[r["group"]].add(r["provisional_split"])
    assert all(len(v) == 1 for v in groups.values()), "group leakage"
    class_splits = {s: collections.Counter() for s in ("train", "val", "test")}
    candidates = [r for r in seen.values() if not r["baseline_exclusion"]]
    for r in candidates:
        class_splits[r["provisional_split"]].update(b["class"] for b in r["positive_boxes"])
    assert all(class_splits[s][n] for s in class_splits for n in NAMES), "missing class in a provisional split"
    result = {"status": "DRAFT_NOT_TRAINING_READY", "classes": NAMES,
              "source_snapshot_sha256": snapshot_hashes,
              "source_images": len(records), "source_annotations": sum(count_all.values()),
              "source_class_boxes": dict(count_all), "byte_unique_images": len(seen),
              "byte_duplicate_images": len(duplicates), "duplicate_label_conflicts": len(conflicts),
              "source_family_groups": len(groups), "invalid_boxes": len(invalid),
              "baseline_candidate_images": len(candidates),
              "baseline_excluded_images": len(seen) - len(candidates),
              "visual_screening_counts": dict(collections.Counter(r["screening"] for r in records)),
              "provisional_unique_images_per_split": dict(collections.Counter(r["provisional_split"] for r in seen.values())),
              "provisional_positive_boxes_per_split": {k: dict(v) for k, v in class_splits.items()},
              "release_gates": ["Verify Andromeda edit access and central annotation revisions",
                                "Resolve byte-identical images with conflicting annotations",
                                "Review selected person crops for all 11 classes, including unannotated positives",
                                "Verify near-duplicate/scene groups beyond filename and exact SHA256",
                                "Create person crops and recheck split membership before release"],
              "central_source_labels_modified": False}
    representatives, representative_counts = representative_candidates(records)
    assert len({r["group"] for r in representatives}) == len(representatives)
    for split in class_splits:
        coverage = {b["class"] for r in representatives if r["provisional_split"] == split
                    for b in r["positive_boxes"]}
        assert coverage == set(NAMES), "missing class among provisional representatives"
    result.update(representative_candidate_images=len(representatives),
                  representative_class_boxes=dict(representative_counts),
                  candidate_splits_are_provisional=True,
                  representative_selection="One per known source/confirmed duplicate group; prefer larger image dimensions. All remain pending complete semantic review.")
    for filename, value in [("summary.json", result), ("exact-duplicates.json", duplicates),
                            ("duplicate-label-conflicts.json", conflicts)]:
        (OUT / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    with (OUT / "draft-manifest.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "draft-splits.csv").open("w") as f:
        fields = ["project_id", "image_id", "file_name", "group", "provisional_split", "sha256", "byte_duplicate_of", "screening", "baseline_exclusion"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (OUT / "baseline-candidates.jsonl").open("w") as f:
        for record in representatives:
            f.write(json.dumps(record) + "\n")
    with (OUT / "baseline-review-queue.csv").open("w") as f:
        fields = ["project_id", "image_id", "file_name", "pool_path", "local_mirror",
                  "group", "provisional_split", "screening", "review_priority"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(representatives)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
