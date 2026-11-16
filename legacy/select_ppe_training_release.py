"""Copy only currently hash-matching, explicitly complete PPE annotation reviews."""

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from legacy.apply_verified_ppe_labels import REPORTS, validate_labels

ROOT = REPORTS.parent.parent
OUT = ROOT / "data/ppe-verified-20261109"
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
COMPLETE = {
    "complete",
    "complete_for_visible_targets",
    "complete_visible_targets",
    "complete_for_visible_targets_and_negative_boxes",
    "complete_for_visible_targets_and_status_boxes",
    "complete_visible_boxes",
    "complete_visible_labels",
    "complete_for_visible_target_objects",
    "complete_for_visible_target_target_objects",
    "complete_reviewed",
    "complete_reviewed_product_only",
    "complete_reviewed_stock_portrait",
}


def read_jsonl(path):
    return (
        [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if path.exists()
        else []
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = {
        (str(r["project_id"]), str(r["image_id"])): r
        for r in read_jsonl(REPORTS / "draft-manifest.jsonl")
    }
    selected, exclusions = {}, Counter()
    for shard in range(1, 6):
        directory = REPORTS / f"luna-full-audit/shard-{shard}"
        reviews = read_jsonl(directory / "reviews.jsonl")
        post = {
            (str(x.get("project_id", "85")), str(x["image_id"])): x
            for x in read_jsonl(directory / "postproposal-verification.jsonl")
        }
        release = {
            (str(x["project_id"]), str(x["image_id"])): x
            for x in read_jsonl(directory / "release-review.jsonl")
        }
        for review in reviews:
            key = str(review["project_id"]), str(review["image_id"])
            source = "cap-pavithrap" if key[0] == "81" else "safety-working"
            image = (
                Path("/opt/mount/project/data")
                / source
                / "images"
                / review["file_name"]
            )
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            current = sha(label)
            completeness = str(
                review.get("label_completeness", review.get("completeness", ""))
            )
            reason = None
            if (
                completeness in COMPLETE
                or completeness.startswith("complete: visible positive target PPE")
            ) and current == review.get("label_sha256"):
                reason = "explicit_complete_original_review_current_hash"
            rel = release.get(key)
            if (
                rel
                and rel["status"] == "label_complete"
                and current == rel["current_label_sha256"]
            ):
                reason = "complete_current_source_release_review"
            verification = post.get(key)
            if verification:
                visual = verification.get("postproposal_visual_review", {})
                correct = visual.get(
                    "eligible_after_apply",
                    verification.get("eligible_after_apply", False),
                )
                if shard == 5:
                    correct = (
                        verification.get("whole_image_labels_correct_after_apply")
                        is True
                    )
                final_hash = verification.get(
                    "final_label_sha256",
                    verification.get(
                        "proposed_final_label_sha256",
                        verification.get("proposed_label_sha256"),
                    ),
                )
                actual_view = (
                    verification.get("verified") is True
                    or verification.get("actual_view_performed") is True
                    or verification.get("visual_verified_by_full_overlay") is True
                    or verification.get("evidence_status")
                    == "visually_verified_final_label_bytes"
                )
                if (
                    correct
                    and actual_view
                    and current == final_hash
                    and not visual.get("visible_errors")
                    and not visual.get("remaining_unresolved_findings")
                ):
                    reason = "final_overlay_complete_and_applied_hash_matches"
                elif not correct:
                    reason = None
            if (
                rel
                and rel["status"] in ("needs_repair", "quarantine")
                and current == rel["current_label_sha256"]
            ):
                reason = None
            if key in {("85", "4"), ("85", "1971"), ("85", "5188")}:
                reason = None
            if not reason:
                exclusions[
                    "unfixed_unresolved_or_no_explicit_current_complete_review"
                ] += 1
                continue
            if sha(image) != review.get("image_sha256"):
                exclusions["source_image_changed"] += 1
                continue
            names = yaml.safe_load((image.parent.parent / "data.yaml").read_text())[
                "names"
            ]
            names = dict(enumerate(names)) if isinstance(names, list) else names
            try:
                validate_labels(label.read_bytes(), set(map(int, names)))
            except ValueError:
                exclusions["invalid_label_geometry"] += 1
                continue
            boxes = []
            for line in label.read_text().splitlines():
                cls, *values = line.split()
                name = names[int(cls)]
                if name in CLASSES:
                    boxes.append([CLASSES.index(name), *map(float, values)])
            row = manifest[key]
            selected[key] = {
                "project_id": key[0],
                "image_id": key[1],
                "dataset": source,
                "source_image": str(image),
                "source_label": str(label),
                "image_sha256": review["image_sha256"],
                "label_sha256": current,
                "file_name": image.name,
                "group": row["group"],
                "selection_reason": reason,
                "positive_boxes": boxes,
            }
    assert not OUT.exists(), f"Release already exists: {OUT}"
    OUT.mkdir(parents=True)
    counts, image_counts = Counter(), Counter()
    for row in selected.values():
        dest = OUT / "source" / row["dataset"]
        (dest / "images").mkdir(parents=True, exist_ok=True)
        (dest / "labels").mkdir(parents=True, exist_ok=True)
        copied_image = dest / "images" / row["file_name"]
        copied_label = dest / "labels" / (Path(row["file_name"]).stem + ".txt")
        shutil.copy2(row["source_image"], copied_image)
        shutil.copy2(row["source_label"], copied_label)
        assert (
            sha(copied_image) == row["image_sha256"]
            and sha(copied_label) == row["label_sha256"]
        )
        row["copied_image"], row["copied_source_label"] = (
            str(copied_image),
            str(copied_label),
        )
        present = set()
        for box in row["positive_boxes"]:
            counts[CLASSES[box[0]]] += 1
            present.add(CLASSES[box[0]])
        image_counts.update(present)
    (OUT / "selected-source-manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in selected.values())
    )
    summary = {
        "selected_source_images": len(selected),
        "excluded_source_images": 10030 - len(selected),
        "exclusions": dict(exclusions),
        "classes": CLASSES,
        "per_class": {
            c: {"boxes": counts[c], "images": image_counts[c]} for c in CLASSES
        },
        "source_copies_are_byte_verified": True,
        "cropping_and_splits_pending": True,
    }
    (OUT / "selection-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
