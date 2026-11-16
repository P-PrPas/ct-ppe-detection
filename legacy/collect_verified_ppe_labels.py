"""Freeze reviewed label bytes into a repair plan; never writes shared sources."""

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from legacy.apply_verified_ppe_labels import REPORTS, digest, validate_labels


def first(row, *keys):
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    raise ValueError("Missing required field: " + "/".join(keys))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def resolve_reference(value, directory):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "reports":
        return REPORTS.parent.parent / path
    return directory / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    args = parser.parse_args()
    require(
        args.batch_id.replace("-", "").replace("_", "").isalnum(), "Invalid batch id"
    )
    root = REPORTS / "luna-full-audit"
    frozen = root / ("parent-frozen-" + args.batch_id)
    frozen.mkdir(exist_ok=False)
    latest_applied = {}
    for journal in sorted((REPORTS / "pool-edit-backups").glob("*/journal.json")):
        rows = json.loads(journal.read_text())
        if isinstance(rows, list):
            for row in rows:
                if row.get("status") == "applied_readback_verified" and row.get(
                    "source_label_path"
                ):
                    latest_applied[row["source_label_path"]] = row[
                        "proposed_final_label_sha256"
                    ]
    plans, held = [], []
    for shard in range(1, 6):
        directory = root / f"shard-{shard}"
        ledger = directory / "postproposal-verification.jsonl"
        if not ledger.exists():
            continue
        reviews = {
            str(row["image_id"]): row
            for row in map(
                json.loads, (directory / "reviews.jsonl").read_text().splitlines()
            )
        }
        for row in map(json.loads, ledger.read_text().splitlines()):
            visual = row.get("postproposal_visual_review", {})
            eligible = visual.get(
                "eligible_after_apply", row.get("eligible_after_apply", False)
            )
            accepted = (
                row.get("changed_actions_correct")
                if shard == 3
                else (
                    row.get("whole_image_labels_correct_after_apply")
                    if shard == 5
                    else eligible
                )
            )
            if accepted is not True:
                continue
            image_id = str(row["image_id"])
            try:
                source_value = row.get(
                    "source_label_path", row.get("current_source_label_path")
                )
                if source_value is None:
                    dataset = (
                        "cap-pavithrap"
                        if str(row.get("project_id")) == "81"
                        else "safety-working"
                    )
                    source_value = str(
                        Path("/opt/mount/project/data")
                        / dataset
                        / "labels"
                        / (Path(reviews[image_id]["file_name"]).stem + ".txt")
                    )
                source = Path(source_value)
                current = source.read_bytes()
                if latest_applied.get(str(source)) == digest(current):
                    # A fresh review can still propose another repair to an applied image.
                    final_hint = row.get(
                        "final_label_sha256",
                        row.get(
                            "proposed_final_label_sha256",
                            row.get("proposed_label_sha256"),
                        ),
                    )
                    if final_hint == digest(current):
                        continue
                if shard == 1:
                    require(
                        row.get("evidence_status")
                        == "visually_verified_final_label_bytes",
                        "Final bytes not visually verified",
                    )
                    require(
                        not visual.get("visible_errors")
                        and not visual.get("remaining_unresolved_findings"),
                        "Unresolved final image",
                    )
                    data = row["final_label_bytes_utf8"].encode()
                else:
                    require(
                        row.get("verified") is True
                        or row.get("actual_view_performed") is True
                        or row.get("visual_verified_by_full_overlay") is True,
                        "Missing actual view",
                    )
                    proposal = resolve_reference(
                        first(row, "proposed_final_label_path", "proposed_label_path"),
                        directory,
                    )
                    data = proposal.read_bytes()
                    if shard == 5:
                        require(
                            row.get(
                                "checked_existing_boxes_missing_targets_and_contradictions"
                            )
                            is True,
                            "Incomplete image review",
                        )
                final_hash = first(
                    row,
                    "final_label_sha256",
                    "proposed_final_label_sha256",
                    "proposed_label_sha256",
                )
                require(digest(data) == final_hash, "Proposed final bytes changed")
                if data == current:
                    continue
                old_hash = first(row, "source_label_sha256", "label_sha256")
                require(digest(current) == old_hash, "Source hash changed")
                evidence = resolve_reference(
                    first(
                        row,
                        "overlay_path",
                        "evidence_overlay_path",
                        "current_vs_proposed_overlay",
                        "actual_final_overlay_path",
                        "final_overlay_path",
                    ),
                    directory,
                )
                require(evidence.is_file(), "Missing overlay evidence")
                evidence_hash = row.get(
                    "actual_overlay_sha256", row.get("final_overlay_sha256")
                )
                if evidence_hash:
                    require(
                        digest(evidence.read_bytes()) == evidence_hash,
                        "Reviewed overlay changed",
                    )
                names = yaml.safe_load(
                    (source.parent.parent / "data.yaml").read_text()
                )["names"]
                classes = (
                    set(range(len(names)))
                    if isinstance(names, list)
                    else set(map(int, names))
                )
                validate_labels(data, classes)
                target = frozen / f"{shard}_{image_id}.txt"
                target.write_bytes(data)
                plans.append(
                    {
                        "project_id": str(row.get("project_id", "85")),
                        "shard": shard,
                        "image_id": image_id,
                        "source_label_path": str(source),
                        "source_label_sha256": old_hash,
                        "proposed_label_path": str(target),
                        "proposed_final_label_sha256": final_hash,
                        "parent_visual_verified": True,
                        "parent_review_basis": "Accepted delegated actual final-overlay verification with frozen bytes, source/final hashes, bounds and evidence checked. Source repair does not imply whole-image training approval.",
                        "parent_evidence": str(evidence),
                        "delegate_verification": row,
                    }
                )
            except (ValueError, KeyError, FileNotFoundError) as error:
                held.append(
                    {"shard": shard, "image_id": image_id, "reason": str(error)}
                )
    require(
        len({p["source_label_path"] for p in plans}) == len(plans),
        "Duplicate source targets",
    )
    plan_path = root / f"parent-ready-{args.batch_id}.json"
    plan_path.write_text(json.dumps(plans, indent=2))
    (root / f"parent-held-{args.batch_id}.json").write_text(json.dumps(held, indent=2))
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "files": len(plans),
                "by_shard": dict(Counter(p["shard"] for p in plans)),
                "held_count": len(held),
            }
        )
    )


if __name__ == "__main__":
    main()
