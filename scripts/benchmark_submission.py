"""Validate predictions and reconcile Andromeda submissions before retrying."""

import json
import math
import subprocess
from pathlib import Path

from release_validation import digest, require


def find_run(project, definition, weights_hash):
    result = subprocess.run(
        ["andromeda", "detection", "runs", str(project), str(definition), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    found = [
        run
        for run in json.loads(result.stdout)["runs"]
        if run["candidate"].get("weights_sha256") == weights_hash
    ]
    require(len(found) <= 1, "Duplicate server submissions need inspection")
    return found[0] if found else None


def validate_predictions(predictions, members, classes=None):
    require(
        set(predictions) == {member["sample_id"] for member in members},
        "Prediction sample coverage differs from exam",
    )
    for boxes in predictions.values():
        for prediction in boxes:
            box = prediction["box"]
            require(
                len(box) == 4
                and all(math.isfinite(value) for value in box)
                and box[2] > box[0]
                and box[3] > box[1],
                "Invalid prediction box",
            )
            confidence = prediction["conf"]
            require(
                math.isfinite(confidence) and 0 <= confidence <= 1,
                "Invalid prediction confidence",
            )
            if classes is not None:
                require(prediction["cls"] in classes, "Unmapped prediction class")


def submit_confirmed(
    project,
    definition,
    directory,
    weights,
    name,
    version,
    *,
    drop_unmapped=False,
    receipt_name="receipt.json",
):
    """A local receipt without a server match is ambiguous: never submit again."""
    weights = Path(weights)
    weights_hash = digest(weights)
    existing = find_run(project, definition, weights_hash)
    if existing is None:
        receipts = ("receipt.json", "submission-result.json", "submission-pending.json")
        require(
            not any((directory / receipt).exists() for receipt in receipts),
            "Receipt or pending submission without server confirmation; inspect before retry",
        )
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        require(
            config.get("weights_sha256", config.get("sha256")) == weights_hash,
            "Checkpoint differs from prediction config",
        )
        command = [
            "andromeda",
            "submit",
            str(project),
            str(definition),
            "--predictions",
            str(directory / "predictions.json"),
            "--name",
            name,
            "--version",
            version,
            "--weights",
            str(weights),
            "--config",
            str(config_path),
        ]
        if drop_unmapped:
            command.append("--drop-unmapped")
        # Keep an intent record if the process dies after the server accepts the upload.
        (directory / "submission-pending.json").write_text(
            json.dumps(dict(weights_sha256=weights_hash))
        )
        result = subprocess.run(command, capture_output=True, text=True)
        (directory / "submit.stdout").write_text(result.stdout)
        (directory / "submit.stderr").write_text(result.stderr)
        require(
            result.returncode == 0,
            "CLI submission failed; inspect saved stdout/stderr before retry",
        )
        (directory / receipt_name).write_text(
            json.dumps(
                dict(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                ),
                indent=2,
            )
            + "\n"
        )
        existing = find_run(project, definition, weights_hash)
        require(
            existing is not None, "CLI returned success without server confirmation"
        )
    (directory / "confirmed-server-run.json").write_text(
        json.dumps(existing, indent=2) + "\n"
    )
    return existing
