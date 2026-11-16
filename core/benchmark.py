"""Validate predictions and reconcile Andromeda submissions before retrying."""

import json
import math
import subprocess
import time
from pathlib import Path

from core.training import write_json
from core.validation import digest, require


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


def valid_box(box):
    return (
        len(box) == 4
        and all(math.isfinite(v) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def predict_submit(config, torch, yolo):
    out, exam, weights = (
        Path(config["out"]),
        Path(config["exam"]),
        Path(config["weights"]),
    )
    project, definition = config["project"], config["definition"]
    expected_images, device, imgsz = (
        config["expected_images"],
        config["device"],
        config["imgsz"],
    )
    candidate, version, export_conf = config["name"], config["version"], 0.05
    out.mkdir(parents=True, exist_ok=True)

    def state(status, **details):
        write_json(
            out / "state.json", dict(status=status, updated_at=time.time(), **details)
        )

    existing = find_run(project, definition, digest(weights))
    if existing:
        state("submitted", server_run=existing["number"])
        return
    if any(
        (out / name).exists()
        for name in ("submission-result.json", "submission-pending.json")
    ):
        raise RuntimeError("Receipt without server confirmation; inspect before retry")
    d = json.loads((exam / "members.json").read_text())
    members = d["members"]
    require(
        str(d["definition"]["id"]) == str(definition),
        "Exam definition differs from submission target",
    )
    require(
        len(members) == expected_images,
        "Integrity check failed: len(members) == EXPECTED",
    )
    require(
        d["definition"]["roles"] == ["test"]
        and d["definition"]["min_conf"] == 0.5
        and d["definition"]["iou"] == 0.5,
        "Integrity check failed: d['definition']['roles'] == ['test'] and d['definition']['min_conf'] == 0.5 and (d['definition']['iou'] == 0.5)",
    )
    require(
        json.loads((out / "preparation.json").read_text())["images_sha256_verified"],
        "Integrity check failed: json.loads((OUT / 'preparation.json').read_text())['images_sha256_verified']",
    )
    paths = [str(exam / "images" / (m["sample_id"] + m["ext"])) for m in members]
    require(
        all(Path(p).is_file() for p in paths),
        "Integrity check failed: all((Path(p).is_file() for p in paths))",
    )
    for member, path in zip(members, paths, strict=True):
        require(digest(path) == member["sha256"], f"Exam image changed: {path}")
    torch.set_num_threads(2)
    model = yolo(str(weights))
    allowed = {c["name"] for c in d["classes"]}
    require(
        set(model.names.values()).issubset(allowed),
        "Integrity check failed: set(model.names.values()).issubset(allowed)",
    )
    config = dict(
        model=f"{candidate}; {version}",
        weights_sha256=digest(weights),
        imgsz=imgsz,
        batch=4,
        confidence=export_conf,
        nms_iou=0.7,
        max_det=300,
        benchmark_manifest=d["manifest_sha256"],
        scorer=d["definition"]["scorer"],
        matching_iou=0.5,
        benchmark_min_conf=0.5,
        model_classes=list(model.names.values()),
        benchmark_classes=sorted(allowed),
        overlap=json.loads((out / "preparation.json").read_text()).get(
            "exact_image_overlap_with_training_release",
            json.loads((out / "preparation.json").read_text()).get(
                "exact_image_overlap_with_first_release", {}
            ),
        ),
        device=device,
        note="Export conf0.05; server scores at its frozen min_conf0.5. No threshold tuning. Overlap documented in preparation.json. Inference resolution recorded above; frozen exam unchanged.",
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model.add_callback("on_predict_start", lambda predictor: torch.set_num_threads(2))
    predictions = {}
    state("predicting", processed=0, total=len(paths))
    start = time.monotonic()
    by_id = {m["sample_id"]: m for m in members}
    results = model.predict(
        str(exam / "images"),
        device=device,
        imgsz=imgsz,
        batch=4,
        conf=export_conf,
        iou=0.7,
        max_det=300,
        stream=True,
        verbose=False,
    )
    for idx, result in enumerate(results):
        sample_id = Path(result.path).stem
        require(
            sample_id in by_id and sample_id not in predictions,
            "Integrity check failed: sample_id in by_id and sample_id not in predictions",
        )
        m = by_id[sample_id]
        boxes = []
        for xyxy, conf, cls in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            strict=True,
        ):
            if valid_box(xyxy):
                boxes.append(dict(cls=model.names[int(cls)], box=xyxy, conf=conf))
        predictions[m["sample_id"]] = boxes
        if (idx + 1) % 200 == 0:
            state("predicting", processed=idx + 1, total=len(paths))
            print(
                f"{idx + 1}/{len(paths)} images; elapsed {time.monotonic() - start:.1f}s",
                flush=True,
            )
    validate_predictions(predictions, members, allowed)
    path = out / "predictions.json"
    path.write_text(json.dumps(predictions, separators=(",", ":")))
    require(
        path.stat().st_size < 64 * 1024 * 1024,
        "Integrity check failed: path.stat().st_size < 64 * 1024 * 1024",
    )
    state(
        "submitting", images=len(predictions), boxes=sum(map(len, predictions.values()))
    )
    server = submit_confirmed(
        project,
        definition,
        out,
        weights,
        candidate,
        version,
        receipt_name="submission-result.json",
    )
    state("submitted", server_run=server["number"])
    print(json.dumps(server), flush=True)
