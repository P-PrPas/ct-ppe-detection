"""Run the first full-current model on an official Andromeda snapshot and submit once."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

from benchmark_submission import find_run, submit_confirmed, validate_predictions
from release_validation import require

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
PROJECT = 98
DEFINITION = "1"
EXPECTED = 5931
DEVICE = "cpu"
EXPORT_CONF = 0.05
IMGSZ = 640
CANDIDATE = "yolo11s-full-current-20261109"
VERSION = "best-epoch68"
OUT = ROOT / "reports/andromeda-benchmark-test-only"
EXAM = OUT / "exam"
WEIGHTS = ROOT / "runs/ppe-full-current-20261109/weights/best.pt"


def valid_box(box):
    return (
        len(box) == 4
        and all(math.isfinite(v) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def state(status, **kw):
    (OUT / "state.json").write_text(
        json.dumps(
            dict(
                status=status,
                updated_at=time.time(),
                project_id=PROJECT,
                definition_id=DEFINITION,
                model=str(WEIGHTS),
                **kw,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    existing = find_run(
        PROJECT, DEFINITION, hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    )
    if existing:
        state("submitted", server_run=existing["number"])
        return
    if any(
        (OUT / name).exists()
        for name in ("submission-result.json", "submission-pending.json")
    ):
        raise RuntimeError("Receipt without server confirmation; inspect before retry")
    d = json.loads((EXAM / "members.json").read_text())
    members = d["members"]
    require(
        str(d["definition"]["id"]) == str(DEFINITION),
        "Exam definition differs from submission target",
    )
    require(
        len(members) == EXPECTED, "Integrity check failed: len(members) == EXPECTED"
    )
    require(
        d["definition"]["roles"] == ["test"]
        and d["definition"]["min_conf"] == 0.5
        and d["definition"]["iou"] == 0.5,
        "Integrity check failed: d['definition']['roles'] == ['test'] and d['definition']['min_conf'] == 0.5 and (d['definition']['iou'] == 0.5)",
    )
    require(
        json.loads((OUT / "preparation.json").read_text())["images_sha256_verified"],
        "Integrity check failed: json.loads((OUT / 'preparation.json').read_text())['images_sha256_verified']",
    )
    paths = [str(EXAM / "images" / (m["sample_id"] + m["ext"])) for m in members]
    require(
        all(Path(p).is_file() for p in paths),
        "Integrity check failed: all((Path(p).is_file() for p in paths))",
    )
    torch.set_num_threads(2)
    model = YOLO(str(WEIGHTS))
    allowed = {c["name"] for c in d["classes"]}
    require(
        set(model.names.values()).issubset(allowed),
        "Integrity check failed: set(model.names.values()).issubset(allowed)",
    )
    config = dict(
        model=f"{CANDIDATE}; {VERSION}",
        weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        imgsz=IMGSZ,
        batch=4,
        confidence=EXPORT_CONF,
        nms_iou=0.7,
        max_det=300,
        benchmark_manifest=d["manifest_sha256"],
        scorer=d["definition"]["scorer"],
        matching_iou=0.5,
        benchmark_min_conf=0.5,
        model_classes=list(model.names.values()),
        benchmark_classes=sorted(allowed),
        overlap=json.loads((OUT / "preparation.json").read_text()).get(
            "exact_image_overlap_with_training_release",
            json.loads((OUT / "preparation.json").read_text()).get(
                "exact_image_overlap_with_first_release", {}
            ),
        ),
        device=DEVICE,
        note="Export conf0.05; server scores at its frozen min_conf0.5. No threshold tuning. Overlap documented in preparation.json. Inference resolution recorded above; frozen exam unchanged.",
    )
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model.add_callback("on_predict_start", lambda predictor: torch.set_num_threads(2))
    predictions = {}
    state("predicting", processed=0, total=len(paths))
    start = time.monotonic()
    by_id = {m["sample_id"]: m for m in members}
    results = model.predict(
        str(EXAM / "images"),
        device=DEVICE,
        imgsz=IMGSZ,
        batch=4,
        conf=EXPORT_CONF,
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
    path = OUT / "predictions.json"
    path.write_text(json.dumps(predictions, separators=(",", ":")))
    require(
        path.stat().st_size < 64 * 1024 * 1024,
        "Integrity check failed: path.stat().st_size < 64 * 1024 * 1024",
    )
    state(
        "submitting", images=len(predictions), boxes=sum(map(len, predictions.values()))
    )
    server = submit_confirmed(
        PROJECT,
        DEFINITION,
        OUT,
        WEIGHTS,
        CANDIDATE,
        VERSION,
        receipt_name="submission-result.json",
    )
    state("submitted", server_run=server["number"])
    print(json.dumps(server), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=int, default=98)
    parser.add_argument("--definition", default="1")
    parser.add_argument("--expected-images", type=int, default=5931)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--name", default=CANDIDATE)
    parser.add_argument("--version", default=VERSION)
    args = parser.parse_args()
    PROJECT = args.project
    DEFINITION = args.definition
    EXPECTED = args.expected_images
    OUT = args.out.resolve()
    EXAM = OUT / "exam"
    DEVICE = args.device
    WEIGHTS = args.weights.resolve()
    CANDIDATE = args.name
    VERSION = args.version
    IMGSZ = args.imgsz
    require(
        IMGSZ > 0 and IMGSZ % 32 == 0,
        "Integrity check failed: IMGSZ > 0 and IMGSZ % 32 == 0",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except BaseException as e:
        state("failed", error=repr(e))
        raise
