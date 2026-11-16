from core.benchmark import submit_confirmed, validate_predictions

"""Score all distinct trained PPE best checkpoints on frozen customer-nvr snapshot2."""
import os

from core.validation import require

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/andromeda-customer-nvr-20261116"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def server_runs():
    return json.loads(
        subprocess.check_output(
            ["andromeda", "detection", "runs", "107", "7", "--json"]
        )
    )["runs"]


def main():
    exam = read(OUT / "exam/members.json")
    require(
        exam["definition"]["id"] == "7" and len(exam["members"]) == 87,
        "Integrity check failed: exam['definition']['id'] == '7' and len(exam['members']) == 87",
    )
    require(
        exam["manifest_sha256"]
        == "dc72f2e32c774925e482f76ca75a10fa447786c35c31758a80bec76e71cdfe45",
        "Integrity check failed: exam['manifest_sha256'] == 'dc72f2e32c774925e482f76ca75a10fa447786c35c31758a80bec76e71cdfe45'",
    )
    paths = []
    for m in exam["members"]:
        p = OUT / "exam/images" / (m["sample_id"] + m["ext"])
        require(sha(p) == m["sha256"], "Integrity check failed: sha(p) == m['sha256']")
        paths.append(str(p))
    models = []
    seen = set()
    for w in sorted((ROOT / "runs").glob("*/weights/best.pt")):
        run = w.parent.parent
        args = yaml.safe_load((run / "args.yaml").read_text())
        directory = OUT / run.name
        directory.mkdir(exist_ok=True)
        state = (
            read(run / "training-state.json")
            if (run / "training-state.json").exists()
            else {}
        )
        partial = state.get("status") in ["training", "starting", "evaluating"]
        pilot = "pilot" in run.name
        if partial:
            frozen = directory / "frozen-best.pt"
            if not frozen.exists():
                before = sha(w)
                shutil.copy2(w, frozen)
                require(
                    sha(w) == sha(frozen) == before,
                    "Checkpoint changed while copying; retry snapshot",
                )
            w = frozen
        h = sha(w)
        if h in seen:
            continue
        seen.add(h)
        models.append(
            dict(
                name=run.name,
                weights=str(w),
                sha256=h,
                imgsz=args["imgsz"],
                dataset=args["data"],
                partial=partial,
                pilot=pilot,
                state_at_snapshot=state,
            )
        )
    require(len(models) >= 9, "Integrity check failed: len(models) >= 9")
    (OUT / "inventory.json").write_text(json.dumps(models, indent=2) + "\n")
    results = []
    torch.set_num_threads(4)
    for model in models:
        directory = OUT / model["name"]
        print("START", model["name"], flush=True)
        found = [
            r
            for r in server_runs()
            if r["candidate"].get("weights_sha256") == model["sha256"]
        ]
        require(len(found) <= 1, "Integrity check failed: len(found) <= 1")
        if not found:
            require(
                not any(
                    (directory / name).exists()
                    for name in ("receipt.json", "submission-pending.json")
                ),
                "Receipt without server confirmation; inspect before retry",
            )
            net = YOLO(model["weights"])
            require(
                {c["name"] for c in exam["classes"]} <= set(net.names.values()),
                "Integrity check failed: {c['name'] for c in exam['classes']} <= set(net.names.values())",
            )
            net.add_callback("on_predict_start", lambda _: torch.set_num_threads(4))
            bypath = dict(zip(paths, exam["members"], strict=True))
            pred = {}
            for r in net.predict(
                paths,
                device=0,
                imgsz=model["imgsz"],
                batch=1,
                conf=0.05,
                iou=0.7,
                max_det=300,
                stream=True,
                verbose=False,
            ):
                pred[bypath[r.path]["sample_id"]] = [
                    dict(cls=net.names[int(c)], box=b, conf=v)
                    for b, v, c in zip(
                        r.boxes.xyxy.cpu().tolist(),
                        r.boxes.conf.cpu().tolist(),
                        r.boxes.cls.cpu().tolist(),
                        strict=True,
                    )
                ]
            del net
            gc.collect()
            torch.cuda.empty_cache()
            validate_predictions(pred, exam["members"])
            (directory / "predictions.json").write_text(json.dumps(pred))
            config = dict(
                **model,
                drop_unmapped=True,
                exam_classes=[c["name"] for c in exam["classes"]],
                unmapped_box_counts={
                    name: sum(
                        b["cls"] == name for boxes in pred.values() for b in boxes
                    )
                    for name in sorted(
                        {b["cls"] for boxes in pred.values() for b in boxes}
                        - {c["name"] for c in exam["classes"]}
                    )
                },
                confidence=0.05,
                nms_iou=0.7,
                max_det=300,
                batch=1,
                device="physical GPU0 / visible CUDA0",
                benchmark_manifest=exam["manifest_sha256"],
                scorer=exam["definition"]["scorer"],
                benchmark_min_conf=exam["definition"]["min_conf"],
                matching_iou=exam["definition"]["iou"],
                note="Full-frame detection; no ROI/tracking or test tuning. Pilot/in-progress marked; benchmark labels stay on server.",
            )
            (directory / "config.json").write_text(json.dumps(config, indent=2))
            version = (
                "in-progress-frozen-best"
                if model["partial"]
                else ("pilot-incomplete" if model["pilot"] else "best-validation")
            )
            found = [
                submit_confirmed(
                    107,
                    7,
                    directory,
                    model["weights"],
                    model["name"],
                    version,
                    drop_unmapped=True,
                )
            ]
        confirmed = found[0]
        (directory / "confirmed-server-run.json").write_text(
            json.dumps(confirmed, indent=2)
        )
        results.append(
            dict(
                model=model,
                server_run=confirmed["number"],
                results=confirmed["results"],
            )
        )
        (OUT / "results.json").write_text(json.dumps(results, indent=2))
        print("CONFIRMED", model["name"], confirmed["number"], flush=True)
    (OUT / "state.json").write_text(
        json.dumps(dict(status="complete", models=len(results), updated_at=time.time()))
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        (OUT / "state.json").write_text(
            json.dumps(dict(status="failed", error=repr(e)))
        )
        raise
