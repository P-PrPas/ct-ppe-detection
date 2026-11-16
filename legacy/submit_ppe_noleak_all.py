from core.benchmark import submit_confirmed, validate_predictions

"""Submit each distinct trained best checkpoint to frozen project102/exam4, once."""
import os

from core.validation import require

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/andromeda-noleak-all-models"


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def runs():
    return json.loads(
        subprocess.check_output(
            ["andromeda", "detection", "runs", "102", "4", "--json"]
        )
    )["runs"]


def main():
    new = read(OUT / "exam/members.json")
    old = read(ROOT / "reports/andromeda-benchmark-positive/exam/members.json")
    require(
        len(new["members"]) == 3783 and new["definition"]["id"] == "4",
        "Integrity check failed: len(new['members']) == 3783 and new['definition']['id'] == '4'",
    )
    require(
        new["definition"]["min_conf"] == 0.5 and new["definition"]["iou"] == 0.5,
        "Integrity check failed: new['definition']['min_conf'] == 0.5 and new['definition']['iou'] == 0.5",
    )
    old_by_hash = {m["sha256"]: m for m in old["members"]}
    require(
        all(m["sha256"] in old_by_hash for m in new["members"]),
        "Integrity check failed: all((m['sha256'] in old_by_hash for m in new['members']))",
    )
    paths = []
    for m in new["members"]:
        o = old_by_hash[m["sha256"]]
        p = (
            ROOT
            / "reports/andromeda-benchmark-positive/exam/images"
            / (o["sample_id"] + o["ext"])
        )
        require(sha(p) == m["sha256"], "Integrity check failed: sha(p) == m['sha256']")
        paths.append(str(p))
    models = []
    for w in sorted((ROOT / "runs").glob("*/weights/best.pt")):
        a = yaml.safe_load((w.parent.parent / "args.yaml").read_text())
        models.append(
            dict(
                run=w.parent.parent.name,
                weights=str(w),
                sha=sha(w),
                imgsz=a["imgsz"],
                dataset=a["data"],
                pilot="pilot" in w.parent.parent.name,
            )
        )
    require(
        len(models) == 7 and len({m["sha"] for m in models}) == 7,
        "Integrity check failed: len(models) == 7 and len({m['sha'] for m in models}) == 7",
    )
    (OUT / "inventory.json").write_text(json.dumps(models, indent=2) + "\n")
    previous = []
    for c in (ROOT / "reports").glob("*/config.json"):
        try:
            conf = read(c)
            if (
                "weights_sha256" in conf
                and (c.parent / "predictions.json").exists()
                and conf.get("benchmark_manifest") == old["manifest_sha256"]
            ):
                previous.append((c.parent, conf))
        except (ValueError, KeyError):
            pass
    summary = []
    for model in models:
        print("START", model["run"], flush=True)
        directory = OUT / model["run"]
        directory.mkdir(exist_ok=True)
        (OUT / "state.json").write_text(
            json.dumps(
                dict(
                    stage="processing",
                    model=model["run"],
                    completed=len(summary),
                    total=7,
                )
            )
        )
        existing = [
            r for r in runs() if r["candidate"].get("weights_sha256") == model["sha"]
        ]
        require(len(existing) <= 1, "Integrity check failed: len(existing) <= 1")
        if existing:
            require(
                existing[0]["candidate"]["config"]["imgsz"] == model["imgsz"],
                "Integrity check failed: existing[0]['candidate']['config']['imgsz'] == model['imgsz']",
            )
            server = existing[0]
        else:
            require(
                not (directory / "receipt.json").exists(),
                "Receipt without server confirmation: inspect before retry",
            )
            candidates = [
                (p, c)
                for p, c in previous
                if c["weights_sha256"] == model["sha"]
                and c["imgsz"] == model["imgsz"]
                and c["confidence"] == 0.05
                and c["nms_iou"] == 0.7
                and c["max_det"] == 300
            ]
            pred = {}
            method = "fresh inference"
            if candidates:
                p, c = candidates[0]
                cached = read(p / "predictions.json")
                for m in new["members"]:
                    pred[m["sample_id"]] = cached[old_by_hash[m["sha256"]]["sample_id"]]
                method = (
                    "reused exact image-hash and weights-hash predictions from "
                    + str(p)
                )
            else:
                import torch
                from ultralytics import YOLO

                torch.set_num_threads(2)
                net = YOLO(model["weights"])
                require(
                    set(net.names.values()) == {c["name"] for c in new["classes"]},
                    "Integrity check failed: set(net.names.values()) == {c['name'] for c in new['classes']}",
                )
                net.add_callback("on_predict_start", lambda _: torch.set_num_threads(2))
                bypath = dict(zip(paths, new["members"], strict=True))
                for i, r in enumerate(
                    net.predict(
                        paths,
                        device=0,
                        imgsz=model["imgsz"],
                        batch=4,
                        conf=0.05,
                        iou=0.7,
                        max_det=300,
                        stream=True,
                        verbose=False,
                    ),
                    1,
                ):
                    m = bypath[r.path]
                    require(
                        m["sample_id"] not in pred,
                        "Integrity check failed: m['sample_id'] not in pred",
                    )
                    pred[m["sample_id"]] = [
                        dict(cls=net.names[int(c)], box=b, conf=v)
                        for b, v, c in zip(
                            r.boxes.xyxy.cpu().tolist(),
                            r.boxes.conf.cpu().tolist(),
                            r.boxes.cls.cpu().tolist(),
                            strict=True,
                        )
                    ]
                    if i % 500 == 0:
                        print("PREDICT", i, 3783, flush=True)
                del net
                gc.collect()
                torch.cuda.empty_cache()
            validate_predictions(
                pred, new["members"], {c["name"] for c in new["classes"]}
            )
            (directory / "predictions.json").write_text(
                json.dumps(pred, separators=(",", ":"))
            )
            config = dict(
                weights_sha256=model["sha"],
                imgsz=model["imgsz"],
                batch=4,
                confidence=0.05,
                nms_iou=0.7,
                max_det=300,
                benchmark_manifest=new["manifest_sha256"],
                scorer=new["definition"]["scorer"],
                benchmark_min_conf=0.5,
                matching_iou=0.5,
                provenance=method,
                pilot=model["pilot"],
                dataset=model["dataset"],
                note="All trained best checkpoints, pilot marked explicitly. No test threshold tuning. Benchmark labels stay on server.",
            )
            (directory / "config.json").write_text(json.dumps(config, indent=2))
            server = submit_confirmed(
                102,
                4,
                directory,
                model["weights"],
                model["run"],
                "best-pilot-incomplete" if model["pilot"] else "best-validation",
            )
        (directory / "confirmed-server-run.json").write_text(
            json.dumps(server, indent=2) + "\n"
        )
        summary.append(
            dict(**model, server_run=server["number"], results=server["results"])
        )
        (OUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("CONFIRMED", server["number"], json.dumps(server["results"]), flush=True)
    (OUT / "state.json").write_text(
        json.dumps(
            dict(
                stage="complete",
                completed=len(summary),
                total=7,
                updated_at=time.time(),
            )
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        (OUT / "state.json").write_text(json.dumps(dict(stage="failed", error=repr(e))))
        raise
