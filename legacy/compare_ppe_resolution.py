"""Compare the frozen best model on validation only; leave data and weights unchanged."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_MODE"] = "disabled"
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ppe-full-current-20261109"
WEIGHTS = ROOT / "runs/ppe-full-current-10030-yolo26s-20261111/weights/best.pt"
OUT = ROOT / "reports/yolo26s-resolution-validation"


def match(gt, pred):
    """Class-aware greedy confidence matching at IoU .5, original pixel coordinates."""
    used = set()
    for p in sorted(pred, key=lambda p: -p[5]):
        candidates = []
        for i, g in enumerate(gt):
            if i in used or g[0] != p[0]:
                continue
            a, b = np.array(g[1:5]), np.array(p[1:5])
            inter = np.maximum(
                np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0
            ).prod()
            iou = inter / (
                np.maximum(a[2:] - a[:2], 0).prod()
                + np.maximum(b[2:] - b[:2], 0).prod()
                - inter
                + 1e-9
            )
            if iou >= 0.5:
                candidates.append((iou, i))
        if candidates:
            used.add(max(candidates)[1])
    return used


def boxes(path, w, h, prediction=False):
    result = []
    for line in path.read_text().splitlines() if path.exists() else []:
        a = list(map(float, line.split()))
        assert len(a) == (6 if prediction else 5)
        c, x, y, bw, bh = a[:5]
        assert 0 <= c < 11 and c.is_integer()
        result.append(
            [
                int(c),
                (x - bw / 2) * w,
                (y - bh / 2) * h,
                (x + bw / 2) * w,
                (y + bh / 2) * h,
            ]
            + a[5:]
        )
    return result


def main():
    assert match(
        [[0, 0, 0, 10, 10]], [[0, 0, 0, 10, 10, 0.9], [0, 0, 0, 10, 10, 0.8]]
    ) == {0}
    assert not match([[0, 0, 0, 10, 10]], [[1, 0, 0, 10, 10, 0.9]])
    OUT.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text().splitlines()]
    val = [r for r in rows if r["split"] == "val"]
    assert len(val) == 1003
    sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    results = {}
    for size in (640, 960):
        out = OUT / f"val-{size}"
        assert not out.exists(), f"Existing output needs inspection: {out}"
        (OUT / "state.json").write_text(
            json.dumps(dict(stage="validation", imgsz=size, updated_at=time.time()))
        )
        started = time.monotonic()
        m = YOLO(str(WEIGHTS)).val(
            data=str(DATA / "data.yaml"),
            split="val",
            imgsz=size,
            batch=4,
            device=0,
            workers=4,
            conf=0.001,
            iou=0.7,
            max_det=300,
            save_txt=True,
            save_conf=True,
            plots=True,
            project=str(OUT),
            name=f"val-{size}",
        )
        totals = Counter()
        per = defaultdict(Counter)
        sizes = defaultdict(Counter)
        images = []
        for row in val:
            image = Path(row["image"])
            with Image.open(image) as im:
                w, h = im.size
            gt = boxes(Path(row["label"]), w, h)
            pred = [
                p
                for p in boxes(out / "labels" / f"{image.stem}.txt", w, h, True)
                if p[5] >= 0.5
            ]
            used = match(gt, pred)
            totals.update(
                tp=len(used), fp=len(pred) - len(used), fn=len(gt) - len(used)
            )
            for j, g in enumerate(gt):
                per[g[0]]["gt"] += 1
                per[g[0]]["tp"] += j in used
                # Fixed 640-reference bins keep the exact same objects in each bin across both runs.
                edge = np.sqrt(max(0, (g[3] - g[1]) * (g[4] - g[2]))) * 640 / max(w, h)
                key = (
                    "<16"
                    if edge < 16
                    else "16-32"
                    if edge < 32
                    else "32-96"
                    if edge < 96
                    else ">=96"
                )
                sizes[key]["gt"] += 1
                sizes[key]["tp"] += j in used
            for p in pred:
                per[p[0]]["pred"] += 1
            images.append(
                dict(
                    image=str(image),
                    label=row["label"],
                    group=row["group"],
                    tp=len(used),
                    fp=len(pred) - len(used),
                    fn=len(gt) - len(used),
                    missed_classes=[gt[j][0] for j in range(len(gt)) if j not in used],
                )
            )
        for v in [totals, *per.values()]:
            tp = v["tp"]
            gt = v.get("gt", tp + v.get("fn", 0))
            n = v.get("pred", tp + v.get("fp", 0))
            v.update(
                precision=tp / max(1, n),
                recall=tp / max(1, gt),
                f1=2 * tp / max(1, n + gt),
            )
        for v in sizes.values():
            v["recall"] = v["tp"] / v["gt"]
        result = dict(
            imgsz=size,
            weights_sha256=sha,
            images=1003,
            split="val",
            elapsed_seconds=time.monotonic() - started,
            fixed_conf_05_iou_05=dict(totals),
            per_class={m.names[k]: dict(v) for k, v in per.items()},
            size_reference_640=dict(sizes),
            ultralytics_metrics=m.results_dict,
            metric_note="Ultralytics P/R are macro at selected confidence; fixed totals use class-aware greedy matching. Size bins use sqrt box area scaled to640 for BOTH runs.",
        )
        (OUT / f"result-{size}.json").write_text(json.dumps(result, indent=2) + "\n")
        (OUT / f"images-{size}.json").write_text(json.dumps(images, indent=2) + "\n")
        results[size] = result
        print("VALIDATION COMPLETE", size, json.dumps(result), flush=True)
    (OUT / "state.json").write_text(
        json.dumps(
            dict(stage="complete", updated_at=time.time(), results=results), indent=2
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        OUT.mkdir(exist_ok=True)
        (OUT / "state.json").write_text(
            json.dumps(dict(stage="failed", error=repr(e), updated_at=time.time()))
        )
        raise
