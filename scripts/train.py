"""One entry point for PPE experiments. Recipe paths are relative to --root."""

import argparse
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path

from release_validation import CLASSES, digest, require, verify_release

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_recipe(path, root):
    recipe = json.loads(path.read_text())
    required = {"data", "run", "weights", "validation", "train", "eval_batch", "gpu"}
    require(
        required <= recipe.keys(), f"Missing recipe keys: {required - recipe.keys()}"
    )
    for key in (
        "data",
        "run",
        "weights",
        "state",
        "metrics",
        "baseline_weights",
        "preflight",
    ):
        if recipe.get(key):
            recipe[key] = str((root / recipe[key]).resolve())
    run = Path(recipe["run"])
    recipe.setdefault("state", str(run / "training-state.json"))
    recipe.setdefault("metrics", str(run / "test-metrics.json"))
    require(
        recipe["train"]["imgsz"] > 0 and recipe["train"]["imgsz"] % 32 == 0,
        "imgsz must be a positive multiple of 32",
    )
    require(
        recipe["train"]["batch"] > 0 and recipe["eval_batch"] > 0,
        "Batch sizes must be positive",
    )
    return recipe


def select_checkpoint(run, torch):
    candidates = [run / "weights/last.pt"] + sorted(
        (run / "weights").glob("epoch*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    existing = [path for path in candidates if path.exists()]
    for path in existing:
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("epoch", -1) >= 0 and saved.get("optimizer") is not None:
                return path
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            print(f"Skipping unreadable checkpoint {path}: {error}", flush=True)
    require(not existing, "No checkpoint with optimizer state available to resume")
    return None


def evaluate(model, recipe, name):
    data, run = Path(recipe["data"]), Path(recipe["run"])
    metrics = model.val(
        data=str(data / "data.yaml"),
        split="test",
        imgsz=recipe["train"]["imgsz"],
        batch=recipe["eval_batch"],
        device=0,
        project=str(run),
        name=name,
        plots=True,
    )
    boxes, images = Counter(), Counter()
    for path in (data / "labels/test").glob("*.txt"):
        ids = [
            int(line.split()[0])
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        boxes.update(ids)
        images.update(set(ids))
    per_class = []
    indices = list(metrics.box.ap_class_index)
    for index, label in enumerate(CLASSES):
        values = (
            metrics.box.class_result(indices.index(index))
            if index in indices
            else (0.0, 0.0, 0.0, 0.0)
        )
        per_class.append(
            dict(
                name=label,
                precision=float(values[0]),
                recall=float(values[1]),
                mAP50=float(values[2]),
                mAP50_95=float(values[3]),
                boxes=boxes[index],
                images=images[index],
            )
        )
    return dict(overall=metrics.results_dict, per_class=per_class)


def run_experiment(recipe, root, torch, yolo):
    """The same lifecycle is used by the CLI and tests with a fake YOLO adapter."""
    run, data = Path(recipe["run"]), Path(recipe["data"])
    state_path, metrics_path = Path(recipe["state"]), Path(recipe["metrics"])
    best = run / "weights/best.pt"

    phase = "validating"

    def state(status, **details):
        nonlocal phase
        phase = status
        write_json(
            state_path,
            dict(status=status, updated_at=time.time(), run=str(run), **details),
        )

    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get("status") == "complete":
        resolved = run / "resolved-recipe.json"
        if resolved.exists():
            require(
                json.loads(resolved.read_text())["recipe"] == recipe,
                "Recipe changed for completed run; use a new run directory",
            )
        require(
            best.is_file() and metrics_path.is_file(),
            "Completed run is missing artifacts; inspect before restarting",
        )
        print(f"Already complete: {run}", flush=True)
        return
    try:
        require(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "Expose exactly one CUDA device",
        )
        if recipe.get("minimum_free_gib"):
            free, _ = torch.cuda.mem_get_info()
            require(
                free >= recipe["minimum_free_gib"] * 1024**3,
                "Insufficient free GPU memory; do not disturb other jobs",
            )
        if recipe.get("preflight"):
            preflight = json.loads(Path(recipe["preflight"]).read_text())
            require(
                len(preflight) == 2
                and all(
                    r["forward_backward"] == "passed" and r["batch"] == 4
                    for r in preflight
                ),
                "YOLO26 preflight has not passed",
            )
        snapshots = verify_release(data, recipe["validation"], root)
        run.mkdir(parents=True, exist_ok=True)
        record = dict(
            recipe=recipe,
            dataset=str(data),
            base_weights=recipe["weights"],
            base_weights_sha256=digest(recipe["weights"]),
            **recipe["train"],
        )
        record["input_hashes"] = digest(data / "manifest.jsonl")
        record_path = run / "resolved-recipe.json"
        if record_path.exists():
            require(
                json.loads(record_path.read_text()) == record,
                "Recipe or inputs changed for an existing run; use a new run directory",
            )
        else:
            write_json(record_path, record)
        if not (run / "experiment.json").exists():
            write_json(run / "experiment.json", record)
        snapshot_path = run / "input-hashes.jsonl"
        snapshot_text = "".join(json.dumps(row) + "\n" for row in snapshots)
        if snapshot_path.exists():
            old = [json.loads(line) for line in snapshot_path.read_text().splitlines()]
            require(old == snapshots, "Frozen run inputs changed")
        else:
            snapshot_path.write_text(snapshot_text)
        evaluation_pending = previous.get("status") in {
            "evaluating",
            "evaluating_baseline",
        } or previous.get("phase") in {"evaluating", "evaluating_baseline"}
        if evaluation_pending:
            require(best.is_file(), "Evaluation checkpoint is missing")
        else:
            checkpoint = select_checkpoint(run, torch)
            if checkpoint and (run / "args.yaml").exists():
                import yaml

                saved_args = yaml.safe_load((run / "args.yaml").read_text())
                for key in ("imgsz", "batch", "epochs", "patience", "seed"):
                    require(
                        saved_args.get(key) == recipe["train"][key],
                        f"Resume setting differs: {key}; use original recipe",
                    )
                require(
                    Path(saved_args["data"]).resolve()
                    == (data / "data.yaml").resolve(),
                    "Resume dataset differs",
                )
            model = yolo(str(checkpoint or recipe["weights"]))
            model.add_callback(
                "on_model_save",
                lambda trainer: state(
                    "training",
                    epoch=trainer.epoch + 1,
                    last_checkpoint=str(run / "weights/last.pt"),
                ),
            )
            model.add_callback(
                "on_train_epoch_start",
                lambda trainer: state(
                    "training",
                    epoch=trainer.epoch + 1,
                    checkpoint_exists=(run / "weights/last.pt").exists(),
                ),
            )
            state("starting", resume=checkpoint is not None)
            if checkpoint:
                model.train(resume=True, device=0)
            else:
                model.train(
                    data=str(data / "data.yaml"),
                    project=str(run.parent),
                    name=run.name,
                    exist_ok=True,
                    device=0,
                    **recipe["train"],
                )
        state("evaluating", best=str(best))
        if evaluation_pending and metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        else:
            metrics = evaluate(yolo(str(best)), recipe, "heldout-test")
            write_json(metrics_path, metrics)
        details = dict(best=str(best), metrics=str(metrics_path))
        if recipe.get("baseline_weights"):
            state("evaluating_baseline", **details)
            baseline = evaluate(
                yolo(recipe["baseline_weights"]), recipe, "baseline-on-reviewed-test"
            )
            comparison = run / "comparison.json"
            write_json(
                comparison,
                dict(
                    test="Identical reviewed test split:188 images, same labels, same evaluation settings",
                    baseline=baseline,
                    reviewed=metrics,
                    delta={
                        key: value - baseline["overall"][key]
                        for key, value in metrics["overall"].items()
                        if key.startswith("metrics/")
                    },
                    limitations="One seed; small rare-class support. Reviewed subset changes both quality and quantity, not a causal isolation of label quality.",
                ),
            )
            details["comparison"] = str(comparison)
        state("complete", **details)
    except BaseException as error:
        state("failed", error=repr(error), phase=phase)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recipes", nargs="+", type=Path, help="JSON recipe(s), executed in order"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Dataset/model/output root; defaults to repository root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved recipes without importing CUDA or writing files",
    )
    args = parser.parse_args()
    recipes = [load_recipe(path, args.root.resolve()) for path in args.recipes]
    require(
        len({recipe["gpu"] for recipe in recipes}) == 1,
        "Sequential recipes must use the same physical GPU",
    )
    if args.dry_run:
        print(json.dumps(recipes, indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(recipes[0]["gpu"])
    os.environ["WANDB_MODE"] = "disabled"
    import torch
    from ultralytics import YOLO

    for recipe in recipes:
        run_experiment(recipe, args.root.resolve(), torch, YOLO)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
