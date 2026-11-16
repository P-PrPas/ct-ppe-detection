"""PPE experiments: config-driven dataset creation, training and benchmark submission."""

import argparse
import json
import os
from pathlib import Path

from core.training import load_recipe, run_experiment, write_json
from core.validation import require

ROOT = Path(__file__).resolve().parent


def load_config(command, path, root):
    if command == "train":
        return load_recipe(path, root)
    config = json.loads(path.read_text())
    if command == "dataset":
        required = {"manifest", "out", "expected_images", "protected_exams"}
        paths = ("manifest", "out")
    else:
        required = {
            "project",
            "definition",
            "expected_images",
            "out",
            "exam",
            "weights",
            "name",
            "version",
            "imgsz",
            "gpu",
            "device",
        }
        paths = ("out", "exam", "weights")
    require(
        not config.keys() - required, f"Unknown config keys: {config.keys() - required}"
    )
    require(
        required <= config.keys(), f"Missing config keys: {required - config.keys()}"
    )
    require(
        isinstance(config["expected_images"], int) and config["expected_images"] > 0,
        "expected_images must be a positive integer",
    )
    for key in paths:
        config[key] = str((root / config[key]).resolve())
    if command == "dataset":
        require(
            isinstance(config["protected_exams"], list),
            "protected_exams must be a list",
        )
        config["protected_exams"] = [
            str((root / p).resolve()) for p in config["protected_exams"]
        ]
    else:
        require(
            config["imgsz"] > 0 and config["imgsz"] % 32 == 0,
            "imgsz must be a positive multiple of 32",
        )
        require(
            str(config["device"]) in {"0", "cpu"},
            "device must be visible CUDA 0 or cpu",
        )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("dataset", "train", "submit"):
        sub = commands.add_parser(command)
        sub.add_argument("configs", nargs="+", type=Path)
        sub.add_argument("--root", type=Path, default=ROOT)
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Resolve config without importing CUDA, contacting servers or writing outputs",
        )
    args = parser.parse_args()
    root = args.root.resolve()
    configs = [load_config(args.command, path, root) for path in args.configs]
    if args.command != "dataset":
        require(
            len({str(config["gpu"]) for config in configs}) == 1,
            "Sequential jobs must use the same physical GPU",
        )
    if args.dry_run:
        print(json.dumps(configs, indent=2))
        return
    if args.command == "dataset":
        from core.datasets import build_release

        for config in configs:
            print(json.dumps(build_release(config), indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(configs[0]["gpu"])
    os.environ["WANDB_MODE"] = "disabled"
    import torch
    from ultralytics import YOLO

    for config in configs:
        if args.command == "train":
            run_experiment(config, root, torch, YOLO)
        else:
            from core.benchmark import predict_submit

            try:
                predict_submit(config, torch, YOLO)
            except BaseException as error:
                write_json(
                    Path(config["out"]) / "state.json",
                    dict(status="failed", error=repr(error)),
                )
                raise
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
