"""CPU-only regression tests; no model download, GPU work, or server calls."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_submission as benchmark
import dataset_release
import release_validation as validation
import train

REPO = Path(__file__).resolve().parents[1]


class Workflows(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        rows = []
        for index, split in enumerate(("train", "val", "test")):
            image = self.data / "images" / split / "example.jpg"
            label = self.data / "labels" / split / "example.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(f"image-{index}".encode())
            label.write_text("0 0.5 0.5 0.2 0.2\n")
            rows.append(
                dict(
                    image=str(image),
                    label=str(label),
                    image_sha256=validation.digest(image),
                    label_sha256=validation.digest(label),
                    group=split,
                    split=split,
                )
            )
        self.rows = rows
        (self.data / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        import yaml

        (self.data / "data.yaml").write_text(
            yaml.safe_dump(dict(names=validation.CLASSES))
        )
        self.weights = self.root / "model.pt"
        self.weights.write_bytes(b"weights")
        self.recipe = train.load_recipe(REPO / "configs/full-yolo26s.json", self.root)
        self.recipe.update(
            data=str(self.data), weights=str(self.weights), validation=dict(images=3)
        )
        self.run = Path(self.recipe["run"])
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            mem_get_info=lambda: (10 * 1024**3, 12 * 1024**3),
        )
        self.torch = SimpleNamespace(
            cuda=self.cuda, load=Mock(return_value=dict(epoch=1, optimizer={}))
        )
        self.training_calls = []
        self.evaluation_calls = []
        owner = self

        class YOLO:
            def __init__(self, weights):
                self.weights = weights

            def add_callback(self, name, callback):
                pass

            def train(self, **kwargs):
                owner.training_calls.append(kwargs)
                target = owner.run / "weights/best.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"best")

            def val(self, **kwargs):
                owner.evaluation_calls.append(kwargs)
                return SimpleNamespace(
                    results_dict={"metrics/mAP50": 0.5},
                    box=SimpleNamespace(
                        ap_class_index=[0],
                        class_result=lambda index: (0.4, 0.5, 0.6, 0.7),
                    ),
                )

        self.yolo = YOLO

    def run_training(self):
        train.run_experiment(self.recipe, self.root, self.torch, self.yolo)

    def test_fresh_complete_and_recipe_change(self):
        self.run_training()
        self.assertEqual(self.training_calls[0]["batch"], 4)
        self.assertEqual(self.training_calls[0]["optimizer"], "MuSGD")
        self.assertEqual(self.evaluation_calls[0]["split"], "test")
        self.assertEqual(
            len(json.loads(Path(self.recipe["metrics"]).read_text())["per_class"]), 11
        )
        self.run_training()
        self.assertEqual(len(self.training_calls), 1)
        self.recipe["train"]["batch"] = 2
        with self.assertRaisesRegex(ValueError, "Recipe changed"):
            self.run_training()

    def test_resume_falls_back_from_corrupt_last(self):
        folder = self.run / "weights"
        folder.mkdir(parents=True)
        for name in ("last.pt", "epoch1.pt"):
            (folder / name).write_bytes(b"x")
        self.torch.load.side_effect = [EOFError("corrupt"), dict(epoch=1, optimizer={})]
        self.run_training()
        self.assertEqual(self.training_calls, [dict(resume=True, device=0)])
        self.assertEqual(self.torch.load.call_args.args[0], folder / "epoch1.pt")

    def test_no_usable_checkpoint_fails_without_training(self):
        folder = self.run / "weights"
        folder.mkdir(parents=True)
        (folder / "last.pt").write_bytes(b"x")
        self.torch.load.return_value = dict(epoch=-1, optimizer=None)
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            self.run_training()
        self.assertFalse(self.training_calls)
        self.assertEqual(
            json.loads(Path(self.recipe["state"]).read_text())["status"], "failed"
        )

    def test_evaluation_failure_retries_without_retraining(self):
        original = self.yolo.val
        with patch.object(
            self.yolo, "val", side_effect=RuntimeError("evaluation failed")
        ):
            with self.assertRaises(RuntimeError):
                self.run_training()
        self.assertEqual(
            json.loads(Path(self.recipe["state"]).read_text())["phase"], "evaluating"
        )
        self.run_training()
        self.assertEqual(len(self.training_calls), 1)
        self.assertIs(self.yolo.val, original)

    def test_mutated_label_fails_before_training(self):
        Path(self.rows[0]["label"]).write_text("1 0.5 0.5 0.2 0.2\n")
        with self.assertRaisesRegex(ValueError, "Label changed"):
            self.run_training()
        self.assertFalse(self.training_calls)

    def test_group_leakage_fails_before_training(self):
        self.rows[1]["group"] = self.rows[0]["group"]
        (self.data / "manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows)
        )
        with self.assertRaisesRegex(ValueError, "Family crosses"):
            self.run_training()

    def test_gpu_memory_guard(self):
        self.recipe["minimum_free_gib"] = 20
        with self.assertRaisesRegex(ValueError, "GPU memory"):
            self.run_training()
        self.assertFalse(self.training_calls)

    def test_reviewed_baseline_comparison(self):
        self.recipe["baseline_weights"] = str(self.weights)
        self.run_training()
        comparison = json.loads((self.run / "comparison.json").read_text())
        self.assertEqual(comparison["delta"], {"metrics/mAP50": 0})
        self.assertEqual(len(self.evaluation_calls), 2)

    def test_connected_families_keep_transitive_duplicates(self):
        rows = [
            dict(image_sha256=h, original_family=f)
            for h, f in [("a", "one"), ("b", "two"), ("a", "two"), ("c", "three")]
        ]
        groups = dataset_release.connected_groups(rows)
        self.assertEqual(list(groups), [2, 3])
        self.assertEqual(groups[2], rows[:3])

    def test_append_split_guards_preserve_core_and_quarantine(self):
        choose = dataset_release.choose_append_split
        rng = Mock()
        self.assertEqual(choose({"train"}, False, True, False, rng), "test")
        self.assertEqual(choose({"val"}, False, False, False, rng), "val")
        self.assertEqual(choose(set(), True, False, True, rng), "test")
        self.assertEqual(choose(set(), False, False, True, rng), "train")
        for previous, forced in [({"val", "test"}, False), ({"val"}, True)]:
            with self.assertRaises(ValueError):
                choose(previous, forced, False, False, rng)
        rng.random.assert_not_called()

    def test_recipe_matrix_preserves_original_settings(self):
        expected = {
            "full-yolo11s": ("yolo11s", 640, 12, 10030),
            "reviewed-yolo11s": ("yolo11s", 640, 12, 2250),
            "reviewed-yolo26s": ("yolo26s", 640, 4, 2250),
            "reviewed-yolo26m": ("yolo26m", 640, 4, 2250),
            "full-yolo26s": ("yolo26s", 640, 4, 10030),
            "full-yolo26s-960": ("yolo26s", 960, 4, 10030),
            "expanded-five": ("yolo26s", 960, 4, 14536),
            "expanded-eight": ("yolo26s", 960, 4, 18150),
            "expanded-latest-s": ("yolo26s", 960, 4, 22709),
            "expanded-latest-l": ("yolo26l", 960, 2, 22709),
        }
        for name, values in expected.items():
            recipe = train.load_recipe(REPO / "configs" / (name + ".json"), self.root)
            settings = recipe["train"]
            self.assertEqual(
                (
                    Path(recipe["weights"]).stem,
                    settings["imgsz"],
                    settings["batch"],
                    recipe["validation"]["images"],
                ),
                values,
            )
            self.assertEqual(
                (settings["epochs"], settings["patience"], settings["seed"]),
                (100, 20, 42),
            )
            self.assertEqual(settings["mosaic"], 0.5)
            self.assertEqual(
                settings["optimizer"], "auto" if name == "full-yolo11s" else "MuSGD"
            )

    def test_release_refuses_overwrite_for_both_policies(self):
        for policy in ("next-three", "four-more"):
            with self.assertRaises((ValueError, AssertionError)):
                dataset_release.build_append(self.data, self.data, self.root, policy)

    def test_labels_preserve_negative_policy_and_reject_nan(self):
        label = self.root / "labels.txt"
        label.write_text("0 .5 .5 .2 .2\n1 .5 .5 .2 .2\n")
        lines, omitted, _ = dataset_release.parse(label, ["Helmet", "no_helmet"])
        self.assertEqual(lines, ["0 .5 .5 .2 .2"])
        self.assertEqual(omitted, {"no_helmet": 1})
        label.write_text("0 nan .5 .2 .2\n")
        with self.assertRaises((ValueError, AssertionError)):
            dataset_release.parse(label, ["Helmet"])

    def test_submit_confirms_and_never_duplicates(self):
        directory = self.root / "submission"
        directory.mkdir()
        (directory / "config.json").write_text(
            json.dumps(dict(weights_sha256=validation.digest(self.weights)))
        )
        server = dict(number=8)
        with (
            patch.object(benchmark, "find_run", side_effect=[None, server, server]),
            patch.object(
                benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
            ) as submit,
        ):
            for _ in range(2):
                self.assertEqual(
                    benchmark.submit_confirmed(
                        1, 2, directory, self.weights, "test", "best"
                    ),
                    server,
                )
            self.assertEqual(submit.call_count, 1)

    def test_receipt_without_server_blocks_retry(self):
        (self.root / "receipt.json").write_text("{}")
        with (
            patch.object(benchmark, "find_run", return_value=None),
            patch.object(benchmark.subprocess, "run") as submit,
        ):
            with self.assertRaisesRegex(ValueError, "Receipt"):
                benchmark.submit_confirmed(
                    1, 2, self.root, self.weights, "test", "best"
                )
            submit.assert_not_called()

    def test_failed_upload_leaves_pending_record(self):
        (self.root / "config.json").write_text(
            json.dumps(dict(weights_sha256=validation.digest(self.weights)))
        )
        with (
            patch.object(benchmark, "find_run", return_value=None),
            patch.object(
                benchmark.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=1, stdout="", stderr="failed"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "CLI submission failed"):
                benchmark.submit_confirmed(
                    1, 2, self.root, self.weights, "test", "best"
                )
        self.assertTrue((self.root / "submission-pending.json").exists())

    def test_prediction_validation(self):
        members = [dict(sample_id="a")]
        good = {"a": [dict(cls="Helmet", box=[0, 0, 1, 1], conf=0.5)]}
        benchmark.validate_predictions(good, members, {"Helmet"})
        for predictions in (
            {},
            {"a": [dict(cls="Helmet", box=[0, 0, float("nan"), 1], conf=0.5)]},
            {"a": [dict(cls="Helmet", box=[0, 0, 1, 1], conf=float("nan"))]},
        ):
            with self.assertRaises(ValueError):
                benchmark.validate_predictions(predictions, members, {"Helmet"})

    def test_shared_modules_import_without_gpu_or_filesystem_side_effects(self):
        code = "import os,sys; sys.path.insert(0, sys.argv[1]); os.environ['CUDA_VISIBLE_DEVICES']='sentinel'; import train,dataset_release,benchmark_submission; assert os.environ['CUDA_VISIBLE_DEVICES']=='sentinel'; assert 'torch' not in sys.modules; assert 'ultralytics' not in sys.modules"
        subprocess.run(
            [sys.executable, "-c", code, str(REPO / "scripts")],
            cwd=self.root,
            check=True,
        )

    def test_all_recipes_dry_run_without_cuda_import(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/train.py"),
                *[str(p) for p in sorted((REPO / "configs").glob("*.json"))],
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        recipes = json.loads(result.stdout)
        self.assertEqual(len(recipes), 10)
        self.assertEqual({r["train"]["imgsz"] for r in recipes}, {640, 960})
        self.assertEqual(
            {Path(r["weights"]).stem for r in recipes},
            {"yolo11s", "yolo26s", "yolo26m", "yolo26l"},
        )


if __name__ == "__main__":
    unittest.main()
