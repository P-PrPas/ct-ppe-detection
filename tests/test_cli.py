"""Exercise the agent-facing commands with real tiny image/label fixtures."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from core import benchmark, datasets
from core.validation import digest, verify_release

REPO = Path(__file__).resolve().parents[1]


class Commands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index, split in enumerate(("train", "val", "test")):
            image, label = self.root / f"{index}.png", self.root / f"{index}.txt"
            Image.new("RGB", (8, 8), (index * 60, 0, 0)).save(image)
            label.write_text("0 .5 .5 .25 .25\n")
            self.rows.append(
                dict(
                    image=image.name,
                    label=label.name,
                    split=split,
                    group=f"family-{index}",
                    names=["Helmet"],
                )
            )
        self.source = self.root / "source.jsonl"
        self.save_rows()
        self.config = dict(
            manifest=str(self.source),
            out=str(self.root / "release"),
            expected_images=3,
            protected_exams=[],
        )

    def save_rows(self):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def test_dataset_command_creates_trainable_release(self):
        config = self.root / "build.json"
        config.write_text(json.dumps(self.config))
        result = subprocess.run(
            [sys.executable, str(REPO / "ppe.py"), "dataset", str(config)],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            json.loads(result.stdout)["images"], {"train": 1, "val": 1, "test": 1}
        )
        verified = verify_release(self.root / "release", dict(images=3), self.root)
        self.assertEqual(len(verified), 3)
        self.assertEqual((self.root / "0.txt").read_text(), "0 .5 .5 .25 .25\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            datasets.build_release(self.config)

    def test_dataset_rejects_family_and_exact_duplicate_leakage(self):
        self.rows[1]["group"] = self.rows[0]["group"]
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "Family crosses"):
            datasets.build_release(self.config)
        self.rows[1]["group"] = "different"
        self.rows[1]["image"] = self.rows[0]["image"]
        self.save_rows()
        with self.assertRaisesRegex(ValueError, "duplicate crosses"):
            datasets.build_release(self.config)
        self.assertFalse((self.root / "release").exists())

    def test_dataset_rejects_benchmark_overlap(self):
        exam = self.root / "exam.json"
        exam.write_text(
            json.dumps(dict(members=[dict(sha256=digest(self.root / "0.png"))]))
        )
        self.config["protected_exams"] = [str(exam)]
        with self.assertRaisesRegex(ValueError, "Benchmark image"):
            datasets.build_release(self.config)
        self.assertFalse((self.root / "release").exists())

    def test_dry_run_all_commands_has_no_side_effects(self):
        for command, config in [
            ("dataset", REPO / "configs/dataset/example.json"),
            ("train", REPO / "configs/train/full-yolo26s.json"),
            ("submit", REPO / "configs/submit/full-yolo26s.json"),
        ]:
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "ppe.py"),
                    command,
                    str(config),
                    "--root",
                    str(self.root),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIsInstance(json.loads(result.stdout), list)
        self.assertFalse((self.root / "reports").exists())
        self.assertFalse((self.root / "runs").exists())

    def test_submit_predicts_verified_exam_then_uses_confirmation(self):
        out = self.root / "submission"
        out.mkdir()
        exam = self.root / "exam"
        (exam / "images").mkdir(parents=True)
        image = exam / "images/a.png"
        Image.new("RGB", (8, 8)).save(image)
        members = [dict(sample_id="a", ext=".png", sha256=digest(image))]
        (exam / "members.json").write_text(
            json.dumps(
                dict(
                    members=members,
                    classes=[dict(name="Helmet")],
                    manifest_sha256="frozen",
                    definition=dict(
                        id="3", roles=["test"], min_conf=0.5, iou=0.5, scorer="test"
                    ),
                )
            )
        )
        (out / "preparation.json").write_text(
            json.dumps(dict(images_sha256_verified=True))
        )
        weights = self.root / "weights.pt"
        weights.write_bytes(b"weights")
        config = dict(
            out=str(out),
            exam=str(exam),
            weights=str(weights),
            project=101,
            definition="3",
            expected_images=1,
            device="cpu",
            imgsz=640,
            name="test",
            version="best",
        )

        def tensor(values):
            return SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: values))

        result = SimpleNamespace(
            path=str(image),
            boxes=SimpleNamespace(
                xyxy=tensor([[0, 0, 5, 5]]), conf=tensor([0.9]), cls=tensor([0])
            ),
        )
        model = Mock(names={0: "Helmet"})
        model.predict.return_value = [result]
        yolo = Mock(return_value=model)
        with (
            patch.object(benchmark, "find_run", return_value=None),
            patch.object(
                benchmark, "submit_confirmed", return_value={"number": 2}
            ) as submit,
        ):
            benchmark.predict_submit(config, Mock(), yolo)
            self.assertEqual(submit.call_args.args[:2], (101, "3"))
            self.assertEqual(
                json.loads((out / "predictions.json").read_text())["a"][0]["cls"],
                "Helmet",
            )
            self.assertEqual(
                json.loads((out / "state.json").read_text())["status"], "submitted"
            )
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Exam image changed"):
                benchmark.predict_submit(config, Mock(), yolo)


if __name__ == "__main__":
    unittest.main()
