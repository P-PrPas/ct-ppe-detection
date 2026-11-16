"""Annotate every video frame with local YOLO weights; save video and pixel-box JSONL."""

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="0", help="Visible CUDA index, or cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()
    assert args.source.is_file() and args.weights.is_file()
    assert 0 < args.conf <= 1 and args.imgsz > 0
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / "annotated.mp4"
    raw = args.out / "annotated-mp4v.mp4"
    assert not final.exists() and not raw.exists(), "Refuse to overwrite video"
    cap = cv2.VideoCapture(str(args.source))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # shortcut: render at average FPS; use timestamp-aware decoding if exact variable-frame timing matters.
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps > 0 and width > 0 and height > 0
    model = YOLO(str(args.weights))
    writer = cv2.VideoWriter(
        str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    counts = Counter()
    n = 0
    start = time.time()
    try:
        with (args.out / "predictions.jsonl").open("x") as output:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                )[0]
                boxes = []
                for xyxy, score, cls in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                ):
                    name = result.names[int(cls)]
                    counts[name] += 1
                    assert len(xyxy) == 4 and all(
                        isinstance(x, (int, float)) for x in xyxy
                    )
                    boxes.append(dict(cls=name, box=xyxy, conf=score))
                output.write(
                    json.dumps(dict(frame=n, time_seconds=n / fps, predictions=boxes))
                    + "\n"
                )
                annotated = result.plot(line_width=2, font_size=12)
                writer.write(annotated)
                if n in {0, expected // 2, max(0, expected - 1)}:
                    cv2.imwrite(str(args.out / f"frame-{n:06d}.jpg"), annotated)
                n += 1
                if n % 250 == 0:
                    print(
                        f"frames {n}/{expected}, elapsed {time.time() - start:.1f}s",
                        flush=True,
                    )
    finally:
        cap.release()
        writer.release()
    finalize(args, n, fps, width, height, counts, start)


def finalize(args, n, fps, width, height, counts, start=None):
    raw = args.out / "annotated-mp4v.mp4"
    final = args.out / "annotated.mp4"
    # Container frame counts may be estimates; independently decode before accepting complete output.
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "2",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(args.source),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    source_frames = int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"])
    assert n > 0 and n == source_frames, (n, source_frames)
    # Keep source audio if present; rendered video preserves every decoded frame and its FPS.
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-i",
            str(raw),
            "-i",
            str(args.source),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(final),
        ],
        check=True,
    )
    check = cv2.VideoCapture(str(final))
    assert check.isOpened() and int(check.get(cv2.CAP_PROP_FRAME_COUNT)) == n
    check.release()
    metadata = dict(
        source=str(args.source.resolve()),
        weights=str(args.weights.resolve()),
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        imgsz=args.imgsz,
        confidence=args.conf,
        device=args.device,
        frames=n,
        fps=fps,
        width=width,
        height=height,
        elapsed_seconds=(time.time() - start if start is not None else None),
        detection_frame_counts=dict(counts),
        output=str(final.resolve()),
        notes="Predictions, not ground truth or unique object counts. No tracking or person ROI. No accuracy/FPS deployment claim.",
    )
    (args.out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    raw.unlink()
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
