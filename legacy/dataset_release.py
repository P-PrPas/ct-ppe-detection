"""Frozen release construction: labels, connected families, and append policies."""

import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from core.validation import require

ROOT = Path(__file__).resolve().parents[1]
from core.validation import CLASSES


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def family(name):
    return (
        re.split(r"\.rf\.", name, flags=re.I)[0]
        if ".rf." in name.lower()
        else Path(name).stem
    )


def parse(path, names):
    lines = []
    omitted = Counter()
    edge_boxes = 0
    require(path.is_file(), path)
    for text in path.read_text().splitlines():
        if not text.strip():
            continue
        a = text.split()
        require(len(a) == 5, (path, text))
        c = int(a[0])
        require(0 <= c < len(names), (path, c, names))
        vals = list(map(float, a[1:]))
        require(all(math.isfinite(v) for v in vals), (path, text))
        x, y, w, h = vals
        require(
            0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1.00001 and 0 < h <= 1.00001,
            (
                path,
                text,
            ),
        )
        edge_boxes += (
            x - w / 2 < -0.00001
            or y - h / 2 < -0.00001
            or x + w / 2 > 1.00001
            or y + h / 2 > 1.00001
        )
        name = names[c]
        if name.lower().startswith("no_"):
            omitted[name] += 1
            continue
        require(name in CLASSES, (path, name))
        lines.append(str(CLASSES.index(name)) + " " + " ".join(a[1:]))
    return lines, omitted, edge_boxes


def connected_groups(rows):
    """Keep original union order: group roots determine seeded split assignment."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    hashes, families = {}, {}
    for index, row in enumerate(rows):
        for key, seen in [
            (row["image_sha256"], hashes),
            (row["original_family"], families),
        ]:
            if key in seen:
                left, right = find(index), find(seen[key])
                if left != right:
                    parent[right] = left
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    return groups


def video(name):
    match = re.match(r"(.+?_(?:mp4|avi|mov))[-_]\d+", name, re.I)
    if not match:
        raise ValueError(f"Unrecognized video frame: {name}")
    return match[1].lower()


def choose_append_split(previous, forced, quarantine, has_video, rng):
    require(
        not {"val", "test"}.issubset(previous),
        "New source bridges previous val/test groups; inspect before building",
    )
    if forced:
        require(
            "val" not in previous,
            "New benchmark family would invalidate fixed validation; inspect",
        )
        split = "test"
    elif "test" in previous:
        split = "test"
    elif "val" in previous:
        split = "val"
    elif quarantine:
        split = "test"
    elif "train" in previous:
        split = "train"
    elif has_video:
        split = "train"
    else:
        v = rng.random()
        split = "train" if v < 0.8 else ("val" if v < 0.9 else "test")
    require(
        not quarantine or split != "train",
        "Integrity check failed: not quarantine or split != 'train'",
    )
    return split


def build_append(base, out, report, policy):
    four_more = policy == "four-more"
    require(
        policy in {"four-more", "next-three"},
        "Integrity check failed: policy in {'four-more', 'next-three'}",
    )
    require(not out.exists(), "Refuse to overwrite a release")
    sources = json.loads((report / "source-snapshots.json").read_text())
    expected_projects = {83, 78, 79, 86} if four_more else {40, 75, 72}
    require(
        {source["project_id"] for source in sources} == expected_projects,
        "Integrity check failed: {source['project_id'] for source in sources} == expected_projects",
    )
    if four_more:
        review = json.loads((report / "cap83-reviewed-annotations.json").read_text())
        require(len(review) == 432, "Integrity check failed: len(review) == 432")
        quarantine_families = {
            f"83:{row['family']}"
            for row in review
            if row["decision"] != "include_baseball_cap"
        }
        require(
            len(quarantine_families) == 26,
            "Integrity check failed: len(quarantine_families) == 26",
        )
        decisions = {(row["label"], row["line"]): row["decision"] for row in review}
        protected_videos = set(
            json.loads((report / "source86-video-protection.json").read_text())[
                "benchmark_video_keys"
            ]
        )
    else:
        review = json.loads((report / "shoe-review-index.json").read_text())
        require(len(review) == 80, "Integrity check failed: len(review) == 80")
        quarantine_families = {f"72:{row['family']}" for row in review}
        require(
            len(quarantine_families) == 14,
            "Integrity check failed: len(quarantine_families) == 14",
        )
        protected_videos = set()
    expected_base = 18150 if four_more else 14536
    expected_total = 22709 if four_more else 18150
    rows = []
    for row in map(json.loads, (base / "manifest.jsonl").read_text().splitlines()):
        rows.append(
            dict(
                project_id=row["project_id"],
                source_image=row["image"],
                source_label=row["label"],
                original_family=row["group"],
                file_name=row["file_name"],
                previous_split=row["split"],
                prior_subset=row.get("subset", row["split"]),
                prior_quarantine=row.get("semantic_quarantine", False),
                is_base=True,
                source_names=CLASSES,
            )
        )
    require(
        len(rows) == expected_base, "Integrity check failed: len(rows) == expected_base"
    )
    for source in sources:
        p = Path(source["snapshot"])
        names = (p / "classes.txt").read_text().splitlines()
        require(
            names == source["classes"],
            "Integrity check failed: names == source['classes']",
        )
        images = sorted(x for x in (p / "images").rglob("*") if x.is_file())
        require(
            len(images) == source["images"],
            "Integrity check failed: len(images) == source['images']",
        )
        for image in images:
            rel = image.relative_to(p / "images")
            label = (p / "labels" / rel).with_suffix(".txt")
            rows.append(
                dict(
                    project_id=source["project_id"],
                    source_image=str(image),
                    source_label=str(label),
                    original_family=(
                        f"86:video:{video(image.name)}"
                        if source["project_id"] == 86
                        else f"{source['project_id']}:{family(image.name)}"
                    ),
                    file_name=image.name,
                    previous_split=None,
                    is_base=False,
                    source_names=names,
                )
            )
    require(
        len(rows) == expected_total,
        "Integrity check failed: len(rows) == expected_total",
    )
    omitted = Counter()
    edges = Counter()
    sizes = {}
    for i, row in enumerate(rows):
        names = row.pop("source_names")
        row["quarantine"] = (
            row.get("prior_quarantine", False)
            or f"{row['project_id']}:{family(row['file_name'])}" in quarantine_families
        )
        if not four_more and row["project_id"] == 72:
            names = [
                "no_ambiguous_shoe" if name == "not_safety_shoe" else name
                for name in names
            ]
        if four_more and row["project_id"] == 83 and not row["is_base"]:
            raw = Path(row["source_label"])
            prepared = report / "prepared-cap83-labels" / raw.name
            prepared.parent.mkdir(exist_ok=True)
            raw_lines = raw.read_text().splitlines()
            prepared.write_text(
                "".join(
                    line + "\n"
                    for n, line in enumerate(raw_lines, 1)
                    if line.strip()
                    and decisions[(str(raw), n)] == "include_baseball_cap"
                )
            )
            row["raw_source_label"] = str(raw)
            row["raw_source_label_sha256"] = digest(raw)
            row["source_label"] = str(prepared)
        lines, skip, edge = parse(Path(row["source_label"]), names)
        row["lines"] = lines
        omitted.update(skip)
        edges[row["project_id"]] += edge
        sha = digest(Path(row["source_image"]))
        row.update(
            image_sha256=sha, source_label_sha256=digest(Path(row["source_label"]))
        )
        if sha not in sizes:
            with Image.open(row["source_image"]) as image:
                sizes[sha] = image.size
                image.verify()
        row["width"], row["height"] = sizes[sha]
    groups = connected_groups(rows)
    exam = json.loads(
        (ROOT / "reports/andromeda-noleak-all-models/exam/members.json").read_text()
    )
    protected = {m["sha256"] for m in exam["members"]}
    index = json.loads((report / "benchmark-family-index.json").read_text())
    require(len(index) == 3783, "Integrity check failed: len(index) == 3783")
    protected_families = {row["original_family"] for row in index}
    rng = random.Random(42)
    keys = sorted(groups)
    rng.shuffle(keys)
    manifest = []
    counts = Counter()
    physical_counts = Counter()
    support = defaultdict(Counter)
    sources_by_split = defaultdict(Counter)
    moves = []
    conflicts = []
    quarantine_count = 0
    for k in keys:
        items = groups[k]
        previous = {row["previous_split"] for row in items if row["previous_split"]}
        forced = any(
            row["image_sha256"] in protected
            or f"{row['project_id']}:{family(row['file_name'])}" in protected_families
            or (row["project_id"] == 86 and video(row["file_name"]) in protected_videos)
            for row in items
        )
        quarantine = any(row["quarantine"] for row in items)
        split = choose_append_split(
            previous,
            forced,
            quarantine,
            any(row["project_id"] == 86 for row in items),
            rng,
        )
        group_id = (
            "group-"
            + hashlib.sha256(
                "\n".join(sorted({row["original_family"] for row in items})).encode()
            ).hexdigest()[:20]
        )
        variants = defaultdict(set)
        for row in items:
            variants[row["image_sha256"]].add(
                hashlib.sha256("\n".join(sorted(row["lines"])).encode()).hexdigest()
            )
        conflicts.extend(
            dict(image_sha256=h, group=group_id, label_variants=len(v))
            for h, v in variants.items()
            if len(v) > 1
        )
        for row in items:
            core = row["is_base"] and row["prior_subset"] in ["val", "test"]
            if core:
                require(
                    row["previous_split"] == split,
                    "Integrity check failed: row['previous_split'] == split",
                )
            subset = split if split == "train" or core else "extra_" + split
            filename = (
                Path(row["source_image"]).name
                if row["is_base"]
                else f"{row['project_id']}_{Path(row['source_image']).name}"
            )
            image = out / "images" / subset / filename
            label = out / "labels" / subset / Path(filename).with_suffix(".txt")
            if image.exists():
                filename = (
                    hashlib.sha256(row["source_image"].encode()).hexdigest()[:10]
                    + "_"
                    + filename
                )
                image = out / "images" / subset / filename
                label = out / "labels" / subset / Path(filename).with_suffix(".txt")
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["source_image"], image)
            require(
                digest(image) == row["image_sha256"],
                "Integrity check failed: digest(image) == row['image_sha256']",
            )
            lines = row.pop("lines")
            label.write_text("\n".join(lines) + "\n" if lines else "")
            support[subset].update(int(l.split()[0]) for l in lines)
            counts[split] += 1
            physical_counts[subset] += 1
            sources_by_split[row["project_id"]][subset] += 1
            quarantine_count += quarantine
            if row["previous_split"] and row["previous_split"] != split:
                moves.append(
                    dict(
                        image=row["source_image"],
                        old=row["previous_split"],
                        new=split,
                        reason="connected source group protection",
                    )
                )
            manifest.append(
                dict(
                    **row,
                    image=str(image),
                    label=str(label),
                    label_sha256=digest(label),
                    split=split,
                    subset=subset,
                    group=group_id,
                    core_evaluation=core,
                    semantic_quarantine=quarantine,
                    benchmark_exact_match=row["image_sha256"] in protected,
                )
            )
    require(
        len(manifest) == expected_total,
        "Integrity check failed: len(manifest) == expected_total",
    )
    group_splits = defaultdict(set)
    hash_splits = defaultdict(set)
    for row in manifest:
        group_splits[row["group"]].add(row["split"])
        hash_splits[row["image_sha256"]].add(row["split"])
        require(
            row["split"] == "test" or row["image_sha256"] not in protected,
            "Integrity check failed: row['split'] == 'test' or row['image_sha256'] not in protected",
        )
        require(
            row["split"] == "test"
            or f"{row['project_id']}:{family(row['file_name'])}"
            not in protected_families,
            "Integrity check failed: row['split'] == 'test' or f\"{row['project_id']}:{family(row['file_name'])}\" not in protected_families",
        )
        require(
            row["split"] == "test"
            or row["project_id"] != 86
            or video(row["file_name"]) not in protected_videos,
            "Integrity check failed: row['split'] == 'test' or row['project_id'] != 86 or video(row['file_name']) not in protected_videos",
        )
        require(
            not row["semantic_quarantine"]
            or row["subset"].startswith("extra_")
            or row["core_evaluation"],
            "Integrity check failed: not row['semantic_quarantine'] or row['subset'].startswith('extra_') or row['core_evaluation']",
        )
    require(
        all(len(source) == 1 for source in group_splits.values())
        and all(len(source) == 1 for source in hash_splits.values()),
        "Integrity check failed: all((len(source) == 1 for source in group_splits.values())) and all((len(source) == 1 for source in hash_splits.values()))",
    )
    require(
        physical_counts["val"] == 1337 and physical_counts["test"] == 2544,
        "Integrity check failed: physical_counts['val'] == 1337 and physical_counts['test'] == 2544",
    )
    prior = [json.loads(l) for l in (base / "manifest.jsonl").read_text().splitlines()]
    for split in ["val", "test"]:
        require(
            sorted(
                (row["image_sha256"], row["label_sha256"])
                for row in prior
                if row.get("subset", row["split"]) == split
            )
            == sorted(
                (row["image_sha256"], row["label_sha256"])
                for row in manifest
                if row.get("subset", row["split"]) == split
            ),
            "Integrity check failed: sorted(((row['image_sha256'], row['label_sha256']) for row in prior if row.get('subset', row['split']) == split)) == sorted(((row['image_sha256'], row['label_sha256']) for row in manifest if row.get('subset', row['split']) == split))",
        )
    for split in ["train", "val", "test"]:
        require(
            set(support[split]) == set(range(11)),
            "Integrity check failed: set(support[split]) == set(range(11))",
        )
    for subset, n in physical_counts.items():
        require(
            len(list((out / "images" / subset).iterdir()))
            == len(list((out / "labels" / subset).iterdir()))
            == n,
            "Integrity check failed: len(list((out / 'images' / subset).iterdir())) == len(list((out / 'labels' / subset).iterdir())) == n",
        )
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            dict(
                path=str(out),
                train="images/train",
                val="images/val",
                test="images/test",
                names=CLASSES,
            ),
            sort_keys=False,
        )
    )
    summary = dict(
        total_images=expected_total,
        images=dict(physical_counts),
        logical_split_counts=dict(counts),
        classes=CLASSES,
        source_counts={str(k): dict(v) for k, v in sources_by_split.items()},
        box_support={
            source: {CLASSES[i]: v[i] for i in range(11)}
            for source, v in support.items()
        },
        group_leakage=0,
        exact_hash_split_leakage=0,
        benchmark_exact_leakage_train_val=0,
        benchmark_known_family_leakage_train_val=0,
        core_eval_hashes_and_labels_unchanged=True,
        core_validation_images=1337,
        core_test_images=2544,
        semantic_quarantine_images=quarantine_count,
        omitted_boxes=omitted,
        edge_boxes=dict(edges),
        exact_duplicate_label_conflicts=len(conflicts),
        base_split_moves=len(moves),
        base=str(base),
        new_snapshots=sources,
        recipe_note="Same YOLO26s960 recipe and primary validation/test as previous expansion; train additions only. Extra heldouts are retained separately and not used for early stopping or primary test.",
        limitations=[
            "No exhaustive near-duplicate/scene or label-completeness audit.",
            "Partial-class source annotations can suppress unlabelled visible PPE; no automatic pseudo-labels or source repairs.",
            "All images retained; uncertain shoe and Cap families kept in extra heldout, not treated as negative training truth.",
        ],
    )
    if four_more:
        summary.update(
            cap83_quarantined_families=26,
            cap83_box_decisions=dict(Counter(row["decision"] for row in review)),
            benchmark_video86_protected_images=563,
            benchmark_video_leakage_train_val=0,
        )
    else:
        summary["semantic_quarantine_source_families"] = 14
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "split-moves.json").write_text(json.dumps(moves, indent=2) + "\n")
    (out / "duplicate-label-conflicts.json").write_text(
        json.dumps(conflicts, indent=2) + "\n"
    )
    (report / "build-state.json").write_text(
        json.dumps(
            dict(
                status="complete",
                release=str(out),
                images=dict(physical_counts),
                total=expected_total,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
