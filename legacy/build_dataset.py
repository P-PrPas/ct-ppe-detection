"""Clean a Roboflow YOLO export into one flat project dataset (copies; the source is untouched).
- renames classes to project names via classes.yaml `sources[<name>]` (key = source class name or id; null = drop)
- keeps only project classes present, ids compacted in classes.yaml order
- merges train/valid/test into images/ + labels/, drops images with no kept box
usage: python3 -I build_dataset.py <classes.yaml> <source-name> <export-dir> <dest-dir>"""

import collections
import pathlib
import shutil
import sys

import yaml

cfg_p, src, exp, dest = (
    sys.argv[1],
    sys.argv[2],
    pathlib.Path(sys.argv[3]),
    pathlib.Path(sys.argv[4]),
)
cfg = yaml.safe_load(open(cfg_p))
y = next(exp.rglob("data.yaml"))
root = y.parent
src_names = yaml.safe_load(open(y))["names"]
src_names = dict(enumerate(src_names)) if isinstance(src_names, list) else src_names
mp = cfg["sources"][src]
to_proj = {
    i: mp.get(i, mp.get(n)) for i, n in src_names.items()
}  # source id -> project name / None
missing = [n for i, n in src_names.items() if i not in mp and n not in mp]
assert not missing, f"unmapped source classes: {missing}"
used = {n for n in to_proj.values() if n}
names = {i: n for i, n in enumerate(n for n in cfg["names"].values() if n in used)}
to_id = {n: i for i, n in names.items()}
assert not dest.exists(), f"{dest} exists"
(dest / "images").mkdir(parents=True)
(dest / "labels").mkdir()
cnt, kept_img, dropped_img = collections.Counter(), 0, 0
for lf in sorted(root.glob("*/labels/*.txt")):
    out = []
    for line in lf.read_text().splitlines():
        if not line.strip():
            continue
        c, *rest = line.split()
        n = to_proj[int(c)]
        if n:
            out.append(" ".join([str(to_id[n]), *rest]))
            cnt[n] += 1
    if not out:
        dropped_img += 1
        continue
    img = next(p for p in (lf.parent.parent / "images").glob(lf.stem + ".*"))
    assert not (dest / "images" / img.name).exists(), f"name clash {img.name}"
    shutil.copy2(img, dest / "images" / img.name)
    (dest / "labels" / lf.name).write_text("\n".join(out) + "\n")
    kept_img += 1
yaml.safe_dump(
    {"path": dest.name, "names": names},
    open(dest / "data.yaml", "w"),
    sort_keys=False,
    allow_unicode=True,
)
print(
    f"{src}: kept {kept_img} images, dropped {dropped_img} without our classes; boxes {dict((n, cnt[n]) for n in names.values())}"
)
