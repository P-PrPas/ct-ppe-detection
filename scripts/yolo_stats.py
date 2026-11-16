"""Class distribution of YOLO exports (Roboflow layout: x/{train,valid,test}/{images,labels}, data.yaml).
Prints per class: boxes in export, and boxes counting one copy per source image (Roboflow augmentation copies share the name before `.rf.`).
usage: python3 -I yolo_stats.py <dir> ..."""
import sys, pathlib, collections, yaml, re
for d in map(pathlib.Path, sys.argv[1:]):
    y = next(d.rglob("data.yaml")); root = y.parent
    names = yaml.safe_load(open(y))["names"]; names = dict(enumerate(names)) if isinstance(names, list) else names
    tot, uniq, seen, bg, files, per = collections.Counter(), collections.Counter(), set(), 0, 0, collections.Counter()
    for f in sorted(root.glob("*/labels/*.txt")):
        files += 1; per[f.parent.parent.name] += 1
        cs = [names[int(l.split()[0])] for l in f.read_text().splitlines() if l.strip()]
        bg += not cs; tot.update(cs)
        base = re.sub(r"\.rf\.[0-9a-f]+$", "", f.stem)
        if base not in seen: seen.add(base); uniq.update(cs)
    print(f"## {d.name}: {files} label files {dict(per)}, {len(seen)} source images, {bg} background")
    for c in names.values(): print(f"  {c:28s} {tot[c]:7d}  (1 copy/source: {uniq[c]})")
