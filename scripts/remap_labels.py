"""Rename an in-place YOLO dataset's classes to project names using dataset/classes.yaml.
data.yaml gets only the project classes present in the dataset (ids are dataset-local).
usage: python3 -I remap_labels.py <classes.yaml> <source-name> <dataset-dir>
Run once per dataset: it overwrites labels and data.yaml (not idempotent)."""
import sys, pathlib, yaml

cfg, src, root = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
cfg = yaml.safe_load(open(cfg))
# keep only project classes this dataset actually has, in classes.yaml order, ids compacted to 0..k-1
src_map = {s: n for s, n in cfg["sources"][src].items() if n is not None}
names = {i: n for i, n in enumerate(n for n in cfg["names"].values() if n in src_map.values())}
to_id = {n: i for i, n in names.items()}
m = {s: to_id[n] for s, n in src_map.items()}

kept = dropped = 0
for f in root.glob("labels/**/*.txt"):  # flat or per-split
    out = []
    for line in f.read_text().splitlines():
        c, *rest = line.split()
        if int(c) in m:
            out.append(" ".join([str(m[int(c)]), *rest])); kept += 1
        else:
            dropped += 1
    f.write_text("\n".join(out) + ("\n" if out else ""))  # empty file = background image

yaml.safe_dump({"path": root.name, "names": names}, open(root / "data.yaml", "w"), sort_keys=False, allow_unicode=True)
print(f"kept {kept}, dropped {dropped}")
