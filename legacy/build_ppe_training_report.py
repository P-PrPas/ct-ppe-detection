"""Build an offline branded report from frozen training and test artifacts."""

import base64
import csv
import html
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ppe-training-20261109"
AS = OUT / "assets"
AS.mkdir(exist_ok=True)
DATA = ROOT / "data/ppe-full-current-20261109"
RUN = ROOT / "runs/ppe-full-current-20261109"
REF = Path("/root/P/voice-assistance-ai/docs/reports/_build")
TEAL = "#00a2a2"
NAVY = "#0e1b2d"
GREY = "#6b7686"
LIGHT = "#e9f6f6"
CORAL = "#e17b55"
BLUE = "#4779b5"
for w in ["Regular", "Bold"]:
    p = Path("/usr/local/share/fonts/plex") / f"IBMPlexSansThai-{w}.ttf"
    font_manager.fontManager.addfont(str(p))
    shutil.copy2(p, AS / p.name)
plt.rcParams.update(
    {
        "font.family": "IBM Plex Sans Thai",
        "font.size": 10,
        "text.color": NAVY,
        "axes.labelcolor": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "axes.edgecolor": "#d6dfe5",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
)
for n in ["logo.png", "strip.png"]:
    if (REF / n).exists():
        shutil.copy2(REF / n, AS / n)
E = json.loads((OUT / "evaluation.json").read_text())
OLD = json.loads((DATA / "test-metrics.json").read_text())
S = json.loads((DATA / "summary.json").read_text())
C = S["classes"]
N = len(C)
Z = np.load(OUT / "curves.npz")
PC = E["per_class"]
M = Z["confusion"]
assert M.shape == (12, 12)
# Keep original test result as the canonical score; report rerun only exports curves and illustrative predictions.
for key, v in OLD["overall"].items():
    if key.startswith("metrics/"):
        assert abs(v - E["overall"][key]) < 0.015, (key, v, E["overall"][key])
H = [
    {k.strip(): float(v) for k, v in r.items()}
    for r in csv.DictReader((RUN / "results.csv").open())
]
assert (
    len(H) == 88 and int(max(H, key=lambda r: r["metrics/mAP50-95(B)"])["epoch"]) == 68
)
palette = [
    TEAL,
    BLUE,
    CORAL,
    "#9b6bb5",
    "#d4aa35",
    "#5c9e60",
    "#cc6b8f",
    "#5966a6",
    "#668c91",
    "#b48d74",
    "#7b8794",
]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(AS / (name + ".png"), dpi=165, bbox_inches="tight")
    plt.close(fig)


def chart_note(ax):
    ax.axvline(68, color=TEAL, ls="--", lw=1, alpha=0.6)
    ax.axvline(78, color=GREY, ls=":", lw=1, alpha=0.7)


fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, key in zip(axes, ["box_loss", "cls_loss", "dfl_loss"]):
    for prefix, color in [("train", TEAL), ("val", CORAL)]:
        ax.plot(
            [r["epoch"] for r in H],
            [r[f"{prefix}/{key}"] for r in H],
            label=prefix,
            color=color,
            lw=1.8,
        )
    chart_note(ax)
    ax.set(xlabel="Epoch", ylabel="Loss", title=key.replace("_", " ").title())
    ax.legend()
save(fig, "training-loss")
fig, ax = plt.subplots(figsize=(11, 3.5))
for k, label, col in [
    ("precision", "Precision", TEAL),
    ("recall", "Recall", CORAL),
    ("mAP50", "mAP50", BLUE),
    ("mAP50-95", "mAP50–95", "#9b6bb5"),
]:
    ax.plot(
        [r["epoch"] for r in H],
        [r[f"metrics/{k}(B)"] for r in H],
        label=label,
        color=col,
    )
chart_note(ax)
ax.set(xlabel="Epoch", ylabel="Validation score", ylim=(0, 1))
ax.legend(ncol=4, loc="lower right")
save(fig, "training-metrics")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, key, title in zip(
    axes.flat,
    ["p", "r", "f1", "pr"],
    [
        "Precision vs confidence",
        "Recall vs confidence",
        "F1 vs confidence",
        "Precision–Recall",
    ],
):
    for i, c in enumerate(C):
        ax.plot(Z["px"], Z[key][i], color=palette[i], alpha=0.5, lw=0.9)
    ax.plot(Z["px"], Z[key].mean(0), color=NAVY, lw=2.5, label="Macro average")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall" if key == "pr" else "Confidence",
        ylabel="Precision" if key == "pr" else title.split()[0],
        title=title,
    )
    ax.legend()
save(fig, "test-curves")
fig, ax = plt.subplots(figsize=(12, 5.5))
y = np.arange(N)
w = 0.18
for off, k, lab, col in [
    (-1.5, "precision", "Precision", TEAL),
    (-0.5, "recall", "Recall", CORAL),
    (0.5, "f1", "F1", BLUE),
    (1.5, "mAP50_95", "mAP50–95", "#9b6bb5"),
]:
    ax.barh(y + off * w, [p[k] for p in PC], height=w, color=col, label=lab)
ax.set_yticks(y, C)
ax.invert_yaxis()
ax.set(xlim=(0, 1.05), xlabel="Score")
ax.legend(ncol=4, loc="lower right")
save(fig, "class-metrics")
for norm, name in [(False, "confusion-counts"), (True, "confusion-normalized")]:
    a = M / (M.sum(0, keepdims=True) + 1e-9) if norm else M
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(
        a,
        cmap=colors.LinearSegmentedColormap.from_list(
            "brand", ["#ffffff", LIGHT, TEAL, NAVY]
        ),
        vmin=0,
        vmax=1 if norm else None,
    )
    for i in range(12):
        for j in range(12):
            v = a[i, j]
            if v:
                ax.text(
                    j,
                    i,
                    f"{v:.0%}" if norm else f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > (0.5 if norm else a.max() * 0.55) else NAVY,
                )
    ax.set_xticks(range(12), C + ["Background"], rotation=55, ha="right")
    ax.set_yticks(range(12), C + ["Background"])
    ax.set(
        xlabel="Ground truth",
        ylabel="Prediction",
        title="Normalized by ground-truth column"
        if norm
        else "Confusion matrix • counts",
    )
    fig.colorbar(im, ax=ax, shrink=0.7)
    save(fig, name)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
width = 0.25
x = np.arange(11)
for i, s in enumerate(["train", "val", "test"]):
    axes[0].bar(
        x + (i - 1) * width,
        [S["box_support"][s][c] for c in C],
        width,
        label=s,
        color=[TEAL, BLUE, CORAL][i],
    )
axes[0].set_xticks(x, C, rotation=65, ha="right")
axes[0].set(
    yscale="log", ylabel="Annotated boxes (log scale)", title="Class support by split"
)
axes[0].legend()
axes[1].barh(C, M[-1, :-1], label="Missed GT (FN)", color=CORAL)
axes[1].barh(C, -M[:-1, -1], label="Unmatched predictions (FP)", color=TEAL)
axes[1].axvline(0, color=GREY, lw=0.8)
axes[1].set(title="Unmatched detections at conf 0.25", xlabel="FP ← Count → FN")
axes[1].legend(fontsize=8)
save(fig, "support-errors")
fig, ax = plt.subplots(figsize=(10, 3))
splits = ["train", "val", "test"]
pos = [S["images"][s] - S["negative_images"][s] for s in splits]
neg = [S["negative_images"][s] for s in splits]
ax.barh(splits, pos, color=TEAL, label="With positive boxes")
ax.barh(splits, neg, left=pos, color="#b8c1cc", label="Empty positive labels")
for i, s in enumerate(splits):
    ax.text(S["images"][s] + 80, i, f"{S['images'][s]:,}", va="center")
ax.set(xlim=(0, 9300), xlabel="Images")
ax.legend()
save(fig, "split-images")
# Predictions from report evaluation; exact duplicate rows can arise from repeated save_txt appends.
manifest = [json.loads(l) for l in (DATA / "manifest.jsonl").read_text().splitlines()]
test = sorted(
    (r for r in manifest if r["split"] == "test"), key=lambda r: r["file_name"]
)
predroot = OUT / "test-evaluation/labels"


def readboxes(p):
    if not p.exists():
        return np.empty((0, 6))
    lines = list(dict.fromkeys(l for l in p.read_text().splitlines() if l.strip()))
    a = np.array([[float(v) for v in l.split()] for l in lines])
    return a


def iou(a, b):
    lt = np.maximum(a[:2], b[:2])
    rb = np.minimum(a[2:], b[2:])
    inter = np.maximum(rb - lt, 0).prod()
    return inter / (
        np.maximum(a[2:] - a[:2], 0).prod()
        + np.maximum(b[2:] - b[:2], 0).prod()
        - inter
        + 1e-9
    )


def xyxy(a):
    return np.array(
        [a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2]
    )


items = []
for r in test:
    gt = readboxes(Path(r["label"]))
    p = readboxes(predroot / (Path(r["image"]).stem + ".txt"))
    if len(p):
        p = p[p[:, 5] >= 0.25]
        p = p[np.argsort(-p[:, 5])]
    matched = set()
    tp = 0
    for box in p:
        opts = [
            (iou(xyxy(box), xyxy(b)), i)
            for i, b in enumerate(gt)
            if i not in matched and int(b[0]) == int(box[0])
        ]
        score, j = max(opts, default=(0, None))
        if score >= 0.5:
            matched.add(j)
            tp += 1
    items.append(dict(row=r, gt=gt, pred=p, tp=tp, fp=len(p) - tp, fn=len(gt) - tp))
assert len(items) == 1003
choices = []
used = set()


def choose(label, key, eligible=lambda i: True):
    candidates = [i for i in items if i["row"]["file_name"] not in used and eligible(i)]
    if candidates:
        v = max(candidates, key=key)
        used.add(v["row"]["file_name"])
        choices.append((label, v))


choose("Many correct detections", lambda i: i["tp"], lambda i: i["tp"] > 0)
choose("Most missed annotations", lambda i: i["fn"])
choose("Most unmatched predictions", lambda i: i["fp"])
choose(
    "Negative frame with predictions",
    lambda i: i["fp"],
    lambda i: len(i["gt"]) == 0 and i["fp"] > 0,
)
choose(
    "Negative frame without predictions",
    lambda i: 0,
    lambda i: len(i["gt"]) == 0 and i["fp"] == 0,
)
choose(
    "Rare class • Cap",
    lambda i: i["tp"],
    lambda i: any(int(b[0]) == 9 for b in i["gt"]),
)
choose("Dense full-frame scene", lambda i: len(i["gt"]))
choose(
    "Ear protection example",
    lambda i: i["fn"],
    lambda i: any(int(b[0]) == 10 for b in i["gt"]),
)
font = ImageFont.truetype("/usr/local/share/fonts/plex/IBMPlexSansThai-Regular.ttf", 15)
bold = ImageFont.truetype("/usr/local/share/fonts/plex/IBMPlexSansThai-Bold.ttf", 18)
sample_meta = []
for idx, (reason, v) in enumerate(choices):
    image = Image.open(v["row"]["image"]).convert("RGB")
    image.thumbnail((600, 530))
    w, h = image.size
    panels = []
    for title, boxes, pred in [
        ("GROUND TRUTH", v["gt"], False),
        ("PREDICTIONS • conf ≥ 0.25", v["pred"], True),
    ]:
        panel = Image.new("RGB", (620, h + 70), "white")
        panel.paste(image, (10, 50))
        d = ImageDraw.Draw(panel)
        d.text((12, 10), title, font=bold, fill=NAVY)
        for b in boxes:
            cls = int(b[0])
            a = xyxy(b) * [w, h, w, h]
            a += np.array([10, 50, 10, 50])
            col = palette[cls]
            d.rectangle(tuple(a), outline=col, width=3)
            text = C[cls] + (f" {b[5]:.2f}" if pred else "")
            bb = d.textbbox((0, 0), text, font=font)
            tw, th = bb[2], bb[3] + 4
            lx = max(10, min(a[0], 620 - tw - 6))
            ly = max(48, a[1] - th)
            d.rectangle((lx, ly, lx + tw + 6, ly + th), fill=col)
            d.text((lx + 3, ly - 1), text, font=font, fill="white")
        panels.append(panel)
    canvas = Image.new("RGB", (1240, h + 70), "white")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (620, 0))
    name = f"example-{idx + 1:02}.jpg"
    canvas.save(AS / name, quality=90)
    sample_meta.append(
        dict(
            asset=name,
            selection=reason,
            source=v["row"]["file_name"],
            group=v["row"]["group"],
            tp=v["tp"],
            fp=v["fp"],
            fn=v["fn"],
            gt=len(v["gt"]),
            pred=len(v["pred"]),
        )
    )
(OUT / "examples.json").write_text(json.dumps(sample_meta, indent=2) + "\n")
with (OUT / "per-image-errors.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["filename", "tp", "fp", "fn", "gt_boxes", "predictions_conf_025"])
    for v in items:
        w.writerow(
            [
                v["row"]["file_name"],
                v["tp"],
                v["fp"],
                v["fn"],
                len(v["gt"]),
                len(v["pred"]),
            ]
        )
with (OUT / "per-class-metrics.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "class",
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50_95",
            "test_boxes",
            "test_images",
        ]
    )
    for p, old in zip(PC, OLD["per_class"]):
        w.writerow(
            [
                p["name"],
                p["precision"],
                p["recall"],
                p["f1"],
                p["mAP50"],
                p["mAP50_95"],
                old["boxes"],
                old["images"],
            ]
        )
np.savetxt(
    OUT / "confusion-matrix.csv",
    M,
    delimiter=",",
    fmt="%d",
    header=",".join(C + ["Background"]),
)
# Shared semantic tokens extracted from reference report; no external runtime or image requests.
CSS = """@font-face{font-family:Plex;src:url(assets/IBMPlexSansThai-Regular.ttf)}@font-face{font-family:Plex;src:url(assets/IBMPlexSansThai-Bold.ttf);font-weight:700}:root{--brand-teal:#00a2a2;--brand-navy:#0e1b2d;--brand-grey:#6b7686;--brand-pale:#f7f8fb;--color-primary:var(--brand-teal);--color-text:var(--brand-navy);--color-muted:var(--brand-grey);--color-border:#e9edf3;--color-tint:#e9f6f6;--font-report:Plex,'IBM Plex Sans Thai',sans-serif;--card-radius:12px}*{box-sizing:border-box}body{font-family:var(--font-report);color:var(--color-text);line-height:1.7;margin:0;background:white}main{max-width:1100px;margin:32px auto;padding:0 22px}h1{font-size:36px;line-height:1.25;margin:12px 0}h2{font-size:24px;color:var(--color-primary);margin:42px 0 12px}h3{font-size:18px}p{margin:12px 0}header{border-bottom:3px solid var(--color-primary);padding-bottom:24px}header img{width:150px}.eyebrow{color:var(--color-primary);font-size:13px;letter-spacing:2px;font-weight:700}.muted,figcaption{color:var(--color-muted);font-size:13px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.kpi{padding:16px;background:var(--color-tint);border-radius:var(--card-radius)}.kpi strong{font-size:30px;display:block;line-height:1.4}.callout{border-left:4px solid var(--color-primary);background:var(--color-tint);padding:14px 18px;border-radius:0 10px 10px 0}.warning{border-left-color:#e17b55;background:#fff5ee}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{border:1px solid #d6dfe5;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:var(--color-tint)}tbody tr:nth-child(even){background:var(--brand-pale)}figure{margin:24px 0}figure img{width:100%;height:auto;border:1px solid var(--color-border);border-radius:8px}figcaption{padding:8px 2px}a{color:var(--color-primary)}nav{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin-top:22px}code{font-size:12px;overflow-wrap:anywhere}details{border:1px solid var(--color-border);padding:12px;border-radius:8px}button,select,input{font:inherit}button,select{border:1px solid #d6dfe5;background:var(--color-tint);border-radius:6px;padding:6px 10px}.matrix-button{margin-right:8px}.table-wrap{overflow-x:auto}.sample{break-inside:avoid}.footer{border-top:1px solid var(--color-border);margin-top:40px;padding:20px 0;font-size:12px;color:var(--color-muted)}@media(max-width:650px){main{padding:0 14px}h1{font-size:28px}.grid{grid-template-columns:repeat(2,1fr)}table{font-size:11px}}@media print{@page{size:A4;margin:17mm}main{margin:0;padding:0;max-width:none}h1{font-size:28px}h2{break-after:avoid}figure,.callout,.kpi,tr{break-inside:avoid}figure img{max-height:225mm;object-fit:contain}nav,.interactive,button,details{display:none}.grid{display:table;width:100%}.kpi{display:table-cell;width:25%}body{font-size:10pt}h2{font-size:17pt}table{font-size:8pt}h3{font-size:12pt}}"""
sections = []


def add(s):
    sections.append(s)


def para(s):
    add("<p>" + s + "</p>")


def heading(id, s):
    add(f'<h2 id="{id}">{s}</h2>')


def fig(name, caption):
    add(
        f'<figure><img loading="lazy" src="assets/{name}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'
    )


def table(head, rows):
    add(
        '<div class="table-wrap"><table><thead><tr>'
        + "".join("<th>" + html.escape(str(c)) + "</th>" for c in head)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(c)) + "</td>" for c in r)
            + "</tr>"
            for r in rows
        )
        + "</tbody></table></div>"
    )


heading("summary", "01 · สรุปผลและสถานะโมเดล")
para(
    "โมเดลล่าสุดคือ <b>YOLO11s pretrained → PPE 11 คลาส</b> ฝึกด้วยภาพเต็ม 10,030 ภาพ หยุดที่ epoch 88/100 ด้วย Early Stopping (patience 20) และเลือก <b>best.pt จาก epoch 68</b> ด้วย validation mAP50–95 โดยไม่ใช้ผล test เลือกโมเดล"
)
add(
    '<div class="grid">'
    + "".join(
        f'<div class="kpi">{label}<strong>{value:.3f}</strong><span class="muted">Held-out test · macro</span></div>'
        for label, value in [
            ("Precision", OLD["overall"]["metrics/precision(B)"]),
            ("Recall", OLD["overall"]["metrics/recall(B)"]),
            ("mAP50", OLD["overall"]["metrics/mAP50(B)"]),
            ("mAP50–95", OLD["overall"]["metrics/mAP50-95(B)"]),
        ]
    )
    + "</div>"
)
para(
    f"Macro F1 จากค่า F1 ของแต่ละคลาสในรอบ export = <b>{E['macro_f1']:.3f}</b> (เฉลี่ย F1 รายคลาส ไม่ใช่ harmonic mean ของ macro precision/recall) ผลทั้งหมดวัดกับ local held-out test 1,003 ภาพ และ current raw labels ที่ผู้ใช้ยอมรับ ไม่ใช่หลักฐานยืนยันความถูกต้องของ labels"
)
add(
    '<div class="callout warning"><b>สถานะ Andromeda:</b> ยังไม่มีผล benchmark ที่ส่งสำเร็จ โปรเจกต์ ppe-full-current-20261109 (ID 97) มี Benchmark set 950 ภาพตามที่ผู้ใช้ยืนยัน แต่ CLI definitions เป็น [] และ token ถูกปฏิเสธการ export testset ด้วย 403 “a token can export pool labels only” จึงไม่ถือว่าผล local test ในรายงานนี้เป็นคะแนน Andromeda</div>'
)
heading("data", "02 · ข้อมูล การแบ่งชุด และความเสี่ยง")
table(
    ["Split", "ภาพ", "กลุ่มต้นฉบับ", "ภาพ label ว่าง", "Positive boxes"],
    [
        [
            s,
            f"{S['images'][s]:,}",
            S["groups"][s],
            S["negative_images"][s],
            f"{sum(S['box_support'][s].values()):,}",
        ]
        for s in ["train", "val", "test"]
    ],
)
fig(
    "split-images.png",
    "ใช้ครบ 10,030 ภาพตาม source keys ที่ได้รับอนุญาต รวมภาพสินค้าและภาพ label ว่าง ไม่มี person-crop filtering",
)
para(
    "ต้นทางคือ safety-working 9,720 ภาพ และ cap-pavithrap 310 ภาพ คัดลอก source images และอ่าน labels ปัจจุบันจาก /opt/mount/project/data เท่านั้น ใช้ historical draft manifest เฉพาะ source keys และ group ของ original/augmentation รวม manual Cap duplicate groups ไม่ใช้ labels เก่า ใน source folder มีไฟล์เพิ่มนอก 10,030 keys นี้ ซึ่งไม่ใช่ scope release ที่ได้รับอนุญาต"
)
para(
    "แยกตามกลุ่มประมาณ 80/10/10 ด้วย seed 42 และตรวจว่าไม่มี group ข้าม split จำนวน 4,294 กลุ่ม (train 3,449 / val 425 / test 420) มี positive boxes ครบทุกคลาสในทั้งสามชุด การตรวจนี้ยืนยันตาม group ที่บันทึกไว้เท่านั้น ไม่ใช่การพิสูจน์ว่าไม่มีภาพคล้ายหรือ near-duplicate ข้ามกลุ่มทุกชนิด"
)
para(
    "ลำดับคลาสสุดท้าย: "
    + ", ".join(f"{i} {c}" for i, c in enumerate(C))
    + ". ตัดเฉพาะกล่อง no_* ออกจาก model labels แต่เก็บภาพไว้ทั้งหมด ไม่มี label audit/repair รอบใหม่ตามคำสั่งผู้ใช้ Shoes หมายถึงรองเท้าปิดส้น และ Ear_Protection รวม headphones ตามนิยามที่ผู้ใช้กำหนด"
)
para(
    "Local test มี positive-labeled 950 ภาพ และ label ว่าง 53 ภาพ รวม 1,003 ภาพ จำนวน 950 สอดคล้องกับ Benchmark set ที่ผู้ใช้เห็น แต่ยังไม่ได้เทียบสมาชิกและ annotations กับ Andromeda โดยตรง จึงไม่ถือว่าทั้งสองชุดเหมือนกันแน่นอน"
)
fig(
    "support-errors.png",
    "ซ้าย: annotation support แบบ log scale • ขวา: FN/FP ที่ไม่ได้จับคู่ใน confusion matrix (conf 0.25, matching IoU 0.50) ข้ามคลาสจะแสดงในช่อง off-diagonal แยกต่างหาก",
)
heading("setup", "03 · การตั้งค่าและสภาพแวดล้อม")
table(
    ["รายการ", "ค่าที่ใช้"],
    [
        [
            "Base model",
            "models/yolo11s.pt; fine-tuned 11 classes; 9,432,049 parameters",
        ],
        [
            "Hardware",
            "Tesla V100-PCIE-16GB, physical GPU 1; CUDA_VISIBLE_DEVICES=1 → logical device 0",
        ],
        [
            "Runtime",
            "Python 3.11 · torch 2.11.0+cu126 · torchvision 0.26.0+cu126 · ultralytics 8.4.208",
        ],
        [
            "Training",
            "imgsz 640 · batch 12 · epochs target 100 · patience 20 · seed 42 · workers 4 · AMP",
        ],
        [
            "Optimizer",
            "auto resolved to MuSGD; initial lr 0.01; momentum 0.9; weight_decay 0.0005 config",
        ],
        [
            "Bounded augmentation",
            "mosaic 0.5, close_mosaic 10, scale 0.3, translate 0.1, degrees 5, fliplr 0.5; mixup/copy_paste 0",
        ],
        ["HSV", "h 0.015 / s 0.5 / v 0.3"],
        [
            "Resume",
            "Paused after epoch 78 due stale quota event; resumed epoch 79 from epoch77.pt with optimizer state",
        ],
        [
            "Stop / selected model",
            "Stop epoch 88; best epoch 68; selection by validation mAP50–95 0.35591",
        ],
        [
            "Recorded training time",
            "7,394.12 s initial segment + 1,068.06 s resumed segment = 8,462.18 s (~2.35 h); excludes idle pause and final test",
        ],
        [
            "Reporting rerun",
            "Frozen best.pt, test split, image 640, batch 8, NMS IoU 0.7, max_det 300; no training changes",
        ],
    ],
)
heading("learning", "04 · Learning curves และการเลือก checkpoint")
fig(
    "training-loss.png",
    "เส้นประแนวตั้ง: best epoch 68 • เส้นจุด: จุดพัก epoch 78 • Loss ฝั่ง train กับ val ใช้บริบท augmentation ต่างกัน จึงไม่ตีความ gap เป็น generalization error โดยตรง",
)
fig(
    "training-metrics.png",
    "Validation metrics ตลอด 88 epochs จาก results.csv ไม่มีการนำ test metric มาเลือก checkpoint; เวลาใน CSV เริ่มนับใหม่เมื่อ resume จึงรวมเวลาเป็นสองช่วง",
)
para(
    "Train loss ลดลงต่อในช่วงท้าย แต่ validation mAP50–95 ไม่เกินค่าสูงสุดหลัง epoch 68 จนหยุดที่ epoch 88 สิ่งนี้บ่งชี้ว่าเพิ่ม epochs ต่อไม่ได้เพิ่มคะแนน validation ในการทดลองนี้ ยังไม่เพียงพอที่จะระบุสาเหตุว่าเกิดจากข้อมูล, model capacity หรือ distribution โดยลำพัง"
)
heading("metrics", "05 · Precision / Recall / F1 / AP รายคลาส")
fig(
    "class-metrics.png",
    "คะแนน macro ต่อคลาสจาก report-only evaluation ของ best.pt บน local test; AP50–95 เฉลี่ย IoU 0.50 ถึง 0.95",
)
table(
    ["Class", "P", "R", "F1", "AP50", "AP50–95", "Boxes", "Images"],
    [
        [
            p["name"],
            *[
                f"{p[k]:.3f}"
                for k in ["precision", "recall", "f1", "mAP50", "mAP50_95"]
            ],
            old["boxes"],
            old["images"],
        ]
        for p, old in zip(PC, OLD["per_class"])
    ],
)
para(
    "Cap มี test support เพียง 30 กล่องใน 30 ภาพ คะแนนสูงจึงต้องอ่านร่วมกับจำนวนตัวอย่าง ส่วน recall ของ Gloves และ Safety_Belt ยังต่ำในชุดนี้ ภาพตัวอย่างด้านล่างใช้ตรวจ failure modes เท่านั้น ไม่ใช่การแก้ labels หรือสรุปเหตุผลเชิงสาเหตุของความผิดพลาด"
)
heading("curves", "06 · Confidence curves และการอ่าน F1")
fig(
    "test-curves.png",
    "เส้นบาง = 11 คลาส • เส้นเข้ม = ค่าเฉลี่ยคลาส • PR curve รวมผลหลาย confidence thresholds; AP คำนวณจาก PR ไม่ใช่ accuracy ของภาพ",
)
para(
    f"Ultralytics รายงาน P/R/F1 ที่จุดใกล้จุดสูงสุดของ smoothed macro F1 curve (confidence ประมาณ {E['diagnostic_confidence']:.3f} ใน report rerun) จุดนี้มาจาก test จึงใช้เพื่ออธิบายผลเท่านั้น ไม่เสนอเป็น deployment threshold การเลือก threshold ใช้งานจริงควรทำบน validation หรือ calibration set แยกต่างหาก"
)
add(
    '<div class="interactive callout"><b>สำรวจ confidence เพื่ออ่านผล</b> <select id="curveClass"><option value="all">Macro average</option>'
    + "".join(f'<option value="{i}">{c}</option>' for i, c in enumerate(C))
    + '</select> <input id="confidence" aria-label="Confidence threshold" type="range" min="0" max="999" value="250"><div id="curveReadout" aria-live="polite"></div><small>Diagnostic only — ไม่ได้บันทึกค่าเพื่อใช้งานจริง</small></div>'
)
heading("confusion", "07 · Confusion matrix และความผิดพลาด")
para(
    "แกนตั้งคือ prediction และแกนนอนคือ ground truth ช่องทแยงคือคลาสที่จับคู่ถูก แถว Background คือ ground-truth objects ที่ไม่พบ prediction จับคู่ (FN); คอลัมน์ Background คือ predictions ที่ไม่พบ ground truth จับคู่ (FP) ไม่ใช่จำนวนภาพ background ทั้งชุด Matrix ใช้ confidence 0.25 และ matching IoU 0.50 ซึ่งต่างจากจุด confidence ที่ใช้รายงาน macro P/R ข้างต้น"
)
add(
    "<div class=\"interactive\"><button class=\"matrix-button\" onclick=\"document.getElementById('matrix').src='assets/confusion-counts.png'\">Counts</button><button onclick=\"document.getElementById('matrix').src='assets/confusion-normalized.png'\">Normalized</button></div>"
)
add(
    '<figure><img id="matrix" src="assets/confusion-counts.png" alt="Confusion matrix counts"><figcaption>Raw counts · detection matching, class-agnostic confusion accounting</figcaption></figure>'
)
fig(
    "confusion-normalized.png",
    "Normalized ตามผลรวมแต่ละ ground-truth column รวมแถว Background; คอลัมน์ Background ใช้ผลรวม unmatched predictions เป็น denominator จึงอ่านต่างจาก recall ของคลาส",
)
heading("examples", "08 · Annotated examples: labels เทียบ predictions")
para(
    "เลือกตัวอย่างด้วยกฎคงที่จากผลทั้งหมด: ตรวจถูกจำนวนมาก, missed annotations มาก, unmatched predictions มาก, ภาพไม่มี GT ทั้งที่มี/ไม่มี prediction, rare class และภาพหนาแน่น ไม่ได้สุ่มอย่างเป็นตัวแทนทั้งชุด และไม่ใช่ค่าเฉลี่ย performance ด้านซ้ายแสดง current labels ด้านขวาแสดง best.pt ที่ conf ≥ 0.25, NMS IoU 0.70 ค่า TP/FP/FN ใน caption ใช้ class-aware greedy matching ที่ IoU ≥ 0.50 จึงอาจต่างจาก class-agnostic confusion matrix เมื่อทำนายผิดคลาส"
)
for sample in sample_meta:
    add('<div class="sample"><h3>' + html.escape(sample["selection"]) + "</h3>")
    fig(
        sample["asset"],
        f"TP {sample['tp']} · FP {sample['fp']} · FN {sample['fn']} · GT {sample['gt']} · predictions {sample['pred']}",
    )
    para(
        '<span class="muted">Source: '
        + html.escape(sample["source"])
        + "<br>Group: "
        + html.escape(sample["group"])
        + "</span>"
    )
    add("</div>")
heading("limits", "09 · ข้อจำกัดและการนำไปใช้")
para(
    "Labels ต้นทางเป็น authoritative ตามคำสั่งผู้ใช้ และไม่มี audit/repair เพิ่ม ดังนั้นภาพที่โมเดลไม่ตรงกับ labels อาจเกิดจาก model error, label error หรือทั้งสองอย่าง คะแนนที่รายงานวัดความสอดคล้องกับ annotations ชุดนี้ ไม่ใช่ข้อพิสูจน์ความถูกต้องด้านความปลอดภัยในสถานที่ทำงาน"
)
para(
    "ฝึกและประเมินบน full frames รวมภาพสินค้า/negatives แต่ deployment ตั้งใจให้ person detector + tracking สร้าง ROI ก่อน PPE inference การกระจายขนาดอุปกรณ์และบริบทจึงต่างกัน ยังไม่ได้ทดสอบ detector+tracker+PPE แบบ end-to-end, วิดีโอจริง, ความต่อเนื่องข้ามเฟรม หรืออัตราแจ้งเตือนต่อคน"
)
para(
    "Speed ที่ framework พิมพ์เป็นเวลา inference ต่อภาพเฉพาะ evaluation บน GPU และ batch ที่กำหนด ไม่รวม person detection, tracking, video I/O หรือ product logic รายงานนี้จึงไม่สรุป FPS ของระบบจริง และไม่เลือกปรับ threshold จาก test หลังเห็นคะแนน"
)
heading("reproduce", "10 · Reproducibility และหลักฐาน")
para(
    "เปิดรายงาน HTML นี้ได้แบบ offline; ภาพและฟอนต์ฝังใน standalone copy ส่วน PDF ใช้รูปแบบเดียวกัน ตัวเลขหลักอ้างจาก test-metrics.json เดิม กราฟและตัวอย่างมาจาก report-only rerun ซึ่งเก็บ metrics แยกเพื่อให้ตรวจสอบความต่างได้"
)
table(
    ["Artifact", "Location"],
    [
        ["Selected model", "runs/ppe-full-current-20261109/weights/best.pt"],
        [
            "Training settings / history",
            "runs/ppe-full-current-20261109/args.yaml; results.csv",
        ],
        [
            "Dataset provenance",
            "data/ppe-full-current-20261109/manifest.jsonl; summary.json; data.yaml",
        ],
        [
            "Original heldout metrics",
            "data/ppe-full-current-20261109/test-metrics.json",
        ],
        [
            "Report evaluation",
            "reports/ppe-training-20261109/evaluation.json; curves.npz; test-evaluation/",
        ],
        [
            "Machine-readable report tables",
            "per-class-metrics.csv; confusion-matrix.csv; per-image-errors.csv; examples.json",
        ],
        [
            "Brand reference",
            "voice-assistance-ai/docs/reports/_build/make_docx.py; make_cpu_threads.py; logo.png; strip.png",
        ],
    ],
)
para("SHA256 best.pt: <code>" + E["model_sha256"] + "</code>")
para(
    "สร้างข้อมูลประกอบซ้ำด้วย .venv-ppe-v100/bin/python legacy/collect_ppe_report_metrics.py แล้ว python legacy/build_ppe_training_report.py การรัน evaluator ซ้ำควรใช้โฟลเดอร์ output ใหม่เพื่อไม่ให้ save_txt append predictions ซ้ำ"
)
SCRIPT = """const curves=CURVEDATA;const q=document.getElementById('confidence'),s=document.getElementById('curveClass');function update(){const i=+q.value,cls=s.value;const vals=['p','r','f1'].map(k=>cls==='all'?curves[k].reduce((a,r)=>a+r[i],0)/curves[k].length:curves[k][+cls][i]);document.getElementById('curveReadout').textContent=`Confidence ${curves.px[i].toFixed(3)} · Precision ${vals[0].toFixed(3)} · Recall ${vals[1].toFixed(3)} · F1 ${vals[2].toFixed(3)}`;}q.oninput=update;s.onchange=update;update();""".replace(
    "CURVEDATA", json.dumps({k: Z[k].round(6).tolist() for k in ["px", "p", "r", "f1"]})
)
body = (
    '<header><img src="assets/logo.png" alt="Company logo"><div class="eyebrow">MODEL TRAINING REPORT · 09 NOV 2026</div><h1>PPE Detection<br>YOLO11s · Full Current Release</h1><p>รายงานการฝึกและประเมินโมเดล PPE ล่าสุด · 11 คลาส · 10,030 ภาพ</p><nav>'
    + "".join(
        f'<a href="#{i}">{t}</a>'
        for i, t in [
            ("summary", "Summary"),
            ("data", "Dataset"),
            ("learning", "Training"),
            ("metrics", "Metrics"),
            ("curves", "F1 / PR"),
            ("confusion", "Confusion matrix"),
            ("examples", "Examples"),
            ("reproduce", "Evidence"),
        ]
    )
    + "</nav></header>"
    + "".join(sections)
    + '<div class="footer">Internal evaluation report · Current labels accepted without further audit · Local test results; Andromeda benchmark not submitted</div>'
)
full = (
    '<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPE YOLO11s Training Report</title><style>'
    + CSS
    + "</style><main>"
    + body
    + "</main><script>"
    + SCRIPT
    + "</script></html>"
)
(OUT / "report.html").write_text(full)
# Standalone embeds all local assets, including the toggle target URLs inside JS.

standalone = full
for p in AS.iterdir():
    if p.suffix in [".png", ".jpg", ".ttf"]:
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".ttf": "font/ttf"}[p.suffix]
        uri = "data:" + mime + ";base64," + base64.b64encode(p.read_bytes()).decode()
        standalone = standalone.replace("assets/" + p.name, uri)
(OUT / "PPE-YOLO11s-Training-Report.html").write_text(standalone)
# Concise embedded presentation follows host theme variables and links to the complete artifact.
preview = (
    """<!doctype html><html lang="th"><meta charset="utf-8"><style>:root{--primary:#00a2a2;--font-sans:'IBM Plex Sans Thai',sans-serif}body{margin:0;color:var(--foreground);font-family:var(--font-sans);line-height:1.6}h2{color:var(--primary)}img{width:100%;height:auto;border-radius:8px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.card{background:var(--card);padding:16px;border-radius:var(--radius)}strong{display:block;font-size:23px}small{color:var(--muted-foreground)}@media(max-width:450px){.cards{grid-template-columns:repeat(2,1fr)}}@font-face{font-family:'IBM Plex Sans Thai';src:url('assets/IBMPlexSansThai-Regular.ttf')}</style><h2>PPE · YOLO11s Training Report</h2><p>Best epoch 68 · Early stop 88 · Held-out test 1,003 ภาพ</p><div class="cards">"""
    + "".join(
        f'<div class="card"><small>{l}</small><strong>{v:.3f}</strong></div>'
        for l, v in [
            ("Precision", 0.700776658),
            ("Recall", 0.507968207),
            ("mAP50", 0.514929364),
            ("mAP50–95", 0.31323319),
        ]
    )
    + """</div><p><small>Local test เท่านั้น · Andromeda benchmark ยังไม่ได้ส่ง</small></p><img src="assets/class-metrics.png" alt="Class metrics"><h2>Confusion matrix</h2><img src="assets/confusion-normalized.png" alt="Normalized confusion matrix"><h2>Annotated example</h2><img src="assets/example-02.jpg" alt="Labels compared with model predictions"></html>"""
)
(OUT / "preview.html").write_text(preview)
(OUT / "report-print.html").write_text(full.replace('<img loading="lazy"', "<img"))
(OUT / "design-provenance.json").write_text(
    json.dumps(
        dict(
            reference_repo=str(REF.parents[2]),
            reference_files=[
                str(REF / "make_docx.py"),
                str(REF / "make_cpu_threads.py"),
            ],
            font="IBM Plex Sans Thai",
            colors=dict(teal=TEAL, navy=NAVY, grey=GREY),
            scope="Reports only; reference repository read-only",
        ),
        indent=2,
    )
    + "\n"
)
print(
    "BUILT",
    OUT,
    "F1",
    E["macro_f1"],
    "threshold",
    E["diagnostic_confidence"],
    "examples",
    len(sample_meta),
    flush=True,
)
