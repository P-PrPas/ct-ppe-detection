# PPE experiments

ให้ agent รับคำสั่ง **สร้าง dataset → train → submit** ผ่านคำสั่งกลาง โดยแก้รายละเอียดการทดลองใน config และใช้ implementation ใน `core/` ร่วมกัน

```text
ppe.py                  # ทางเข้าหลักเดียว: dataset / train / submit
core/
  datasets.py           # freeze source manifest เป็น YOLO release
  training.py           # train/resume/evaluate และ state
  benchmark.py          # predict/validate/submit/confirm
  validation.py         # class order, hashes, split และ release guards
configs/
  dataset/              # ข้อมูลและ output ของ dataset
  train/                # model, release, GPU และ training recipe
  submit/               # checkpoint, exam และ Andromeda target
legacy/                 # งานเฉพาะกิจเก่า ใช้เฉพาะเมื่อต้อง reproduce
tests/                 # CPU regression tests
AGENTS.md               # ลำดับทำงานสำหรับ agent
```

## คำสั่งที่ใช้ประจำ

```bash
python ppe.py dataset configs/dataset/example.json --dry-run
python ppe.py train configs/train/expanded-latest-l.json --dry-run
python ppe.py submit configs/submit/full-yolo26s.json --dry-run
```

เอา `--dry-run` ออกเมื่อต้องการรันจริง โหมดนี้แสดง config ที่ resolve แล้วเท่านั้น ไม่ตรวจว่าข้อมูลมีจริง ไม่เรียก CUDA ไม่ติดต่อ server และไม่เขียนผลลัพธ์

ส่งหลาย config ให้คำสั่งเดียวได้ จะทำตามลำดับและหยุดเมื่อมี error งาน train/submit ใน process เดียวกันต้องใช้ physical GPU เดียวกัน

Path ใน config อ้างจาก root ของ repo; `--root /path/to/workspace` เปลี่ยน root ของข้อมูลและ outputs ส่วน path ของไฟล์ config ที่ส่งให้ CLI อ้างจาก working directory ตามปกติ

## สร้าง dataset ใหม่

Copy `configs/dataset/example.json` แล้วระบุ:

- `manifest`: JSONL index ของภาพ/labels ที่ผ่านการคัดเลือก
- `out`: directory ใหม่ที่จะสร้าง release
- `expected_images`: จำนวนภาพที่คาดไว้
- `protected_exams`: รายการ `members.json` ของ benchmark ที่ต้องกันออกจาก train/val; ใส่ `[]` ได้เมื่อไม่มี benchmark ที่ต้องป้องกัน โดยผลลัพธ์จะบันทึกขอบเขตนี้ไว้

หนึ่งบรรทัดใน source manifest:

```json
{"image":"source/images/a.jpg","label":"source/labels/a.txt","group":"camera1-scene42","split":"train","names":["Helmet","Vest"]}
```

`image`/`label` อ้างจาก directory ของ manifest หรือใช้ absolute path; `names` คือ class order ของต้นทาง ถ้าไม่ระบุจะใช้ PPE 11 classes ใน `core/validation.py` ระบุ `image_sha256`/`label_sha256` เพิ่มได้เพื่อบังคับตรวจ frozen input

ต้องมีภาพใน `train`, `val`, `test` และระบุ family group ให้ถูกต้อง ภาพเดียวกันหรือ family เดียวกันห้ามข้าม split ตัว builder **ไม่สุ่ม split และไม่เดาว่าภาพมาจาก scene เดียวกัน**: agent ต้องกำหนดจาก provenance/ข้อกำหนดการทดลองก่อน หากเป็น benchmark family/video ให้ทั้งกลุ่มอยู่ test

Labels ต้องเป็น YOLO detection boxes; class ที่ไม่อยู่ใน PPE taxonomy จะถูกปฏิเสธเพื่อให้ review อย่างชัดเจน ไม่มีการทิ้ง negative classes หรือแก้ label อัตโนมัติ ภาพที่มีไฟล์ label ว่างยังเก็บไว้

ผลลัพธ์คือ `images/`, `labels/`, `data.yaml`, `manifest.jsonl`, `summary.json`, `build-config.json` สร้างใน staging แล้ว publish เมื่อผ่าน checks ทั้งหมด; ไม่แก้ source และปฏิเสธ output ที่มีอยู่แล้ว

การ append release เก่าที่ต้องคง core evaluation, extra heldouts หรือ quarantine เฉพาะแหล่งข้อมูล ใช้ policy ที่บันทึกใน `legacy/README.md` ก่อน Generic builder ไม่ใช่ตัวแทนของกฎ review เหล่านั้น

## เทรน

Copy recipe ที่ใกล้เคียงใน `configs/train/` แล้วเปลี่ยน `data`, `run`, `weights`, `gpu`, `train`, `eval_batch` และ `validation` ให้ตรงกับ release ตัวอย่าง validation สำหรับ release ใหม่ 3 ภาพ:

```json
{"images":3,"counts":{"train":1,"val":1,"test":1}}
```

สำหรับ release เดิมให้เก็บ guards ใน recipe เดิมครบ เช่น baseline split, frozen reference, core evaluation และ expanded summary อย่าลด guards เพียงเพื่อให้ข้อมูลที่เปลี่ยนผ่าน

ทุก experiment ใหม่ใช้ `run` ใหม่ หาก recipe มี `state`/`metrics` ระบุเองต้องเปลี่ยนด้วย การเรียก recipe เดิมซ้ำเป็น resume: ใช้ checkpoint ที่ยังมี optimizer state, ไม่เทรนซ้ำเมื่อ complete และ retry evaluation ได้โดยไม่เทรนใหม่

Artifacts หลัก:

- `resolved-recipe.json` และ `input-hashes.jsonl`: ค่ากับ inputs ที่ใช้จริง
- `training-state.json`: starting/training/evaluating/complete/failed
- `weights/best.pt`, `test-metrics.json`: checkpoint ที่เลือกด้วย validation และ heldout metrics
- Reviewed YOLO11 ยังมี baseline comparison บน test ชุดเดียวกัน

คง path metrics/state เก่าของ full/reviewed YOLO11 ใน configs เพื่อให้รายงานเดิมอ่านได้ การเทรนไม่เรียก submit อัตโนมัติ

## Submit

Copy `configs/submit/full-yolo26s.json` แล้วระบุ target, frozen `exam`, `weights`, output, candidate name, version, resolution และ GPU ให้ตรงกัน Output directory ต้องมี `preparation.json` ที่บันทึกการเตรียม exam เดิม

คำสั่งตรวจ definition, จำนวนภาพ, image hashes, classes, boxes และ sample coverage ก่อนส่ง ใช้ threshold เดิมที่ freeze ไว้ ไม่ปรับจากคะแนน benchmark โดย flow นี้รองรับ exam แบบ test-only ที่ min_conf/IoU 0.5 และ model classes อยู่ใน benchmark classes; customer-NVR/drop-unmapped และ cached multi-model workflow เดิมอยู่ใน legacy

เมื่อส่งสำเร็จต้องมี `confirmed-server-run.json`; receipt หรือ pending ที่ server ไม่พบเป็นสถานะกำกวม ต้อง reconcile ก่อน retry ห้ามลบ receipt แล้วส่งใหม่โดยยังไม่ตรวจ server

## เรียกจาก Python

CLI กับ agent ใช้ implementation เดียวกันได้โดยตรง:

```python
from core.datasets import build_release
from core.training import load_recipe, run_experiment
from core.benchmark import predict_submit
```

`build_release` รับ config ที่ resolve path แล้ว; `load_recipe` resolve training paths; `run_experiment`/`predict_submit` รับ Torch และ YOLO ที่ caller เตรียมไว้ จึงทดสอบโดยไม่ใช้ GPU ได้ ปกติให้ agent ใช้ CLI ซึ่งจัด GPU visibility ก่อน import Torch ให้แล้ว

## Environment และการตรวจงาน

ใช้ Python 3.11+ กับ environment GPU เดิมที่ผ่าน calibration `requirements.txt` มาจาก metadata ของ environment เดิม ไม่ใช่ผลทดสอบติดตั้งใหม่ PyTorch เดิมคือ `2.11.0+cu126` และต้องตรงกับ CUDA/driver

เครื่อง V100 เดิมตั้งค่าก่อนเรียก CLI:

```bash
export LD_LIBRARY_PATH="$PWD/.runtime/nvidia-580.95.05/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv-ppe-v100/bin/python -u ppe.py train configs/train/expanded-latest-l.json
```

`gpu` ระบุ physical GPU; CUDA device ใน trainer เป็น logical `0` ค่า batch และ memory guard เดิมยังอยู่ใน recipes ข้อมูล, weights, runs, reports และ secrets ไม่อยู่ใน Git

```bash
python -m unittest discover -s tests -v
python -O -m unittest discover -s tests -q
python -m compileall -q ppe.py core legacy
uv tool run ruff check ppe.py core legacy tests --select F,I
uv tool run ruff format --check ppe.py core legacy tests
```

Tests ใช้ภาพเล็กจริงและ fake model/server ไม่ยืนยัน GPU performance, คะแนนโมเดล หรือ full historical dataset rebuild

## ย้ายจาก PR รุ่นแรก

`python scripts/train.py configs/X.json` เปลี่ยนเป็น `python ppe.py train configs/train/X.json` การ submit เดี่ยวใช้ `ppe.py submit`; สคริปต์อื่นถูกย้ายไป `legacy/` และเรียกด้วย `python -m legacy.<ชื่อไฟล์>` ดูรายละเอียดใน `legacy/README.md`
