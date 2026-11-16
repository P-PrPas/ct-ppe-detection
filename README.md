# PPE Detection

โค้ดเทรนและประเมิน PPE ด้วย Ultralytics YOLO11/YOLO26 ใช้ **`scripts/train.py` ไฟล์เดียว** สำหรับทุก experiment ที่มีใน repo

## เริ่มอ่านจากตรงไหน

- `configs/*.json` — dataset, weights, GPU, batch, resolution และค่าการเทรน แก้ที่นี่เมื่อต้องการเปลี่ยน experiment
- `scripts/train.py` — ตรวจ recipe → ตรวจข้อมูล → train/resume → heldout evaluation → บันทึกสถานะ
- `scripts/release_validation.py` — class order, frozen hashes, split membership และ release guards
- `scripts/dataset_release.py` — การอ่าน label, รวม family/ภาพซ้ำ และสร้าง release ที่เพิ่มข้อมูลโดยคง core evaluation
- `scripts/benchmark_submission.py` — ตรวจ predictions และยืนยันผลส่งจาก Andromeda ก่อน retry
- `tests/test_workflows.py` — ทดสอบด้วย fixture เล็กและ fake YOLO โดยไม่ใช้ GPU/ส่งข้อมูลขึ้น server

สคริปต์เตรียมข้อมูล/รายงานอื่นอยู่ใน `scripts/` ชื่อเดิม โค้ดถูกจัดรูปแบบให้คำสั่งและ control flow แยกบรรทัด อ่านและแก้ด้วยมือได้

## Environment และข้อมูล

ใช้ Python 3.11+ และ environment GPU เดิมที่ผ่าน calibration ก่อน Repo ไม่รวม `.env`, dataset, model weights, runs หรือ reports ต้องจัดเตรียมไฟล์เหล่านี้แยกต่างหาก

`requirements.txt` บันทึกเวอร์ชัน dependency หลักจาก metadata ของ environment เดิม ไม่ใช่ผลทดสอบการติดตั้งใหม่ PyTorch เดิมคือ `2.11.0+cu126`; ต้องใช้ build ที่ตรงกับ CUDA/driver ของเครื่อง ห้ามเปลี่ยน environment ของงานเทรนที่ยังทำงานอยู่เพียงเพื่อใช้ refactor นี้

เครื่อง V100 เดิมใช้ library directory นี้ก่อนเริ่ม Python:

```bash
export LD_LIBRARY_PATH="$PWD/.runtime/nvidia-580.95.05/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Recipe ระบุ physical GPU ด้วย `gpu` โดย trainer เห็น logical device `0` หลังตั้ง `CUDA_VISIBLE_DEVICES` ก่อน import Torch ค่าเดิมคือ physical GPU `1`; expanded recipes ตรวจหน่วยความจำว่างอย่างน้อย 6 GiB คง batch เดิมไว้จนกว่าจะทดสอบ hardware ใหม่

## เทรนด้วย script เดียว

ตรวจค่าก่อนโดยไม่แตะ CUDA หรือเขียนผลลัพธ์:

```bash
python scripts/train.py configs/expanded-latest-l.json --dry-run
```

รันจริงใน environment เดิม:

```bash
.venv-ppe-v100/bin/python -u scripts/train.py configs/expanded-latest-l.json
```

เมื่อต้องการทดลองใหม่ ให้ copy recipe แล้วเปลี่ยน `run` เป็น directory ใหม่ก่อนเปลี่ยนค่าการเทรน ค่า `state`/`metrics` ที่ระบุเองต้องเปลี่ยนให้ตรงด้วย ส่วน `validation` ต้องสะท้อน dataset release จริง ห้ามลดข้อกำหนดเพื่อให้ข้อมูลที่ผิดผ่าน

Path ภายใน recipe อ้างจาก root ของ repo ไม่ใช่จาก directory ของ JSON; `--root /path/to/PPE-Detection` เปลี่ยน root ได้ แต่ path แบบ absolute ที่ฝังใน manifest/data.yaml ต้องยังเข้าถึงได้ ใช้ `--root` อย่างระมัดระวังเพราะ outputs จะเขียนที่ root นั้นด้วย

ส่งหลาย recipe เพื่อเทรนเรียงลำดับได้ (ต้องใช้ GPU เดียวกัน):

```bash
python scripts/train.py configs/reviewed-yolo26s.json configs/reviewed-yolo26m.json
```

Reviewed YOLO26 ยังต้องมี `reports/ppe-yolo26-reviewed-sequence/preflight.json` ที่ยืนยัน forward/backward batch 4 ผ่านทั้งสองรุ่น การเทรนไม่ส่ง benchmark อัตโนมัติอีกต่อไป

## ย้ายจากคำสั่งเก่า

| Python script เดิม / shell launcher เดิม | Recipe สำหรับ `python scripts/train.py` |
| --- | --- |
| `train_ppe_full_current.py` | `configs/full-yolo11s.json` |
| `train_ppe_reviewed.py` | `configs/reviewed-yolo11s.json` |
| `train_ppe_yolo26_sequence.py` | `configs/reviewed-yolo26s.json` และ `configs/reviewed-yolo26m.json` |
| `train_ppe_yolo26_full.py` | `configs/full-yolo26s.json` |
| `train_ppe_yolo26_960.py` | `configs/full-yolo26s-960.json` |
| `train_ppe_expanded.py` | `configs/expanded-five.json` |
| `train_ppe_next_three.py` | `configs/expanded-eight.json` |
| `train_ppe_four_more.py` / `run_ppe_four_more.sh` | `configs/expanded-latest-s.json` |
| `train_ppe_four_more_l.py` / `run_ppe_four_more_l.sh` | `configs/expanded-latest-l.json` |

Recipe เก็บ run directory, model, batch, optimizer, augmentation และ resolution เดิม แต่โค้ดจะตรวจ image hashes เพิ่มเติม และหยุดเมื่อ input หรือ recipe ของ run เดิมเปลี่ยน

## Resume และ artifacts

- เรียกคำสั่งเดิมเพื่อ resume; ลอง `last.pt` ก่อน แล้ว checkpoint ราย epoch ที่ใหม่สุดและยังมี optimizer state หากไม่มี checkpoint ที่ใช้ได้จะหยุด ไม่เริ่มใหม่ทับผลเดิม
- Run ที่สถานะ `complete` จะไม่เทรน/ประเมินซ้ำ ต้องมี best checkpoint และ metrics อยู่ครบ
- หากล้มระหว่าง evaluation ให้เรียกเดิมเพื่อประเมินต่อจาก best checkpoint โดยไม่เทรนซ้ำ
- `resolved-recipe.json` บันทึก recipe และ hash ที่ใช้จริง; `input-hashes.jsonl` ตรึง input; `training-state.json` บอกสถานะ; `test-metrics.json` มี overall/per-class metrics
- Full YOLO11 เก็บ state/metrics ใน dataset directory เดิม และ reviewed YOLO11 เก็บ metrics ใน dataset directory เดิม เพื่อให้ report readers ยังทำงาน
- Reviewed YOLO11 ยังประเมิน baseline บน reviewed test ชุดเดียวกัน และเขียน `comparison.json`
- ไม่เขียนทับ `experiment.json` เก่าที่มี provenance อยู่แล้ว; run ใหม่บันทึกค่าที่ใช้จริงพร้อม recipe
- Best checkpoint เลือกจาก validation; test ใช้ประเมินหลังเทรน ไม่มีการปรับค่าจากคะแนน benchmark

## Dataset release

ใช้ `build_ppe_expanded.py`, `build_ppe_next_three.py`, `build_ppe_four_more.py` สำหรับ release เดิม แต่ next-three/four-more เรียกกลไก `build_append` ร่วมกัน กฎ source review ต่างกันจึงยังระบุชัดเจนใน module เดียว

คงการปฏิเสธ overwrite, benchmark family/hash/video protection, semantic quarantine, ภาพ extra heldout และ core validation/test hashes ไว้ครบ การรวม family ยังรักษาลำดับ union เดิม เพราะลำดับนี้มีผลต่อ seeded split

## Benchmark เป็นขั้นตอนแยก

ใช้สคริปต์ submit เดิมหลังเทรนเสร็จ การส่งต้องมี frozen exam และ preparation ที่ถูกต้องอยู่ก่อน ตัวอย่าง full YOLO26s ที่เดิมส่งอัตโนมัติ:

```bash
python scripts/submit_ppe_benchmark98.py \
  --project 101 --definition 3 --expected-images 5443 \
  --out reports/andromeda-positive-fullcurrent-yolo26s \
  --weights runs/ppe-full-current-10030-yolo26s-20261111/weights/best.pt \
  --device 0 --name yolo26s-fullcurrent10030-20261111 --version best-validation
```

Directory `--out` ต้องมี `exam` snapshot และ `preparation.json`; trainer ไม่สร้างสิ่งเหล่านี้แล้ว หากต้องการลำดับ train → submit → train → submit ให้เรียกแต่ละคำสั่งสลับกันอย่างชัดเจน Summary เฉพาะ sequence (`results.json`/`experiment-state.json`) เดิมถูกแทนด้วย state/metrics ของแต่ละ run และ confirmed-server-run ของแต่ละ submission

ทุก submission ตรวจ checkpoint hash กับ config, บันทึก intent ก่อนส่ง และยืนยัน server run หลังส่ง หากมี receipt/pending แต่ server ไม่พบ จะหยุดให้ตรวจสอบ ไม่ retry อัตโนมัติ การลบ receipt โดยไม่ตรวจ server อาจทำให้ส่งซ้ำได้

## ตรวจงานก่อนแก้/ส่ง

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts
uv tool run ruff check scripts tests --select F,I
uv tool run ruff format --check scripts tests
```

การทดสอบนี้ไม่ยืนยันผล GPU training เต็มรอบ, throughput หรือคะแนนโมเดล และไม่ได้สร้าง dataset ขนาดจริงหรือส่ง benchmark จริง
