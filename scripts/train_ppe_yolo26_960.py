"""Controlled 960 training experiment; same frozen full dataset and 640 training recipe."""
import json, hashlib
from pathlib import Path
import train_ppe_full_current as training

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/ppe-full-current-10030-yolo26s-960-20261111'
REFERENCE = ROOT/'runs/ppe-full-current-10030-yolo26s-20261111'

def main():
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    RUN.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (REFERENCE/'input-hashes.jsonl').read_text().splitlines()]
    assert len(rows) == 10030
    for r in rows:
        assert hashlib.sha256(Path(r['label']).read_bytes()).hexdigest() == r['label_sha256'], r['label']
    (RUN/'input-hashes.jsonl').write_text((REFERENCE/'input-hashes.jsonl').read_text())
    record = json.loads((REFERENCE/'experiment.json').read_text())
    record.update(imgsz=960,reference=str(REFERENCE),changed_factor='train/eval resolution 640 to960 only',
        decision='User authorized all three investigation steps. Existing640 model inference at960 was worse; this separately tests retraining at960, not a proven improvement.',
        label_risk='Bounded validation review found wrong classes, misplaced and missing boxes. Dataset remains frozen, including products and negatives. No automatic corrections.',
        test_policy='Best checkpoint chosen on validation only; heldout test evaluated once after training. No benchmark tuning.',
        scheduled_automation=False)
    (RUN/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')
    training.RUN=RUN; training.STATE=RUN/'training-state.json'
    training.BASE_WEIGHTS=ROOT/'models/yolo26s.pt'
    training.TRAIN_BATCH=4; training.EVAL_BATCH=4; training.IMGSZ=960
    training.TRAIN_OPTIMIZER='MuSGD'; training.TRAIN_LR0=.01; training.TRAIN_MOMENTUM=.9
    training.TEST_METRICS_PATH=RUN/'test-metrics.json'
    print('VERIFIED labels unchanged; train960 from original pretrained, not fine-tuning best640',flush=True)
    training.main()

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        RUN.mkdir(exist_ok=True)
        (RUN/'training-state.json').write_text(json.dumps(dict(status='failed',error=repr(e)))+'\n')
        raise
