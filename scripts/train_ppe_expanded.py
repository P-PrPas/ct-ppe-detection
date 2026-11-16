"""Train the authorized local five-source expansion; never upload or edit source datasets."""
import json,hashlib,time
from pathlib import Path
import train_ppe_full_current as training
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ppe-expanded-five-sources-20261114'
RUN=ROOT/'runs/ppe-expanded-five-sources-yolo26s-960-20261114'
EXPECTED_IMAGES=14536
BASE_WEIGHTS=ROOT/'models/yolo26s.pt'
TRAIN_BATCH=4
EVAL_BATCH=4

def main():
    import torch,yaml
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    free,_=torch.cuda.mem_get_info();assert free>=6*1024**3,'GPU1 has less than6GiB free; do not disturb other jobs'
    assert yaml.safe_load((DATA/'data.yaml').read_text())['names']==['Helmet','Vest','Mask_Gas','Goggles','Coverall','Gloves','Mask_Face','Shoes','Safety_Belt','Cap','Ear_Protection']
    summary=json.loads((DATA/'summary.json').read_text());assert summary['total_images']==EXPECTED_IMAGES and summary['group_leakage']==0 and summary['benchmark_exact_leakage_train_val']==0
    rows=[json.loads(l) for l in (DATA/'manifest.jsonl').read_text().splitlines()];assert len(rows)==EXPECTED_IMAGES
    for r in rows:assert hashlib.sha256(Path(r['label']).read_bytes()).hexdigest()==r['label_sha256'],r['label']
    RUN.mkdir(exist_ok=True)
    (RUN/'input-hashes.jsonl').write_text(''.join(json.dumps(dict(image=r['image'],label=r['label'],image_sha256=r['image_sha256'],label_sha256=r['label_sha256'],group=r['group'],split=r['split']))+'\n' for r in rows))
    record=dict(dataset=str(DATA),base_weights=str(BASE_WEIGHTS),base_weights_sha256=hashlib.sha256(BASE_WEIGHTS.read_bytes()).hexdigest(),imgsz=960,batch=TRAIN_BATCH,epochs=100,patience=20,seed=42,optimizer='MuSGD',lr0=.01,momentum=.9,nbs=64,device='physicalGPU1, visiblelogical0',source_policy='Frozenpool snapshots copied from authorizedmount; no edits/uploads onAndromeda',split_policy='Protect finalbenchmark originalfamilies/exacthashgroups fromtrain/val; allimagesretained',data_summary=summary,scheduled_automation=False,quota_cutoff=False,created_at=time.time())
    (RUN/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')
    state_path=RUN/'training-state.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('status')=='complete':
        print('Already complete; not restarting',flush=True);return
    training.DATA=DATA;training.RUN=RUN;training.STATE=state_path;training.BASE_WEIGHTS=BASE_WEIGHTS
    training.IMGSZ=960;training.TRAIN_BATCH=TRAIN_BATCH;training.EVAL_BATCH=EVAL_BATCH
    training.TRAIN_OPTIMIZER='MuSGD';training.TRAIN_LR0=.01;training.TRAIN_MOMENTUM=.9
    training.TEST_METRICS_PATH=RUN/'test-metrics.json'
    print(f'VERIFIED expanded{EXPECTED_IMAGES}; labels unchanged; pretrained{BASE_WEIGHTS.stem}960 batch{TRAIN_BATCH}; physicalGPU1',flush=True)
    training.main()
if __name__=='__main__':
    try:main()
    except BaseException as e:
        RUN.mkdir(exist_ok=True);(RUN/'training-state.json').write_text(json.dumps(dict(status='failed',error=repr(e),updated_at=time.time()))+'\n');raise
