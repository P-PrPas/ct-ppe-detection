"""Fixed full10030 YOLO26s experiment, heldout evaluation and Andromeda submission."""
import os,json,time,hashlib,gc,subprocess,sys
from pathlib import Path
from collections import Counter,defaultdict
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch,yaml
import train_ppe_full_current as training
from train_ppe_yolo26_sequence import server_run
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ppe-full-current-20261109'
RUN=ROOT/'runs/ppe-full-current-10030-yolo26s-20261111'
OUT=ROOT/'reports/andromeda-positive-fullcurrent-yolo26s'
EXAM=ROOT/'reports/andromeda-benchmark-positive/exam'
BASE=ROOT/'models/yolo26s.pt'

def state(stage,**kw):
    (RUN/'experiment-state.json').write_text(json.dumps(dict(stage=stage,updated_at=time.time(),dataset=str(DATA),model='yolo26s',**kw),indent=2)+'\n')

def verify():
    names=yaml.safe_load((DATA/'data.yaml').read_text())['names']
    assert names==['Helmet','Vest','Mask_Gas','Goggles','Coverall','Gloves','Mask_Face','Shoes','Safety_Belt','Cap','Ear_Protection']
    rows=[json.loads(l) for l in (DATA/'manifest.jsonl').read_text().splitlines()]
    assert len(rows)==10030
    groups=defaultdict(set);counts=Counter();support=defaultdict(Counter);snap=[]
    for r in rows:
        split=r['split'];counts[split]+=1;groups[r['group']].add(split)
        image=Path(r['image']);label=Path(r['label']);assert image.is_file() and label.is_file()
        for line in label.read_text().splitlines():
            if line.strip():
                fields=line.split();assert len(fields)==5 and 0<=int(fields[0])<11
                support[split][int(fields[0])]+=1
        snap.append(dict(image=str(image),label=str(label),split=split,group=r['group'],image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
    assert counts==dict(train=8024,val=1003,test=1003) and all(len(s)==1 for s in groups.values())
    for split in counts:
        assert len(list((DATA/'images'/split).glob('*')))==counts[split]
        assert len(list((DATA/'labels'/split).glob('*.txt')))==counts[split]
        assert set(support[split])==set(range(11))
    (RUN/'input-hashes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in snap))
    record=dict(dataset=str(DATA),images=counts,classes=names,groups=len(groups),group_leakage=0,base_weights=str(BASE),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),imgsz=640,batch=4,epochs=100,patience=20,seed=42,optimizer='MuSGD',lr0=.01,momentum=.9,nbs=64,inference_head='one-to-many with NMS; nms=None',reference='YOLO26s reviewed2250: same training recipe; only dataset differs',quota_cutoff=False,scheduled_automation=False,benchmark='project101 definition3, frozen5443 images; train overlap401,val48. No benchmark tuning.',input_policy='Existing full-current release unchanged; no source edits, label audit, filtering or person crops.')
    (RUN/'experiment.json').write_text(json.dumps(record,indent=2)+'\n');print('VERIFIED FULL RELEASE',json.dumps(record),flush=True)

def main():
    RUN.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    verify()
    training.DATA=DATA;training.RUN=RUN;training.STATE=RUN/'training-state.json';training.BASE_WEIGHTS=BASE
    training.TRAIN_BATCH=4;training.EVAL_BATCH=4;training.TEST_METRICS_PATH=RUN/'test-metrics.json'
    training.TRAIN_OPTIMIZER='MuSGD';training.TRAIN_LR0=.01;training.TRAIN_MOMENTUM=.9
    previous=json.loads(training.STATE.read_text()) if training.STATE.exists() else {}
    if previous.get('status')!='complete':state('training');training.main()
    assert json.loads(training.STATE.read_text())['status']=='complete'
    gc.collect();torch.cuda.empty_cache();best=RUN/'weights/best.pt'
    if not (OUT/'exam').exists():(OUT/'exam').symlink_to(EXAM,target_is_directory=True)
    prep=json.loads((ROOT/'reports/andromeda-benchmark-positive/preparation.json').read_text())
    (OUT/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    existing=server_run(best)
    if existing is None:
        assert not (OUT/'submission-result.json').exists(),'Local receipt without server confirmation; inspect before retry'
        state('benchmarking')
        cmd=[sys.executable,str(ROOT/'scripts/submit_ppe_benchmark98.py'),'--project','101','--definition','3','--expected-images','5443','--out',str(OUT),'--device','0','--weights',str(best),'--name','yolo26s-fullcurrent10030-20261111','--version','best-validation']
        with (OUT/'run.log').open('a') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        existing=server_run(best);assert existing is not None
    (OUT/'confirmed-server-run.json').write_text(json.dumps(existing,indent=2)+'\n')
    (OUT/'submission-result.json').write_text(json.dumps(dict(confirmed_server_run=existing),indent=2)+'\n')
    state('complete',best=str(best),metrics=str(RUN/'test-metrics.json'),server_run=existing['number'])

if __name__=='__main__':
    try:main()
    except BaseException as e:
        RUN.mkdir(exist_ok=True);state('failed',error=repr(e));raise
