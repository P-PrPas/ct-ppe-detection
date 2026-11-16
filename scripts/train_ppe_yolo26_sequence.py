"""Authorized fixed experiment sequence:26s train/eval/submit, then26m train/eval/submit."""
import os,json,time,hashlib,subprocess,sys,gc
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import train_ppe_reviewed as reviewed
import train_ppe_full_current as training
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/ppe-yolo26-reviewed-sequence'
EXAM=ROOT/'reports/andromeda-benchmark-positive/exam'
PRIVATE=ROOT/'.runtime/nvidia-580.95.05'

def save(stage,**kw):
    REPORT.mkdir(exist_ok=True)
    (REPORT/'state.json').write_text(json.dumps(dict(stage=stage,updated_at=time.time(),dataset='ppe-reviewed-2250',device='physical GPU1',**kw),indent=2)+'\n')

def server_run(weights):
    result=subprocess.run(['andromeda','detection','runs','101','3','--json'],capture_output=True,text=True,check=True)
    sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    found=[r for r in json.loads(result.stdout)['runs'] if r['candidate'].get('weights_sha256')==sha]
    assert len(found)<=1,'Duplicate server submissions need inspection'
    return found[0] if found else None

def phase(name):
    run=ROOT/'runs'/f'ppe-reviewed-2250-{name}-20261110'
    out=ROOT/'reports'/f'andromeda-positive-reviewed2250-{name}'
    run.mkdir(exist_ok=True);out.mkdir(exist_ok=True)
    weights=ROOT/'models'/f'{name}.pt'
    reviewed.RUN=run;reviewed.verify()
    record=json.loads((run/'experiment.json').read_text())
    record.update(base_weights=str(weights),base_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),recipe='640, batch4, epochs100, patience20, seed42, MuSGD lr0=.01 momentum=.9, same bounded augmentation',train_batch=4,nbs=64,gradient_accumulation_target=16,inference_head='one-to-many with external NMS; nms=None',quota_policy='No quota cutoff; revoked by user',sequence='26s train/eval/submit then26m train/eval/submit; no benchmark tuning')
    (run/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')
    training.DATA=reviewed.DATA;training.RUN=run;training.STATE=run/'training-state.json'
    training.BASE_WEIGHTS=weights;training.TRAIN_BATCH=4;training.EVAL_BATCH=4
    training.TRAIN_OPTIMIZER='MuSGD';training.TRAIN_LR0=.01;training.TRAIN_MOMENTUM=.9
    training.TEST_METRICS_PATH=run/'test-metrics.json'
    previous=json.loads(training.STATE.read_text()) if training.STATE.exists() else {}
    if previous.get('status')!='complete':
        save('training',model=name,run=str(run));training.main()
    assert json.loads(training.STATE.read_text())['status']=='complete'
    gc.collect();torch.cuda.empty_cache()
    best=run/'weights/best.pt'
    if not (out/'exam').exists():(out/'exam').symlink_to(EXAM,target_is_directory=True)
    prep=json.loads((ROOT/'reports/andromeda-benchmark-positive-reviewed2250/preparation.json').read_text())
    (out/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    existing=server_run(best)
    if existing is None:
        # A local successful receipt without server confirmation must not trigger a duplicate submit.
        assert not (out/'submission-result.json').exists(),'Local receipt exists but server run missing; inspect first'
        save('benchmarking',model=name,run=str(run),benchmark=str(out))
        command=[sys.executable,str(ROOT/'scripts/submit_ppe_benchmark98.py'),'--project','101','--definition','3','--expected-images','5443','--out',str(out),'--device','0','--weights',str(best),'--name',f'{name}-reviewed2250-20261110','--version','best-validation']
        with (out/'run.log').open('a') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        existing=server_run(best);assert existing is not None,'CLI returned success without server confirmation'
    (out/'confirmed-server-run.json').write_text(json.dumps(existing,indent=2)+'\n')
    (out/'submission-result.json').write_text(json.dumps(dict(confirmed_server_run=existing),indent=2)+'\n')
    save('submitted',model=name,server_run=existing['number'],run=str(run))
    return dict(model=name,run=str(run),test=json.loads((run/'test-metrics.json').read_text()),andromeda_run=existing)

def main():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    pre=json.loads((REPORT/'preflight.json').read_text());assert len(pre)==2 and all(x['forward_backward']=='passed' and x['batch']==4 for x in pre)
    manifest=json.loads((EXAM/'members.json').read_text());assert len(manifest['members'])==5443 and manifest['definition']['id']=='3'
    results=[]
    for name in ['yolo26s','yolo26m']:
        print('START PHASE',name,flush=True);results.append(phase(name));print('FINISHED PHASE',name,flush=True)
    (REPORT/'results.json').write_text(json.dumps(results,indent=2)+'\n');save('complete',models=['yolo26s','yolo26m'])

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save('failed',error=repr(e));raise
