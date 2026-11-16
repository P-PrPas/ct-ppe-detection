"""Screen training entrypoint, resumable checkpoints and one final heldout evaluation."""
import json, os, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['WANDB_MODE']='disabled'
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ppe-full-current-20261109'
RUN=ROOT/'runs/ppe-full-current-20261109'
STATE=DATA/'training-state.json'
BASE_WEIGHTS=ROOT/"models/yolo11s.pt"
TRAIN_BATCH=12
EVAL_BATCH=12
IMGSZ=640
TEST_METRICS_PATH=None
TRAIN_OPTIMIZER="auto"
TRAIN_LR0=.01
TRAIN_MOMENTUM=.937

def state(status,**kwargs):
    STATE.write_text(json.dumps(dict(status=status,updated_at=time.time(),run=str(RUN),**kwargs),indent=2)+'\n')

def check(trainer):
    state('training',epoch=trainer.epoch+1,last_checkpoint=str(RUN/'weights/last.pt'))

def main():
    import torch
    checkpoint=RUN/'weights/last.pt'
    candidates=[checkpoint]+sorted((RUN/'weights').glob('epoch*.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
    if checkpoint.exists():
        for candidate in candidates:
            saved=torch.load(candidate,map_location='cpu',weights_only=False)
            valid=saved.get('epoch',-1)>=0 and saved.get('optimizer') is not None
            del saved
            if valid:
                checkpoint=candidate
                break
        else:
            raise RuntimeError('No checkpoint with optimizer state available to resume')
        print('Resuming from',checkpoint,flush=True)
    model=YOLO(str(checkpoint if checkpoint.exists() else BASE_WEIGHTS))
    model.add_callback('on_model_save',check)
    model.add_callback('on_train_epoch_start',lambda trainer:state('training',epoch=trainer.epoch+1,checkpoint_exists=(RUN/'weights/last.pt').exists()))
    state('starting',resume=checkpoint.exists())
    if checkpoint.exists(): model.train(resume=True,device=0)
    else:
        model.train(data=str(DATA/'data.yaml'),project=str(RUN.parent),name=RUN.name,exist_ok=True,device=0,imgsz=IMGSZ,batch=TRAIN_BATCH,epochs=100,patience=20,seed=42,workers=4,save=True,save_period=1,amp=True,optimizer=TRAIN_OPTIMIZER,lr0=TRAIN_LR0,momentum=TRAIN_MOMENTUM,mosaic=.5,close_mosaic=10,mixup=0,copy_paste=0,degrees=5,translate=.1,scale=.3,shear=0,perspective=0,fliplr=.5,flipud=0,hsv_h=.015,hsv_s=.5,hsv_v=.3)
    best=RUN/'weights/best.pt'
    state('evaluating',best=str(best))
    metrics=YOLO(str(best)).val(data=str(DATA/'data.yaml'),split='test',imgsz=IMGSZ,batch=EVAL_BATCH,device=0,project=str(RUN),name='heldout-test',plots=True)
    from collections import Counter
    boxes=Counter(); images=Counter()
    for p in (DATA/'labels/test').glob('*.txt'):
        ids=[int(l.split()[0]) for l in p.read_text().splitlines() if l.strip()]
        boxes.update(ids); images.update(set(ids))
    per=[]
    for i,name in enumerate(model.names.values()):
        indices=list(metrics.box.ap_class_index)
        values=metrics.box.class_result(indices.index(i)) if i in indices else [0.,0.,0.,0.]
        per.append(dict(name=name,precision=float(values[0]),recall=float(values[1]),mAP50=float(values[2]),mAP50_95=float(values[3]),boxes=boxes[i],images=images[i]))
    metrics_path=TEST_METRICS_PATH or DATA/'test-metrics.json'
    metrics_path.write_text(json.dumps(dict(overall=metrics.results_dict,per_class=per),indent=2)+'\n')
    state('complete',best=str(best),metrics=str(metrics_path))
if __name__=='__main__':
    try: main()
    except BaseException as e:
        state('failed',error=repr(e)); raise
