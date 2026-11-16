"""One post-training continuation: fetch the heldout exam and submit best.pt predictions."""
import json, os, subprocess, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ppe-full-current-20261109'
RUN=ROOT/'runs/ppe-full-current-20261109'
OUT=DATA/'andromeda-benchmark'; OUT.mkdir(exist_ok=True)

def state(status,**kw):
    (OUT/'state.json').write_text(json.dumps(dict(status=status,updated_at=time.time(),project_id=97,**kw),indent=2)+'\n')

def cli(*args):
    r=subprocess.run(['andromeda',*map(str,args)],capture_output=True,text=True,check=True)
    return r.stdout

def main():
    training=json.loads((DATA/'training-state.json').read_text())
    if training['status']!='complete':
        state('training_not_complete',training=training); return
    raw=cli('detection','definitions',97,'--json'); (OUT/'definitions.json').write_text(raw)
    definitions=json.loads(raw)['definitions']
    if len(definitions)!=1:
        state('awaiting_benchmark_definition',definitions=definitions,reason='Need exactly one explicitly heldout benchmark definition.'); return
    definition=definitions[0]['id']
    state('fetching',definition=definition)
    exam=OUT/'exam'
    (OUT/'fetch.log').write_text(cli('fetch',97,definition,'--out',exam))
    candidates=[]
    for p in exam.rglob('*.json'):
        d=json.loads(p.read_text())
        if isinstance(d,dict) and d.get('images') and d.get('categories'): candidates.append(d)
    assert len(candidates)==1,'Expected one COCO exam document'
    coco=candidates[0]
    manifest=[json.loads(l) for l in (DATA/'manifest.jsonl').read_text().splitlines()]
    test={Path(r['image']).name:r['image'] for r in manifest if r['split']=='test'}
    names=[Path(i['file_name']).name for i in coco['images']]
    assert names and len(set(names))==len(names),'Empty or duplicate benchmark membership'
    assert set(names).issubset(test),'Benchmark includes images outside frozen heldout test split'
    from ultralytics import YOLO
    model=YOLO(str(RUN/'weights/best.pt'))
    categories={c['name']:c['id'] for c in coco['categories']}
    assert set(model.names.values())==set(categories),'Benchmark class mismatch'
    predictions=[]
    state('predicting',definition=definition,images=len(names))
    results=model.predict([test[n] for n in names],device=0,imgsz=640,batch=12,conf=.001,iou=.7,max_det=300,stream=True,verbose=False)
    for image,result in zip(coco['images'],results,strict=True):
        for box,score,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist(),strict=True):
            x1,y1,x2,y2=box
            predictions.append(dict(image_id=image['id'],category_id=categories[model.names[int(cls)]],bbox=[x1,y1,x2-x1,y2-y1],score=score))
    path=OUT/'predictions.json'; path.write_text(json.dumps(predictions))
    state('submitting',definition=definition,predictions=len(predictions))
    response=cli('submit',97,definition,'--predictions',path,'--name','yolo11s-full-current-20261109','--weights',RUN/'weights/best.pt','--config',RUN/'args.yaml')
    (OUT/'submission.log').write_text(response)
    state('submitted',definition=definition,response=response,weights=str(RUN/'weights/best.pt'))
if __name__=='__main__':
    try: main()
    except BaseException as e:
        state('failed',error=repr(e)); raise
