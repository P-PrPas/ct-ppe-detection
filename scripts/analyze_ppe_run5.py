"""Read-only diagnostic of Run5 aggregates and shared exam images, using LOCAL GT only."""
import json,csv,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from ultralytics.utils.metrics import ConfusionMatrix
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/yolo26s-full-error-analysis';OUT.mkdir(exist_ok=True)
RUN=ROOT/'runs/ppe-full-current-10030-yolo26s-20261111';B=ROOT/'reports/andromeda-positive-fullcurrent-yolo26s'
meta=json.loads((B/'exam/members.json').read_text());byhash={m['sha256']:m for m in meta['members']};pred=json.loads((B/'predictions.json').read_text());classes=json.loads((ROOT/'data/ppe-full-current-20261109/summary.json').read_text())['classes'];idx={c:i for i,c in enumerate(classes)}

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();return float(inter/(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter+1e-9))
def score(gt,pred):
    taken=set();tp=0
    for p in sorted(pred,key=lambda b:-b['conf']):
        best=-1;value=.5
        for i,g in enumerate(gt):
            v=iou(g['box'],p['box'])
            if i not in taken and g['cls']==p['cls'] and v>=value:best=i;value=v
        if best>=0:taken.add(best);tp+=1
    return dict(tp=tp,fp=len(pred)-tp,fn=len(gt)-tp),taken
assert score([dict(cls='A',box=[0,0,10,10])],[dict(cls='A',box=[0,0,10,10],conf=.9),dict(cls='A',box=[0,0,10,10],conf=.8)])[0]==dict(tp=1,fp=1,fn=0)
assert score([dict(cls='A',box=[0,0,10,10])],[dict(cls='B',box=[0,0,10,10],conf=.9)])[0]==dict(tp=0,fp=1,fn=1)
size_counts=defaultdict(Counter)
cm=ConfusionMatrix(names=dict(enumerate(classes)));counts=Counter();kinds=Counter();items=[];thresholds={t:Counter() for t in [.05,.25,.5]};per=defaultdict(Counter)
for line in (RUN/'input-hashes.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['split']!='test' or r['image_sha256'] not in byhash:continue
    m=byhash[r['image_sha256']];w,h=m['width'],m['height'];gt=[]
    for l in Path(r['label']).read_text().splitlines():
        if not l.strip():continue
        c,x,y,bw,bh=map(float,l.split());gt.append(dict(cls=classes[int(c)],box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
    allp=pred[m['sample_id']];high=[p for p in allp if p['conf']>=.5];sc,taken=score(gt,high);counts.update(sc)
    for j,g in enumerate(gt):
        b=g['box'];edge=math.sqrt(max(0,(b[2]-b[0])*(b[3]-b[1])))*640/max(w,h)
        bucket='<16px' if edge<16 else ('16-32px' if edge<32 else ('32-96px' if edge<96 else '>=96px'))
        size_counts[bucket]['gt']+=1;size_counts[bucket]['tp']+=j in taken
    for t in thresholds:thresholds[t].update(score(gt,[p for p in allp if p['conf']>=t])[0])
    local=Counter()
    for j,g in enumerate(gt):
        if j in taken:continue
        same=[p for p in allp if p['cls']==g['cls']];highsame=[p for p in high if p['cls']==g['cls']]
        if any(iou(p['box'],g['box'])>=.5 for p in highsame):kind='matching_or_duplicate_candidate'
        elif any(iou(p['box'],g['box'])>=.5 for p in same):kind='same_class_low_confidence_candidate'
        elif any(p['cls']!=g['cls'] and iou(p['box'],g['box'])>=.5 for p in high):kind='wrong_class_high_confidence_candidate'
        elif any(iou(p['box'],g['box'])>=.1 for p in same):kind='same_class_localization_candidate'
        else:kind='no_near_same_class_candidate_above_005'
        kinds[kind]+=1;local[kind]+=1;per[g['cls']][kind]+=1
    cm.process_batch(dict(cls=torch.tensor([idx[p['cls']] for p in high]),conf=torch.ones(len(high)),bboxes=torch.tensor([p['box'] for p in high]).reshape(-1,4)),dict(cls=torch.tensor([idx[g['cls']] for g in gt]),bboxes=torch.tensor([g['box'] for g in gt]).reshape(-1,4)),conf=0,iou_thres=.5)
    items.append(dict(source=r,sample_id=m['sample_id'],gt=gt,pred=high,all_pred=allp,counts=sc,fn_candidates=dict(local)))
assert len(items)==523 and sum(kinds.values())==counts['fn']
for t,v in thresholds.items():
    v['precision']=v['tp']/(v['tp']+v['fp'] or 1);v['recall']=v['tp']/(v['tp']+v['fn'] or 1);v['f1']=2*v['tp']/(2*v['tp']+v['fp']+v['fn'] or 1)
for bucket,v in size_counts.items():v['recall']=v['tp']/v['gt']
summary=dict(size_recall_at_640=size_counts,scope='523 exact-hash shared images from frozen LOCAL TEST and Andromeda exam. GT is local release labels, NOT server answer key.',counts=counts,fn_candidate_categories=kinds,per_class_fn_candidates=per,threshold_diagnostics=thresholds,warning='Diagnostic thresholds on test are descriptive only, not recommendations; candidate categories are geometric proxies, not visually established causes. No benchmark answers exported or labels changed.')
(OUT/'shared-local-test-diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n');np.save(OUT/'shared-confusion.npy',cm.matrix)
choices=[];used=set()
for kind in ['same_class_low_confidence_candidate','no_near_same_class_candidate_above_005','same_class_localization_candidate','wrong_class_high_confidence_candidate']:
    candidates=[v for v in items if v['sample_id'] not in used]
    v=max(candidates,key=lambda x:x['fn_candidates'].get(kind,0));choices.append((kind,v));used.add(v['sample_id'])
font=ImageFont.truetype('/usr/local/share/fonts/plex/IBMPlexSansThai-Regular.ttf',13)
examples=[]
for n,(kind,v) in enumerate(choices,1):
    image=Image.open(v['source']['image']).convert('RGB');image.thumbnail((550,430));w,h=image.size;scale=np.array([w/Image.open(v['source']['image']).width,h/Image.open(v['source']['image']).height]*2)
    canvas=Image.new('RGB',(1120,h+65),'white');draw=ImageDraw.Draw(canvas)
    for off,boxes,title in [(0,v['gt'],'LOCAL GT'),(560,v['pred'],'PRED >=0.50')]:
        canvas.paste(image,(off+5,45));draw.text((off+8,8),title,font=font,fill='#0e1b2d')
        for b in boxes:
            xy=np.array(b['box'])*scale+np.array([off+5,45,off+5,45]);color='#00a2a2' if off==0 else '#4779b5';draw.rectangle(tuple(xy),outline=color,width=2)
            text=b['cls']+(f" {b['conf']:.2f}" if 'conf' in b else '');x,y=xy[:2];y=max(44,y-18);tw=draw.textlength(text,font=font);draw.rectangle((x,y,x+tw+4,y+18),fill=color);draw.text((x+2,y),text,font=font,fill='white')
    name=f'shared-example-{n}.jpg';canvas.save(OUT/name,quality=90);examples.append(dict(asset=name,selection=kind,sample_id=v['sample_id'],source=v['source']['image'],counts=v['counts'],fn_candidates=v['fn_candidates']))
(OUT/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
print(json.dumps(summary),flush=True)
