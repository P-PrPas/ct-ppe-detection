"""Submit each distinct trained best checkpoint to frozen project102/exam4, once."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import json,hashlib,subprocess,time,gc,math
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/andromeda-noleak-all-models'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def runs():return json.loads(subprocess.check_output(['andromeda','detection','runs','102','4','--json']))['runs']
def main():
 new=read(OUT/'exam/members.json');old=read(ROOT/'reports/andromeda-benchmark-positive/exam/members.json')
 assert len(new['members'])==3783 and new['definition']['id']=='4'
 assert new['definition']['min_conf']==.5 and new['definition']['iou']==.5
 old_by_hash={m['sha256']:m for m in old['members']};assert all(m['sha256'] in old_by_hash for m in new['members'])
 paths=[]
 for m in new['members']:
  o=old_by_hash[m['sha256']];p=ROOT/'reports/andromeda-benchmark-positive/exam/images'/(o['sample_id']+o['ext'])
  assert sha(p)==m['sha256'];paths.append(str(p))
 models=[]
 for w in sorted((ROOT/'runs').glob('*/weights/best.pt')):
  a=yaml.safe_load((w.parent.parent/'args.yaml').read_text());models.append(dict(run=w.parent.parent.name,weights=str(w),sha=sha(w),imgsz=a['imgsz'],dataset=a['data'],pilot='pilot' in w.parent.parent.name))
 assert len(models)==7 and len({m['sha'] for m in models})==7
 (OUT/'inventory.json').write_text(json.dumps(models,indent=2)+'\n')
 previous=[]
 for c in (ROOT/'reports').glob('*/config.json'):
  try:
   conf=read(c)
   if 'weights_sha256' in conf and (c.parent/'predictions.json').exists() and conf.get('benchmark_manifest')==old['manifest_sha256']:
    previous.append((c.parent,conf))
  except (ValueError,KeyError):pass
 summary=[]
 for model in models:
  print('START',model['run'],flush=True);directory=OUT/model['run'];directory.mkdir(exist_ok=True)
  (OUT/'state.json').write_text(json.dumps(dict(stage='processing',model=model['run'],completed=len(summary),total=7)))
  existing=[r for r in runs() if r['candidate'].get('weights_sha256')==model['sha']];assert len(existing)<=1
  if existing:
   assert existing[0]['candidate']['config']['imgsz']==model['imgsz'];server=existing[0]
  else:
   assert not (directory/'receipt.json').exists(),'Receipt without server confirmation: inspect before retry'
   candidates=[(p,c) for p,c in previous if c['weights_sha256']==model['sha'] and c['imgsz']==model['imgsz'] and c['confidence']==.05 and c['nms_iou']==.7 and c['max_det']==300]
   pred={};method='fresh inference'
   if candidates:
    p,c=candidates[0];cached=read(p/'predictions.json')
    for m in new['members']:pred[m['sample_id']]=cached[old_by_hash[m['sha256']]['sample_id']]
    method='reused exact image-hash and weights-hash predictions from '+str(p)
   else:
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2);net=YOLO(model['weights']);assert set(net.names.values())=={c['name'] for c in new['classes']}
    net.add_callback('on_predict_start',lambda _:torch.set_num_threads(2))
    bypath=dict(zip(paths,new['members'],strict=True))
    for i,r in enumerate(net.predict(paths,device=0,imgsz=model['imgsz'],batch=4,conf=.05,iou=.7,max_det=300,stream=True,verbose=False),1):
     m=bypath[r.path];assert m['sample_id'] not in pred
     pred[m['sample_id']]=[dict(cls=net.names[int(c)],box=b,conf=v) for b,v,c in zip(r.boxes.xyxy.cpu().tolist(),r.boxes.conf.cpu().tolist(),r.boxes.cls.cpu().tolist(),strict=True)]
     if i%500==0:print('PREDICT',i,3783,flush=True)
    del net;gc.collect();torch.cuda.empty_cache()
   assert set(pred)=={m['sample_id'] for m in new['members']}
   for boxes in pred.values():
    for b in boxes:
     assert b['cls'] in {c['name'] for c in new['classes']} and len(b['box'])==4 and all(math.isfinite(v) for v in b['box']) and math.isfinite(b['conf']) and 0<=b['conf']<=1
     assert b['box'][2]>b['box'][0] and b['box'][3]>b['box'][1]
   (directory/'predictions.json').write_text(json.dumps(pred,separators=(',',':')))
   config=dict(weights_sha256=model['sha'],imgsz=model['imgsz'],batch=4,confidence=.05,nms_iou=.7,max_det=300,benchmark_manifest=new['manifest_sha256'],scorer=new['definition']['scorer'],benchmark_min_conf=.5,matching_iou=.5,provenance=method,pilot=model['pilot'],dataset=model['dataset'],note='All trained best checkpoints, pilot marked explicitly. No test threshold tuning. Benchmark labels stay on server.')
   (directory/'config.json').write_text(json.dumps(config,indent=2))
   command=['andromeda','submit','102','4','--predictions',str(directory/'predictions.json'),'--name',model['run'],'--version','best-pilot-incomplete' if model['pilot'] else 'best-validation','--weights',model['weights'],'--config',str(directory/'config.json')]
   result=subprocess.run(command,capture_output=True,text=True);(directory/'submit.stdout').write_text(result.stdout);(directory/'submit.stderr').write_text(result.stderr);assert result.returncode==0,'Submission failed; inspect stderr before retry'
   (directory/'receipt.json').write_text(json.dumps(dict(stdout=result.stdout,returncode=result.returncode)))
   found=[r for r in runs() if r['candidate'].get('weights_sha256')==model['sha']];assert len(found)==1;server=found[0]
  (directory/'confirmed-server-run.json').write_text(json.dumps(server,indent=2)+'\n')
  summary.append(dict(**model,server_run=server['number'],results=server['results']))
  (OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print('CONFIRMED',server['number'],json.dumps(server['results']),flush=True)
 (OUT/'state.json').write_text(json.dumps(dict(stage='complete',completed=len(summary),total=7,updated_at=time.time())))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  (OUT/'state.json').write_text(json.dumps(dict(stage='failed',error=repr(e))));raise
