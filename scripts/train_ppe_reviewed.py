"""Controlled reviewed-data run: same pretrained YOLO11s and training recipe, matched test comparison."""
import hashlib,json
from pathlib import Path
from collections import Counter
import train_ppe_full_current as training
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ppe-reviewed-2250'
RUN=ROOT/'runs/ppe-reviewed-2250-yolo11s-20261109'
training.TRAIN_OPTIMIZER="MuSGD"
training.TRAIN_LR0=.01
training.TRAIN_MOMENTUM=.9
training.DATA=DATA
training.RUN=RUN
training.STATE=RUN/'training-state.json'
RUN.mkdir(parents=True,exist_ok=True)

def verify():
    import yaml
    names=yaml.safe_load((DATA/'data.yaml').read_text())['names']
    expected=['Helmet','Vest','Mask_Gas','Goggles','Coverall','Gloves','Mask_Face','Shoes','Safety_Belt','Cap','Ear_Protection']
    assert names==expected
    rows=[json.loads(l) for l in (DATA/'manifest.jsonl').read_text().splitlines()]
    old=[json.loads(l) for l in (ROOT/'data/ppe-full-current-20261109/manifest.jsonl').read_text().splitlines()]
    old_members={(str(r['project_id']),r['file_name']):r['split'] for r in old}
    old_group_splits={r['group']:r['split'] for r in old}
    groups={};counts=Counter();hash_splits={}
    for r in rows:
        split=r['split'];counts[split]+=1
        groups.setdefault(r['group'],set()).add(split)
        assert old_members[(str(r['project_id']),r['file_name'])]==split,'Changed split membership'
        assert old_group_splits[r['group']]==split,'Group crosses baseline split'
        image=Path(r['image']);label=Path(r['label']);assert image.is_file() and label.is_file()
        ih=hashlib.sha256(image.read_bytes()).hexdigest();lh=hashlib.sha256(label.read_bytes()).hexdigest()
        assert ih==r['image_sha256'] and lh==r['release_label_sha256'],'Dataset differs from frozen reviewed manifest'
        hash_splits.setdefault(ih,set()).add(split)
    assert counts==dict(train=1883,val=179,test=188) and len(rows)==2250
    assert all(len(s)==1 for s in groups.values()) and all(len(s)==1 for s in hash_splits.values())
    record=dict(dataset=str(DATA),images=counts,groups=len(groups),group_leakage=0,exact_hash_leakage=0,original_split_preserved=True,base_weights=str(ROOT/'models/yolo11s.pt'),base_sha256=hashlib.sha256((ROOT/'models/yolo11s.pt').read_bytes()).hexdigest(),recipe='Identical to full-current: 640, batch12, epochs100, patience20, seed42, same augmentation; explicit MuSGD lr0=.01 momentum=.9 matches full-current effective optimizer',primary_comparison='Both models evaluated on the SAME reviewed test188 images and labels. Different dataset test scores alone are not a controlled comparison.',quota_policy='Disabled by explicit user instruction2026-11-10',automations=False)
    (RUN/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')
    print('DATASET VERIFIED',json.dumps(record),flush=True)

def main():
    verify();training.main()
    status=json.loads(training.STATE.read_text())
    if status['status']!='complete': return
    training.state('evaluating_baseline',best=status['best'])
    from ultralytics import YOLO
    m=YOLO(str(ROOT/'runs/ppe-full-current-20261109/weights/best.pt')).val(data=str(DATA/'data.yaml'),split='test',imgsz=640,batch=12,device=0,project=str(RUN),name='baseline-on-reviewed-test',plots=True)
    new=json.loads((DATA/'test-metrics.json').read_text())
    baseline=dict(overall=m.results_dict,per_class=[])
    for c,name in enumerate(['Helmet','Vest','Mask_Gas','Goggles','Coverall','Gloves','Mask_Face','Shoes','Safety_Belt','Cap','Ear_Protection']):
        idx=list(m.box.ap_class_index).index(c)
        p,r,a,b=m.box.class_result(idx)
        baseline['per_class'].append(dict(name=name,precision=float(p),recall=float(r),mAP50=float(a),mAP50_95=float(b)))
    result=dict(test='Identical reviewed test split:188 images, same labels, same evaluation settings',baseline=baseline,reviewed=new,delta={k:new['overall'][k]-m.results_dict[k] for k in new['overall'] if k.startswith('metrics/')},limitations='Reviewed subset changes both dataset quality and quantity; one seed, small rare-class test support; not a causal isolation of label quality alone.')
    (RUN/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    training.state('complete',best=status['best'],metrics=str(DATA/'test-metrics.json'),comparison=str(RUN/'comparison.json'))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        training.state('failed',error=repr(e));raise
