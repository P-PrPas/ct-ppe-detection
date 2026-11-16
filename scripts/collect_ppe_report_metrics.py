"""Report-only heldout evaluation, preserves training artifacts and exports numeric curves."""
import os,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
from ultralytics import YOLO
from ultralytics.utils.metrics import smooth
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/ppe-training-20261109'
model=YOLO(str(ROOT/'runs/ppe-full-current-20261109/weights/best.pt'))
captured={}
model.add_callback('on_val_end',lambda validator: captured.update(matrix=validator.confusion_matrix.matrix.copy()))
m=model.val(data=str(ROOT/'data/ppe-full-current-20261109/data.yaml'),split='test',device=0,imgsz=640,batch=8,workers=2,plots=True,save_txt=True,save_conf=True,project=str(OUT),name='test-evaluation',exist_ok=True)
np.savez_compressed(OUT/'curves.npz',px=m.box.px,p=m.box.p_curve,r=m.box.r_curve,f1=m.box.f1_curve,pr=m.box.prec_values,confusion=captured['matrix'])
idx=smooth(m.box.f1_curve.mean(0),.1).argmax()
result=dict(overall=m.results_dict,macro_f1=float(np.mean(m.box.f1)),diagnostic_confidence=float(m.box.px[idx]),speed=m.speed,curves=[dict(x=np.asarray(c[0]).tolist(),y=np.asarray(c[1]).tolist(),xlabel=c[2],ylabel=c[3]) for c in m.curves_results],per_class=[dict(class_id=int(c),name=model.names[int(c)],precision=float(m.box.p[i]),recall=float(m.box.r[i]),f1=float(m.box.f1[i]),mAP50=float(m.box.ap50[i]),mAP50_95=float(m.box.ap[i])) for i,c in enumerate(m.box.ap_class_index)],confusion_confidence=.25,confusion_matching_iou=.5,nms_iou=.7,model_sha256=hashlib.sha256((ROOT/'runs/ppe-full-current-20261109/weights/best.pt').read_bytes()).hexdigest())
(OUT/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
print('REPORT METRICS SAVED',flush=True)
