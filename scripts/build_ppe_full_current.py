"""Copy the authorized 10030 source keys; read labels exclusively from current pools."""
import json, random, shutil, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import yaml
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/ppe-full-current-20261109'
CLASSES=['Helmet','Vest','Mask_Gas','Goggles','Coverall','Gloves','Mask_Face','Shoes','Safety_Belt','Cap','Ear_Protection']

def main():
    rows=[json.loads(l) for l in (ROOT/'reports/ppe-preparation-2026-11-08/draft-manifest.jsonl').read_text().splitlines()]
    assert len(rows)==10030 and len({(r['project_id'],r['file_name']) for r in rows})==10030
    sources={85:Path('/opt/mount/project/data/safety-working'),81:Path('/opt/mount/project/data/cap-pavithrap')}
    names={k:yaml.safe_load((p/'data.yaml').read_text())['names'] for k,p in sources.items()}
    OUT.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list); omitted=Counter(); missing=0
    for r in rows:
        src=sources[r['project_id']]; fn=r['file_name']; label=src/'labels'/Path(fn).with_suffix('.txt')
        lines=[]; counts=Counter()
        if not label.exists(): missing+=1
        for line in label.read_text().splitlines() if label.exists() else []:
            fields=line.split()
            if not fields: continue
            assert len(fields)==5,(label,line)
            name=names[r['project_id']][int(fields[0])]
            if name.startswith('no_'): omitted[name]+=1; continue
            idx=CLASSES.index(name); counts[idx]+=1
            lines.append(str(idx)+' '+' '.join(fields[1:]))
        groups[r['group']].append({'project_id':r['project_id'],'file_name':fn,'group':r['group'],'source':str(src/'images'/fn),'source_label':str(label),'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest() if label.exists() else None,'lines':lines,'counts':counts})
    # Select by group sizes and positive support only, never test performance.
    keys=sorted(groups); rng=random.Random(42); best=None
    for _ in range(500):
        rng.shuffle(keys); assigned={}; totals=Counter(); support={s:Counter() for s in ['train','val','test']}
        for g in keys:
            s='test' if totals['test']<1003 else ('val' if totals['val']<1003 else 'train')
            assigned[g]=s; totals[s]+=len(groups[g])
            for r in groups[g]: support[s].update(r['counts'])
        score=sum(100000 for s in support for c in range(11) if not support[s][c])+sum(abs(totals[s]-n) for s,n in [('train',8024),('val',1003),('test',1003)])
        if best is None or score<best[0]: best=(score,assigned.copy(),totals.copy(),support)
    _,assigned,totals,support=best
    assert all(support['train'][c]>0 for c in range(11))
    manifest=[]; negatives=Counter()
    for g,items in groups.items():
        s=assigned[g]
        for r in items:
            name=str(r['project_id'])+'_'+r['file_name']
            image=OUT/'images'/s/name; label=OUT/'labels'/s/Path(name).with_suffix('.txt')
            image.parent.mkdir(parents=True,exist_ok=True); label.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(r['source'],image); label.write_text('\n'.join(r.pop('lines'))+'\n' if r['counts'] else '')
            negatives[s]+=not bool(r['counts']); r.pop('counts'); r.update(split=s,image=str(image),label=str(label)); manifest.append(r)
    assert sum(totals.values())==10030
    assert len(list((OUT/'images').glob('*/*')))==10030
    assert len(list((OUT/'labels').glob('*/*.txt')))==10030
    assert all(len({r['split'] for r in manifest if r['group']==g})==1 for g in groups)
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (OUT/'data.yaml').write_text(yaml.safe_dump(dict(path=str(OUT),train='images/train',val='images/val',test='images/test',names=CLASSES),sort_keys=False))
    summary=dict(images=totals,groups=Counter(assigned.values()),classes=CLASSES,box_support={s:{CLASSES[c]:v[c] for c in range(11)} for s,v in support.items()},negative_images=negatives,omitted_boxes=omitted,missing_source_labels=missing,group_leakage=0,source_selection='All 10030 manifest source keys; current pool labels only. Extra pool files outside authorized source keys excluded.',benchmark='No definitions on projects 85/81; CLI lacks definition creation. Local grouped heldout test.',risk='Current raw labels accepted without audit or repair; full frames include products/negatives. Deployment ROI distribution differs.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
