"""Append frozen40/75/72 pools, preserve the previous core evaluation, keep all images."""
import json,hashlib,random,shutil
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image
import yaml
from build_ppe_expanded import CLASSES,parse,family,digest
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/ppe-expanded-five-sources-20261114'
OUT=ROOT/'data/ppe-expanded-eight-sources-20261115'
REPORT=ROOT/'reports/ppe-expansion-build-20261115'

def main():
    assert not OUT.exists(),'Refuse to overwrite a release'
    sources=json.loads((REPORT/'source-snapshots.json').read_text());assert {s['project_id'] for s in sources}=={40,75,72}
    review=json.loads((REPORT/'shoe-review-index.json').read_text());assert len(review)==80
    shoe_families={f"72:{r['family']}" for r in review};assert len(shoe_families)==14
    rows=[]
    for r in map(json.loads,(BASE/'manifest.jsonl').read_text().splitlines()):
        rows.append(dict(project_id=r['project_id'],source_image=r['image'],source_label=r['label'],original_family=r['group'],file_name=r['file_name'],previous_split=r['split'],is_base=True,source_names=CLASSES))
    assert len(rows)==14536
    for s in sources:
        p=Path(s['snapshot']);names=(p/'classes.txt').read_text().splitlines();assert names==s['classes']
        images=sorted(x for x in (p/'images').rglob('*') if x.is_file());assert len(images)==s['images']
        for im in images:
            rel=im.relative_to(p/'images');lab=(p/'labels'/rel).with_suffix('.txt')
            rows.append(dict(project_id=s['project_id'],source_image=str(im),source_label=str(lab),original_family=f"{s['project_id']}:{family(im.name)}",file_name=im.name,previous_split=None,is_base=False,source_names=names))
    assert len(rows)==18150
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a
    hashes={};families={};omitted=Counter();edges=Counter();sizes={}
    for i,r in enumerate(rows):
        names=r.pop('source_names');r['quarantine']=f"{r['project_id']}:{family(r['file_name'])}" in shoe_families
        if r['project_id']==72:
            # These uncertain source categories occur only in held-out families, never negative training truth.
            names=['no_ambiguous_shoe' if n=='not_safety_shoe' else n for n in names]
        lines,skip,edge=parse(Path(r['source_label']),names);r['lines']=lines;omitted.update(skip);edges[r['project_id']]+=edge
        sha=digest(Path(r['source_image']));r.update(image_sha256=sha,source_label_sha256=digest(Path(r['source_label'])))
        if sha not in sizes:
            with Image.open(r['source_image']) as im:sizes[sha]=im.size;im.verify()
        r['width'],r['height']=sizes[sha]
        for key,seen in [(sha,hashes),(r['original_family'],families)]:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    exam=json.loads((ROOT/'reports/andromeda-noleak-all-models/exam/members.json').read_text());protected={m['sha256'] for m in exam['members']}
    index=json.loads((REPORT/'benchmark-family-index.json').read_text());assert len(index)==3783
    protected_families={r['original_family'] for r in index}
    rng=random.Random(42);keys=sorted(groups);rng.shuffle(keys);manifest=[];counts=Counter();physical_counts=Counter();support=defaultdict(Counter);sources_by_split=defaultdict(Counter);moves=[];conflicts=[];quarantine_count=0
    for k in keys:
        items=groups[k];previous={r['previous_split'] for r in items if r['previous_split']};forced=any(r['image_sha256'] in protected or f"{r['project_id']}:{family(r['file_name'])}" in protected_families for r in items);quarantine=any(r['quarantine'] for r in items)
        assert not {'val','test'}.issubset(previous),'New source bridges previous val/test groups; inspect before building'
        if forced:
            assert 'val' not in previous,'New benchmark family would invalidate fixed validation; inspect'
            split='test'
        elif 'test' in previous:split='test'
        elif 'val' in previous:split='val'
        elif quarantine:split='test'
        elif 'train' in previous:split='train'
        else:
            v=rng.random();split='train' if v<.8 else ('val' if v<.9 else 'test')
        assert not quarantine or split!='train'
        gid='group-'+hashlib.sha256('\n'.join(sorted({r['original_family'] for r in items})).encode()).hexdigest()[:20]
        variants=defaultdict(set)
        for r in items:variants[r['image_sha256']].add(hashlib.sha256('\n'.join(sorted(r['lines'])).encode()).hexdigest())
        conflicts.extend(dict(image_sha256=h,group=gid,label_variants=len(v)) for h,v in variants.items() if len(v)>1)
        for r in items:
            core=r['is_base'] and r['previous_split'] in ['val','test']
            if core:assert r['previous_split']==split
            subset=split if split=='train' or core else 'extra_'+split
            filename=Path(r['source_image']).name if r['is_base'] else f"{r['project_id']}_{Path(r['source_image']).name}"
            im=OUT/'images'/subset/filename;lab=OUT/'labels'/subset/Path(filename).with_suffix('.txt')
            if im.exists():
                filename=hashlib.sha256(r['source_image'].encode()).hexdigest()[:10]+'_'+filename;im=OUT/'images'/subset/filename;lab=OUT/'labels'/subset/Path(filename).with_suffix('.txt')
            im.parent.mkdir(parents=True,exist_ok=True);lab.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r['source_image'],im);assert digest(im)==r['image_sha256']
            lines=r.pop('lines');lab.write_text('\n'.join(lines)+'\n' if lines else '')
            support[subset].update(int(l.split()[0]) for l in lines);counts[split]+=1;physical_counts[subset]+=1;sources_by_split[r['project_id']][subset]+=1
            quarantine_count+=quarantine
            if r['previous_split'] and r['previous_split']!=split:moves.append(dict(image=r['source_image'],old=r['previous_split'],new=split,reason='connected source group protection'))
            manifest.append(dict(**r,image=str(im),label=str(lab),label_sha256=digest(lab),split=split,subset=subset,group=gid,core_evaluation=core,semantic_quarantine=quarantine,benchmark_exact_match=r['image_sha256'] in protected))
    assert len(manifest)==18150
    bg=defaultdict(set);bh=defaultdict(set)
    for r in manifest:
        bg[r['group']].add(r['split']);bh[r['image_sha256']].add(r['split'])
        assert r['split']=='test' or r['image_sha256'] not in protected
        assert r['split']=='test' or f"{r['project_id']}:{family(r['file_name'])}" not in protected_families
        assert not r['semantic_quarantine'] or r['subset'].startswith('extra_') or r['core_evaluation']
    assert all(len(s)==1 for s in bg.values()) and all(len(s)==1 for s in bh.values())
    assert physical_counts['val']==1337 and physical_counts['test']==2544
    prior=[json.loads(l) for l in (BASE/'manifest.jsonl').read_text().splitlines()]
    for split in ['val','test']:
        assert sorted((r['image_sha256'],r['label_sha256']) for r in prior if r['split']==split)==sorted((r['image_sha256'],r['label_sha256']) for r in manifest if r['subset']==split)
    for split in ['train','val','test']:assert set(support[split])==set(range(11))
    for subset,n in physical_counts.items():assert len(list((OUT/'images'/subset).iterdir()))==len(list((OUT/'labels'/subset).iterdir()))==n
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (OUT/'data.yaml').write_text(yaml.safe_dump(dict(path=str(OUT),train='images/train',val='images/val',test='images/test',names=CLASSES),sort_keys=False))
    summary=dict(total_images=18150,images=dict(physical_counts),logical_split_counts=dict(counts),classes=CLASSES,source_counts={str(k):dict(v) for k,v in sources_by_split.items()},box_support={s:{CLASSES[i]:v[i] for i in range(11)} for s,v in support.items()},group_leakage=0,exact_hash_split_leakage=0,benchmark_exact_leakage_train_val=0,benchmark_known_family_leakage_train_val=0,core_eval_hashes_and_labels_unchanged=True,core_validation_images=1337,core_test_images=2544,semantic_quarantine_images=quarantine_count,semantic_quarantine_source_families=14,omitted_boxes=omitted,edge_boxes=dict(edges),exact_duplicate_label_conflicts=len(conflicts),base_split_moves=len(moves),base=str(BASE),new_snapshots=sources,recipe_note='Same YOLO26s960 recipe and primary validation/test as previous expansion; train additions only. Extra heldouts are retained separately and not used for early stopping or primary test.',limitations=['No exhaustive near-duplicate/scene or label-completeness audit.','Partial-class source annotations can suppress unlabelled visible PPE; no automatic pseudo-labels or source repairs.','All images retained; uncertain shoe families kept in extra heldout, not treated as negative training truth.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'split-moves.json').write_text(json.dumps(moves,indent=2)+'\n');(OUT/'duplicate-label-conflicts.json').write_text(json.dumps(conflicts,indent=2)+'\n')
    (REPORT/'build-state.json').write_text(json.dumps(dict(status='complete',release=str(OUT),images=dict(physical_counts),total=18150),indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        (REPORT/'build-state.json').write_text(json.dumps(dict(status='failed',error=repr(e)))+'\n');raise
