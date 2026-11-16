"""Build a local frozen release from full10030 and five copied pool exports."""
import json,hashlib,math,random,re,shutil
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image
import yaml
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/ppe-expansion-build-20261114'
OUT=ROOT/'data/ppe-expanded-five-sources-20261114'
BASE=ROOT/'data/ppe-full-current-20261109'
CLASSES=['Helmet','Vest','Mask_Gas','Goggles','Coverall','Gloves','Mask_Face','Shoes','Safety_Belt','Cap','Ear_Protection']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def family(name):return re.split(r'\.rf\.',name,flags=re.I)[0] if '.rf.' in name.lower() else Path(name).stem

def parse(path,names):
    lines=[];omitted=Counter();edge_boxes=0
    assert path.is_file(),path
    for text in path.read_text().splitlines():
        if not text.strip():continue
        a=text.split();assert len(a)==5,(path,text)
        c=int(a[0]);assert 0<=c<len(names),(path,c,names)
        vals=list(map(float,a[1:]));assert all(math.isfinite(v) for v in vals),(path,text)
        x,y,w,h=vals;assert 0<=x<=1 and 0<=y<=1 and 0<w<=1.00001 and 0<h<=1.00001,(path,text)
        edge_boxes+=x-w/2<-.00001 or y-h/2<-.00001 or x+w/2>1.00001 or y+h/2>1.00001
        name=names[c]
        if name.lower().startswith('no_'):
            omitted[name]+=1;continue
        assert name in CLASSES,(path,name)
        lines.append(str(CLASSES.index(name))+' '+' '.join(a[1:]))
    return lines,omitted,edge_boxes

def main():
    assert family('photo_jpg.rf.abc.jpg')==family('photo_jpg.rf.def.jpg')
    assert len(set(CLASSES))==11
    assert not OUT.exists(),'Release already exists; inspect it instead of overwriting'
    sources=json.loads((REPORT/'source-snapshots.json').read_text());assert {r['project_id'] for r in sources}=={71,64,77,82,65}
    rows=[];omitted=Counter();edges=Counter()
    for r in map(json.loads,(BASE/'manifest.jsonl').read_text().splitlines()):
        rows.append(dict(project_id=r['project_id'],source_image=r['image'],source_label=r['label'],original_family=r['group'],file_name=r['file_name'],previous_split=r['split'],origin='full10030'))
    assert len(rows)==10030
    for s in sources:
        p=Path(s['snapshot']);names=(p/'classes.txt').read_text().splitlines();assert names==s['classes']
        imgs=sorted(x for x in (p/'images').rglob('*') if x.is_file());assert len(imgs)==s['images']
        for im in imgs:
            rel=im.relative_to(p/'images');lab=(p/'labels'/rel).with_suffix('.txt')
            rows.append(dict(project_id=s['project_id'],source_image=str(im),source_label=str(lab),original_family=f"{s['project_id']}:{family(im.name)}",file_name=im.name,previous_split=None,origin='five-source-pool',export_relative_image=str(rel),source_names=names))
    assert len(rows)==14536
    # Join exact duplicate images AND source augmentation/original families without removing images.
    parent=list(range(len(rows)))
    def find(i):
        while i!=parent[i]:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    seen_hash={};seen_family={};size_cache={}
    for i,r in enumerate(rows):
        image=Path(r['source_image']);label=Path(r['source_label']);sha=digest(image)
        if sha not in size_cache:
            with Image.open(image) as im:
                size_cache[sha]=im.size;im.verify()
        r.update(image_sha256=sha,source_label_sha256=digest(label),width=size_cache[sha][0],height=size_cache[sha][1])
        lines,skipped,edge=parse(label,r.pop('source_names',CLASSES));r['lines']=lines;omitted.update(skipped);edges[r['project_id']]+=edge
        for key,seen in [(sha,seen_hash),(r['original_family'],seen_family)]:
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[find(i)].append(r)
    exam=json.loads((ROOT/'reports/andromeda-noleak-all-models/exam/members.json').read_text());protected={m['sha256'] for m in exam['members']};assert len(exam['members'])==3783
    benchmark_index=json.loads((REPORT/'benchmark-family-index.json').read_text())
    assert len(benchmark_index)==3783
    protected_families={r['original_family'] for r in benchmark_index}
    rng=random.Random(42);keys=sorted(groups);rng.shuffle(keys);rank={'train':0,'val':1,'test':2};moved=[];counts=Counter();support=defaultdict(Counter);source_counts=defaultdict(Counter);negative=Counter();manifest=[];protected_groups=0;label_conflicts=[]
    for key in keys:
        items=groups[key];forced=any(r['image_sha256'] in protected or f"{r['project_id']}:{family(r['file_name'])}" in protected_families for r in items)
        previous={r['previous_split'] for r in items if r['previous_split']}
        if forced:split='test';protected_groups+=1
        elif previous:split=max(previous,key=lambda x:rank[x])
        else:
            v=rng.random();split='train' if v<.8 else ('val' if v<.9 else 'test')
        gid='group-'+hashlib.sha256('\n'.join(sorted({r['original_family'] for r in items})).encode()).hexdigest()[:20]
        per_hash=defaultdict(set)
        for r in items:per_hash[r['image_sha256']].add(hashlib.sha256('\n'.join(sorted(r['lines'])).encode()).hexdigest())
        label_conflicts.extend(dict(image_sha256=h,group=gid,label_variants=len(v),policy='retain original authoritative labels in same split') for h,v in per_hash.items() if len(v)>1)
        for r in items:
            if r['previous_split'] and r['previous_split']!=split:moved.append(dict(source_image=r['source_image'],old=r['previous_split'],new=split,reason='benchmark original group' if forced else 'exact-duplicate split isolation',group=gid))
            filename=f"{r['project_id']}_{Path(r['source_image']).name}"
            image=OUT/'images'/split/filename;label=OUT/'labels'/split/Path(filename).with_suffix('.txt')
            # Prefix a path hash only when one source legitimately has repeated basenames in different folders.
            if image.exists():
                filename=f"{r['project_id']}_{hashlib.sha256(r['source_image'].encode()).hexdigest()[:10]}_{Path(r['source_image']).name}"
                image=OUT/'images'/split/filename;label=OUT/'labels'/split/Path(filename).with_suffix('.txt')
            image.parent.mkdir(parents=True,exist_ok=True);label.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(r['source_image'],image);assert digest(image)==r['image_sha256']
            lines=r.pop('lines');label.write_text('\n'.join(lines)+'\n' if lines else '')
            ids=[int(l.split()[0]) for l in lines];support[split].update(ids);negative[split]+=not bool(ids);counts[split]+=1;source_counts[r['project_id']][split]+=1
            manifest.append(dict(**r,image=str(image),label=str(label),split=split,group=gid,label_sha256=digest(label),benchmark_exact_match=r['image_sha256'] in protected))
    assert len(manifest)==14536 and sum(counts.values())==14536
    by_group=defaultdict(set);by_hash=defaultdict(set)
    for r in manifest:
        by_group[r['group']].add(r['split']);by_hash[r['image_sha256']].add(r['split'])
        assert r['split']=='test' or r['image_sha256'] not in protected
        assert r['split']=='test' or f"{r['project_id']}:{family(r['file_name'])}" not in protected_families
    assert all(len(v)==1 for v in by_group.values()) and all(len(v)==1 for v in by_hash.values())
    for split in ['train','val','test']:
        assert set(support[split])==set(range(11)),(split,support[split])
        assert len(list((OUT/'images'/split).iterdir()))==counts[split]
        assert len(list((OUT/'labels'/split).iterdir()))==counts[split]
    (OUT/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    (OUT/'data.yaml').write_text(yaml.safe_dump(dict(path=str(OUT),train='images/train',val='images/val',test='images/test',names=CLASSES),sort_keys=False))
    summary=dict(images=counts,total_images=len(manifest),source_counts={str(k):dict(v) for k,v in source_counts.items()},classes=CLASSES,box_support={s:{CLASSES[i]:v[i] for i in range(11)} for s,v in support.items()},negative_images=negative,original_groups=len(groups),exact_image_hashes=len(seen_hash),group_leakage=0,exact_hash_split_leakage=0,benchmark_exact_leakage_train_val=0,benchmark_protected_groups=protected_groups,base_images_moved=len(moved),omitted_no_boxes=omitted,source_boxes_extending_beyond_image=edges,exact_duplicate_label_conflicts=len(label_conflicts),benchmark_manifest=exam['manifest_sha256'],snapshots=sources,policy='All14536selectedpool/baseimages retained inclproducts/negatives; no source edits, annotation repair, pseudolabeling, personcrop or source upload. Source-frozen positive labels authoritative.',limitations=['Cross-source near-duplicate/original scene families not independently audited beyond known within-source families and exact hashes.','Pool metadata flags near_test_images; absence of exact-hash leakage is not proof of no near-duplicate leakage.','New sources have partial-class labels; visible unlabelled other PPE may be treated as background. No completeness claim.','Base labels contain known quality issues; unchanged.','This experiment changes both dataset and necessary benchmark-protecting split assignments, so not a pure data-only comparison.'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'split-moves.json').write_text(json.dumps(moved,indent=2)+'\n');(OUT/'duplicate-label-conflicts.json').write_text(json.dumps(label_conflicts,indent=2)+'\n')
    (REPORT/'build-state.json').write_text(json.dumps(dict(status='complete',release=str(OUT),images=counts,total=14536),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        REPORT.mkdir(exist_ok=True,parents=True);(REPORT/'build-state.json').write_text(json.dumps(dict(status='failed',error=repr(e)))+'\n');raise
