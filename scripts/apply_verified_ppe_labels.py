"""Apply parent-verified final label files; dry run unless --apply is supplied."""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

POOL = Path('/opt/mount/project/data')
REPORTS = Path(__file__).resolve().parents[1] / 'reports/ppe-preparation-2026-11-08'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_labels(data, valid_classes):
    for number, line in enumerate(data.decode('utf-8').splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'Invalid label row {number}')
        cls = int(fields[0])
        cx, cy, w, h = map(float, fields[1:])
        if cls not in valid_classes or not all(math.isfinite(v) for v in (cx, cy, w, h)):
            raise ValueError(f'Invalid class or coordinates at row {number}')
        if not (w > 0 and h > 0 and cx-w/2 >= -1e-6 and cy-h/2 >= -1e-6
                and cx+w/2 <= 1.000001 and cy+h/2 <= 1.000001):
            raise ValueError(f'Out-of-bounds box at row {number}')


def atomic_write(path, data, mode=0o644):
    fd, temporary = tempfile.mkstemp(prefix='.ppe-verified-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            os.fchmod(stream.fileno(), mode)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    raw = json.loads(args.plan.read_text())
    plans = raw if isinstance(raw, list) else [raw]
    prepared = []
    seen = set()
    for plan in plans:
        if plan.get('parent_visual_verified') is not True:
            raise ValueError('Parent visual verification is required')
        path = Path(plan['source_label_path'])
        relative = path.resolve().relative_to(POOL.resolve())
        if len(relative.parts) != 3 or relative.parts[1] != 'labels' or path.suffix != '.txt':
            raise ValueError('Only dataset label files in the shared pool may be edited')
        if path.is_symlink() or path in seen:
            raise ValueError('Symlink or duplicate source target')
        seen.add(path)
        original = path.read_bytes()
        proposed = Path(plan['proposed_label_path']).read_bytes()
        if digest(original) != plan['source_label_sha256']:
            raise ValueError(f'Source changed: {path.name}')
        if digest(proposed) != plan['proposed_final_label_sha256']:
            raise ValueError(f'Proposal changed: {path.name}')
        if not Path(plan['parent_evidence']).is_file():
            raise ValueError('Missing visual evidence')
        names = yaml.safe_load((path.parent.parent / 'data.yaml').read_text())['names']
        valid_classes = set(range(len(names))) if isinstance(names, list) else {int(k) for k in names}
        validate_labels(proposed, valid_classes)
        prepared.append((plan, path, original, proposed, relative))
    if not args.apply:
        print(json.dumps({'dry_run': True, 'validated_files': len(prepared)}))
        return
    usage = json.loads((REPORTS / 'usage-latest.json').read_text())
    age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(usage['checked_at'])).total_seconds()
    if usage['decision'] != 'continue' or not 0 <= age < 1800 or (REPORTS / 'OVERNIGHT_CUTOFF').exists():
        raise ValueError('Fresh quota continue decision required')
    backup = REPORTS / 'pool-edit-backups' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True)
    journal = []
    for plan, path, original, proposed, relative in prepared:
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(original)
        journal.append({**plan, 'backup': str(saved), 'status': 'backed_up'})
    journal_path = backup / 'journal.json'
    atomic_write(journal_path, json.dumps(journal, indent=2).encode())
    for index, (plan, path, original, proposed, _) in enumerate(prepared):
        if path.read_bytes() != original:
            raise ValueError(f'Concurrent source change; backups retained: {path.name}')
        atomic_write(path, proposed, path.stat().st_mode & 0o777)
        if path.read_bytes() != proposed:
            raise ValueError(f'Readback mismatch; backup retained: {path.name}')
        journal[index]['status'] = 'applied_readback_verified'
        atomic_write(journal_path, json.dumps(journal, indent=2).encode())
    print(json.dumps({'applied_files': len(prepared), 'journal': str(journal_path)}))


if __name__ == '__main__':
    main()
