"""Train22709 expansion using previous recipe and frozen core evaluation sets."""
import json,time
from pathlib import Path
import train_ppe_expanded as experiment
ROOT=Path(__file__).resolve().parents[1]
experiment.DATA=ROOT/'data/ppe-expanded-22709-20261116'
experiment.RUN=ROOT/'runs/ppe-expanded-22709-yolo26l-960-20261116'
experiment.EXPECTED_IMAGES=22709
experiment.BASE_WEIGHTS=ROOT/'models/yolo26l.pt'
experiment.TRAIN_BATCH=2
experiment.EVAL_BATCH=2
if __name__=='__main__':
    try:
        summary=json.loads((experiment.DATA/'summary.json').read_text())
        assert summary['core_eval_hashes_and_labels_unchanged'] and summary['core_validation_images']==1337 and summary['core_test_images']==2544
        experiment.main()
    except BaseException as e:
        experiment.RUN.mkdir(exist_ok=True)
        (experiment.RUN/'training-state.json').write_text(json.dumps(dict(status='failed',error=repr(e),updated_at=time.time()))+'\n')
        raise
