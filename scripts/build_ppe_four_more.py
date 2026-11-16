"""Build the frozen four-more release with its recorded source policy."""

import json
from pathlib import Path

from dataset_release import build_append

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/ppe-expanded-eight-sources-20261115"
OUT = ROOT / "data/ppe-expanded-22709-20261116"
REPORT = ROOT / "reports/ppe-expansion-build-20261116"


def main():
    build_append(BASE, OUT, REPORT, "four-more")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        (REPORT / "build-state.json").write_text(
            json.dumps(dict(status="failed", error=repr(e))) + "\n"
        )
        raise
