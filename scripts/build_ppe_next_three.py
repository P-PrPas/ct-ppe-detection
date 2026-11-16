"""Build the frozen next-three release with its recorded source policy."""

import json
from pathlib import Path

from dataset_release import build_append

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/ppe-expanded-five-sources-20261114"
OUT = ROOT / "data/ppe-expanded-eight-sources-20261115"
REPORT = ROOT / "reports/ppe-expansion-build-20261115"


def main():
    build_append(BASE, OUT, REPORT, "next-three")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        (REPORT / "build-state.json").write_text(
            json.dumps(dict(status="failed", error=repr(e))) + "\n"
        )
        raise
