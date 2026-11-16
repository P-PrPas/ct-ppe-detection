"""Print image + per-class annotation counts for Roboflow Universe projects (no download).
usage: python3 -I rf_stats.py <workspace/project> ...   (key read from ~/.roboflow_key)"""

import json
import os
import sys
import urllib.error
import urllib.request

k = open(os.path.expanduser("~/.roboflow_key")).read().strip()
out = {}
for slug in sys.argv[1:]:
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(
                f"https://api.roboflow.com/{slug}?api_key={k}",
                headers={"User-Agent": "Mozilla/5.0"},
            ),
            timeout=30,
        )
        p = json.load(r)["project"]
        out[slug] = {
            "images": p["images"],
            "license": p.get("license"),
            "classes": p["classes"],
        }
    except urllib.error.HTTPError as e:
        out[slug] = {"error": e.code}
print(json.dumps(out, indent=1))
