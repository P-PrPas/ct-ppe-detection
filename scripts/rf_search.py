"""Search Roboflow Universe via API (web search pages are Cloudflare-blocked). usage: python3 -I rf_search.py "class:respirator" ...
Key from ~/.roboflow_key (never printed)."""
import sys, json, urllib.parse, urllib.request, pathlib
k = pathlib.Path("~/.roboflow_key").expanduser().read_text().strip()
for q in sys.argv[1:]:
    d = json.load(urllib.request.urlopen(f"https://api.roboflow.com/universe/search?q={urllib.parse.quote(q)}&api_key={k}", timeout=30))
    print(f"## {q} ({d.get('count')})")
    for r in d.get("results", []):
        print(r["url"], "|", r.get("images"), "imgs |", r.get("classes") if "classes" in r else "", "|", r.get("license", ""))
