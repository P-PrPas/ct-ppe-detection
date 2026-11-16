"""Read recent Codex rate-limit events without exposing conversation or credentials."""

import datetime as dt
import json
from pathlib import Path


def remaining(rate):
    return min(
        100 - float(rate[k]["used_percent"])
        for k in ("primary", "secondary")
        if rate.get(k) and rate[k].get("used_percent") is not None
    )


if __name__ == "__main__":
    assert remaining({"primary": {"used_percent": 30}}) == 70
    assert (
        remaining({"primary": {"used_percent": 15}, "secondary": {"used_percent": 31}})
        == 69
    )
    files = sorted(
        Path("/root/.codex/sessions").glob("**/*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:12]
    events = []
    for path in files:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 2_000_000))
            for line in stream:
                try:
                    row = json.loads(line)
                    payload = row.get("payload", {})
                    rate = payload.get("rate_limits")
                    if (
                        row.get("type") == "event_msg"
                        and payload.get("type") == "token_count"
                        and rate
                    ):
                        events.append(
                            {"timestamp": row["timestamp"], "rate_limits": rate}
                        )
                except (ValueError, KeyError):
                    continue
    latest = max(events, key=lambda x: x["timestamp"]) if events else None
    now = dt.datetime.now(dt.timezone.utc)
    stale = (
        latest is None
        or (
            now - dt.datetime.fromisoformat(latest["timestamp"].replace("Z", "+00:00"))
        ).total_seconds()
        > 3600
    )
    percent = remaining(latest["rate_limits"]) if latest else None
    result = {
        "checked_at": now.isoformat(),
        "latest_event": latest,
        "remaining_percent": percent,
        "stale": stale,
        "cutoff": percent is not None and percent < 70,
        "decision": "hold" if stale else ("cutoff" if percent < 70 else "continue"),
    }
    target = Path("reports/ppe-preparation-2026-11-08/usage-latest.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
