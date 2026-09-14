#!/usr/bin/env python3
"""Extract the bug-description part (after </details>) for selected issues."""
import json, sys, glob, re

nums = [int(a) for a in sys.argv[1:]]
seen = {}
for f in sorted(glob.glob('/workspace/inference_error_review/.scan_20260906/q*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for it in d.get('items', []):
        if it['number'] in nums and it['number'] not in seen:
            seen[it['number']] = it

for n in nums:
    if n not in seen:
        print(f"===== #{n} NOT FOUND\n")
        continue
    it = seen[n]
    kind = 'PR' if 'pull_request' in it else 'I'
    body = (it.get('body') or '').replace('\r', '')
    # skip past the last </details> if present (env block), else use body as-is
    if '</details>' in body:
        idx = body.rfind('</details>')
        seg = body[idx + len('</details>'):]
    else:
        seg = body
    print(f"===== #{n} [{kind}] {it['title']}")
    print(seg.strip()[:3200])
    print()
