#!/usr/bin/env python3
"""Dump bodies for selected issue numbers from cached search JSONs."""
import json, sys, glob

nums = [int(a) for a in sys.argv[1:]]
seen = {}
for f in sorted(glob.glob('/workspace/inference_error_review/.scan_20260906/q*.json')):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for it in d.get('items', []):
        if it['number'] in nums and it['number'] not in seen:
            seen[it['number']] = (f, it)

for n in nums:
    if n not in seen:
        print(f"===== #{n} NOT FOUND")
        continue
    f, it = seen[n]
    kind = 'PR' if 'pull_request' in it else 'I'
    body = (it.get('body') or '').replace('\r', '')
    print(f"===== #{n} [{kind}] ({f}) {it['title']}")
    print(f"url: {it['html_url']}")
    print(body[:2600])
    print()
