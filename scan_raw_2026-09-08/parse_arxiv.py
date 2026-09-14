#!/usr/bin/env python3
import re, glob
for f in sorted(glob.glob('/workspace/inference_error_review/scan_raw_2026-09-08/arxiv*.xml')):
    print(f"===== {f}")
    xml = open(f, encoding='utf-8', errors='replace').read()
    entries = re.findall(r'<entry>(.*?)</entry>', xml, re.S)
    for e in entries:
        aid = re.search(r'<id>(.*?)</id>', e)
        title = re.search(r'<title>(.*?)</title>', e, re.S)
        pub = re.search(r'<published>(.*?)</published>', e)
        t = re.sub(r'\s+', ' ', title.group(1)).strip() if title else '?'
        print(f"- [{pub.group(1)[:10] if pub else '?'}] {t} | {aid.group(1) if aid else '?'}")
