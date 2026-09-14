#!/usr/bin/env python3
import re
doc = open('/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md', encoding='utf-8').read()
# count entries per numbered category
cats = {}
for m in re.finditer(r'^### (\S+) `([^`]+)`', doc, re.M):
    fid = m.group(1)
    if re.match(r'^\d+\.\d+$', fid):
        c = fid.split('.')[0]
        cats[c] = cats.get(c, 0) + 1
total_numbered = sum(cats.values())
a = len(re.findall(r'^### A\d+ ', doc, re.M))
k = len(re.findall(r'^### K\d+\.\d+ ', doc, re.M))
print('numbered:', total_numbered, 'A:', a, 'K:', k, 'TOTAL:', total_numbered + a + k)
for c in sorted(cats, key=int):
    print(f'  cat {c}: {cats[c]}')
