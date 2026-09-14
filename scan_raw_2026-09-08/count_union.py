#!/usr/bin/env python3
import re
doc = open('/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md', encoding='utf-8').read()
pat = re.compile(r'^### (\S+) `([^`]+)`[^\n]*\n\n([^\n]+)\n', re.M)
sets = {'config': set(), 'yes': set(), 'partial': set(), 'no': set(), 'hw_specific': set(), 'other_stack': set(), 'n/a': set()}
ids = []
for m in pat.finditer(doc):
    meta = m.group(3); fid = m.group(1)
    tm = re.search(r'触发\s*`([^`]+)`', meta)
    if tm:
        v = tm.group(1)
        if v in sets: sets[v].add(fid)
print('config|yes union:', len(sets['config'] | sets['yes']))
print('config|yes|partial:', len(sets['config'] | sets['yes'] | sets['partial']))
print('no|hw_specific:', len(sets['no'] | sets['hw_specific']))
print('other_stack total:', len(sets['other_stack']))
