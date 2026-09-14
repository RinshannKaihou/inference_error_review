#!/usr/bin/env python3
import re
doc = open('/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md', encoding='utf-8').read()
pat = re.compile(r'^### (\S+) `([^`]+)`[^\n]*\n\n([^\n]+)\n', re.M)
fam, trig, conf, cov = {}, {}, {}, {}
n = 0
for m in pat.finditer(doc):
    meta = m.group(3)
    n += 1
    # family: first token of meta line
    fm = re.match(r'`?([\w·\-]+)`?', meta)
    f = fm.group(1).split('·')[0] if fm else '?'
    fam[f] = fam.get(f, 0) + 1
    tm = re.search(r'触发\s*`([^`]+)`(⚠)?', meta)
    if tm:
        key = tm.group(1) + ('⚠' if tm.group(2) else '')
        trig[key] = trig.get(key, 0) + 1
    cm = re.search(r'置信度\s*`([^`]+)`', meta)
    if cm:
        conf[cm.group(1)] = conf.get(cm.group(1), 0) + 1
    vm = re.search(r'Cov:\s*`([^`]+)`', meta)
    if vm:
        v = vm.group(1).split('（')[0].split('(')[0].strip()
        cov[v if not v.startswith('var:') else 'var:x'] = cov.get(v if not v.startswith('var:') else 'var:x', 0) + 1
print('entries:', n)
print('FAMILY:', dict(sorted(fam.items(), key=lambda x: -x[1])))
print('TRIGGER:', dict(sorted(trig.items(), key=lambda x: -x[1])))
print('CONF:', dict(sorted(conf.items(), key=lambda x: -x[1])))
print('COV:', dict(sorted(cov.items(), key=lambda x: -x[1])))
