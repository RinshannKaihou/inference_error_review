#!/usr/bin/env python3
import re, sys
sys.path.insert(0, '.')
from export_faults import parse_metadata_line
text = open('FAULT_MASTER_REFERENCE.zh.md').read()
entries = re.findall(r'^### (\S+) `([^`]+)`\n\n(.*?)$', text, re.M | re.S)
fam, conf, trig, cov = {}, {}, {}, {}
for fid, name, rest in entries:
    line = rest.split('\n')[0]
    m = parse_metadata_line(line)
    f = m.get('coverage', '')
    if f.startswith('var'):
        key = 'var'
    elif f:
        key = f
    else:
        key = '(empty)'
    cov[key] = cov.get(key, 0) + 1
    fm = re.match(r'`(\w+)`', line)
    fk = fm.group(1) if fm else '?'
    fam[fk] = fam.get(fk, 0) + 1
    conf[m.get('confidence', '?')] = conf.get(m.get('confidence', '?'), 0) + 1
    t = m.get('trigger', '?').rstrip('⚠')
    trig[t] = trig.get(t, 0) + 1
print('TOTAL', len(entries))
print('FAM', dict(sorted(fam.items(), key=lambda x: -x[1])))
print('CONF', conf)
print('TRIG', dict(sorted(trig.items(), key=lambda x: -x[1])))
print('COV', cov)
