import re
text = open('FAULT_MASTER_REFERENCE.zh.md').read()
headers = re.findall(r'^### ([\dKA][\d.]*) ', text, re.M)
total = len(headers)
a = len([h for h in headers if h.startswith('A')])
k = len([h for h in headers if h.startswith('K')])
n = len([h for h in headers if not h.startswith('A') and not h.startswith('K')])
print(f"total={total} L0={a} L1={n} L2={k} (expect 696/17/565/114)")
# per-category
cats = {}
for h in headers:
    m = re.match(r'^(\d+)\.', h)
    if m: cats[int(m.group(1))] = cats.get(int(m.group(1)), 0) + 1
print("L1 per-cat:", {c: cats.get(c,0) for c in range(1,16)})
# trigger/family/cov/confidence recount from metadata lines
metas = re.findall(r'^(`[^`]+`)?\s*.*?Cov:\s*`([^`]+)`.*?触发\s*`([^`]+)`.*?置信度\s*`([^`]+)`', text, re.M)
from collections import Counter
covc = Counter(); trigc = Counter(); confc = Counter()
for m in re.finditer(r'Cov:\s*`([^`]+)`.*?触发\s*`([^`]+)`(⚠)?.*?置信度\s*`([^`]+)`', text):
    covc[m.group(1).split('（')[0].split('(')[0]] += 1
    trigc[m.group(2)] += 1
    confc[m.group(4)] += 1
print("Cov:", dict(covc))
print("Trigger:", dict(trigc))
print("Conf:", dict(confc))
# family recount
famc = Counter()
for m in re.finditer(r'\n(`infra_off|`model_on|`ambiguous|`n/a)[^`\n]*', text):
    pass
fams = Counter(re.findall(r'\n(`infra_off|`model_on|`ambiguous|`n/a)', text))
print("Family(raw line-start):", dict(fams))
