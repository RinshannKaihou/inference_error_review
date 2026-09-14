#!/usr/bin/env python3
import re, sys
path = "/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md"
text = open(path, encoding="utf-8").read()
lines = text.split("\n")
fam = {}; trig = {}; conf = {}; cov = {}
entries = 0
i = 0
while i < len(lines):
    if re.match(r"^### (?:[0-9K])", lines[i]):
        entries += 1
        # summary line = next non-empty line
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        s = lines[j] if j < len(lines) else ""
        m = re.match(r"^`([a-z_]+)`", s.strip())
        if m: fam[m.group(1)] = fam.get(m.group(1), 0) + 1
        else: fam["<none>"] = fam.get("<none>", 0) + 1
        m = re.search(r"触发 `([a-zA-Z_]+|[^`·]+)`", s)
        t = m.group(1).strip() if m else "<none>"
        trig[t] = trig.get(t, 0) + 1
        m = re.search(r"置信度 `([a-z]+)`", s)
        c = m.group(1) if m else "<none>"
        conf[c] = conf.get(c, 0) + 1
        m = re.search(r"Cov: `([^`]+)`", s)
        v = m.group(1) if m else "<none>"
        if v.startswith("var:"): cov["var"] = cov.get("var", 0) + 1
        elif v == "NEW" or v.startswith("NEW"): cov["NEW"] = cov.get("NEW", 0) + 1
        else: cov["other"] = cov.get("other", 0) + 1
    i += 1
print("entries:", entries)
print("family:", dict(sorted(fam.items(), key=lambda x: -x[1])))
print("trigger:", dict(sorted(trig.items(), key=lambda x: -x[1])))
print("confidence:", conf)
print("cov:", cov)
