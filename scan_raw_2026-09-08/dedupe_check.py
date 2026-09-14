#!/usr/bin/env python3
import re, sys
doc = open('/workspace/inference_error_review/FAULT_MASTER_REFERENCE.zh.md', encoding='utf-8').read()
ids = ['2607.25227','2608.05863','2606.28958','2604.00726','2502.12340','2104.09455','2608.30769','2608.07408','2608.11034','2604.10390','2608.29745','2608.25276','2608.23774','2601.19912','2608.20397','2605.29873']
for i in ids:
    print(f"{i}: {'IN DOC' if i in doc else 'NOT IN DOC'}")
print()
nums = ['55894','55927','55951','55827','38561','38536','38452','38428','18892','18891','55798','55797','55724','55723','55800','55799','55825','55815']
for n in nums:
    c = doc.count(f'#{n}')
    print(f"#{n}: {'ALREADY ('+str(c)+' refs)' if c else 'NEW'}")
