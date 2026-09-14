#!/usr/bin/env python3
import json, glob, re
for f in sorted(glob.glob('tt_*.json')) + sorted(glob.glob('vllm_*.json')):
    d = json.load(open(f))
    body = re.sub(r'[!()\[\]`\r#>_*]', ' ', d.get('body') or '')
    body = re.sub(r'\s+', ' ', body)[:1000]
    print('=' * 15, f, d['number'], d['state'], d['title'])
    print(body)
    print()
