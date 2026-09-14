#!/usr/bin/env python3
import json, re
d = json.load(open('hf_48051.json'))
print(d['number'], d['state'], d['title'])
body = re.sub(r'[!()\[\]`\r#>_*]', ' ', d.get('body') or '')
print(re.sub(r'\s+', ' ', body)[:800])
