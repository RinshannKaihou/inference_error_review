#!/bin/bash
OUTDIR="/workspace/inference_error_review/scan_raw"
run_arxiv() {
  local name="$1"; local q="$2"
  echo "=== $name ==="
  curl -sL "http://export.arxiv.org/api/query?search_query=${q}&sortBy=submittedDate&sortOrder=descending&max_results=5" \
    | grep -oE '<entry>.*?</entry>' | sed 's/<entry>//;s/<\/entry>//' \
    | grep -oE '<title>[^<]+</title>|<id>[^<]+</id>|<published>[^<]+</published>' \
    | head -45 > "$OUTDIR/$name.txt" 2>/dev/null
  # fallback: just grab title+id pairs
  if [ ! -s "$OUTDIR/$name.txt" ]; then
    curl -sL "http://export.arxiv.org/api/query?search_query=${q}&sortBy=submittedDate&sortOrder=descending&max_results=5" \
      | python3 -c "
import sys,re
d=sys.stdin.read()
entries=re.findall(r'<entry>(.*?)</entry>',d,re.S)
for e in entries[:5]:
    t=re.search(r'<title>(.*?)</title>',e,re.S)
    i=re.search(r'<id>(.*?)</id>',e,re.S)
    p=re.search(r'<published>(.*?)</published>',e,re.S)
    if t and i:
        print(f'{(t.group(1).strip()[:120])}|{(i.group(1).strip())}|{(p.group(1).strip() if p else \"\")}')
" > "$OUTDIR/$name.txt" 2>/dev/null
  fi
  cat "$OUTDIR/$name.txt"
  echo
  sleep 4
}
run_arxiv "arxiv_sdc_llm" 'all:%22silent+data+corruption%22+AND+all:LLM'
run_arxiv "arxiv_bitflip" 'all:%22bit+flip%22+AND+all:LLM+AND+all:weight'
run_arxiv "arxiv_kv_cache" 'all:%22KV+cache%22+AND+all:corruption+AND+all:inference'
run_arxiv "arxiv_correctness" 'all:%22LLM+inference%22+AND+all:correctness'
run_arxiv "arxiv_gpu_soft_error" 'all:%22GPU+soft+error%22+AND+all:%22machine+learning%22'
echo "DONE"
