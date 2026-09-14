#!/bin/bash
cd /workspace/inference_error_review/scan_raw_2026-09-08
curl -sL "http://export.arxiv.org/api/query?search_query=all:%22silent+data+corruption%22+AND+all:LLM&sortBy=submittedDate&sortOrder=descending&max_results=5" -o arxiv1_sdc.xml
sleep 3
curl -sL "http://export.arxiv.org/api/query?search_query=all:%22bit+flip%22+AND+all:LLM+inference&sortBy=submittedDate&sortOrder=descending&max_results=5" -o arxiv2_bitflip.xml
sleep 3
curl -sL "http://export.arxiv.org/api/query?search_query=all:%22KV+cache%22+AND+all:corruption&sortBy=submittedDate&sortOrder=descending&max_results=5" -o arxiv3_kvcache.xml
sleep 3
curl -sL "http://export.arxiv.org/api/query?search_query=all:LLM+inference+AND+all:correctness&sortBy=submittedDate&sortOrder=descending&max_results=8" -o arxiv4_correctness.xml
sleep 3
curl -sL "http://export.arxiv.org/api/query?search_query=all:GPU+AND+all:%22soft+error%22+AND+all:inference&sortBy=submittedDate&sortOrder=descending&max_results=5" -o arxiv5_gpuerr.xml
for f in arxiv*; do echo "$f: $(grep -c '<entry>' $f 2>/dev/null || echo 0)"; done
