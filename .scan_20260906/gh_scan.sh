#!/bin/bash
# Daily GitHub scan - 2026-09-06 (fixed: pass q via --data-urlencode)
# 10 queries, 8s+ spacing to respect 10/min unauthenticated limit
OUT=/workspace/inference_error_review/.scan_20260906
mkdir -p $OUT

run_q() {
  local name="$1"; local query="$2"
  echo "=== $name ===" >&2
  curl -s -G "https://api.github.com/search/issues" \
    -H "Accept: application/vnd.github+json" \
    --data-urlencode "q=$query" \
    --data-urlencode "sort=created" \
    --data-urlencode "order=desc" \
    --data-urlencode "per_page=15" \
    > $OUT/$name.json
  sleep 8
}

run_q q1 'repo:vllm-project/vllm created:>2026-09-05 wrong OR incorrect OR corruption OR NaN OR silent'

run_q q2 'repo:sgl-project/sglang created:>2026-09-05 bug OR wrong OR corrupt OR NaN OR garbage'

run_q q3 'repo:huggingface/transformers created:>2026-09-05 wrong OR incorrect OR corrupt OR NaN'

run_q q4 'repo:NVIDIA/TensorRT-LLM created:>2026-09-05 bug OR wrong OR corrupt OR NaN OR correctness'

run_q q5 'repo:triton-lang/triton created:>2026-09-05 wrong OR incorrect OR miscompile OR corrupt'

run_q q6 'repo:tenstorrent/tt-metal created:>2026-09-05 wrong OR incorrect OR corrupt OR NaN OR mismatch'

run_q q7 'repo:vllm-project/vllm created:>2026-09-05 quant OR FP8 OR NVFP4 OR KV cache label:bug'

run_q q8 'repo:huggingface/transformers created:>2026-09-05 tokenizer OR rope OR config label:bug'

run_q q9 'repo:vllm-project/vllm is:issue created:>2026-09-05 correctness OR accuracy OR "wrong result"'

run_q q10 'repo:sgl-project/sglang created:>2026-09-05 correctness OR accuracy OR silent OR deterministic'

echo DONE
