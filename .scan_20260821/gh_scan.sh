#!/bin/bash
# Daily GitHub scan - 2026-08-21
# 8 queries, 7s+ spacing to respect 10/min unauthenticated limit
OUT=/workspace/inference_error_review/.scan_20260821
mkdir -p $OUT

run_q() {
  local name="$1"; shift
  echo "=== $name ===" >&2
  curl -s -G "https://api.github.com/search/issues" \
    -H "Accept: application/vnd.github+json" \
    --data-urlencode "sort=created" \
    --data-urlencode "order=desc" \
    --data-urlencode "per_page=12" \
    "$@" > $OUT/$name.json
  sleep 8
}

run_q q1 'q=repo:vllm-project/vllm is:issue created:>2026-08-14 wrong OR incorrect OR corruption OR NaN in:title'

run_q q2 'q=repo:vllm-project/vllm is:issue created:>2026-08-14 quant OR quantization wrong in:title'

run_q q3 'q=repo:sgl-project/sglang is:issue created:>2026-08-14 wrong OR incorrect OR corruption OR NaN in:title'

run_q q4 'q=repo:NVIDIA/TensorRT-LLM is:issue created:>2026-08-14 wrong OR incorrect OR correctness OR NaN in:title'

run_q q5 'q=repo:huggingface/transformers is:issue created:>2026-08-14 wrong output OR wrong result OR RoPE in:title'

run_q q6 'q=repo:triton-lang/triton is:issue created:>2026-08-14 incorrect OR numerical OR precision OR mismatch in:title'

run_q q7 'q=repo:tenstorrent/tt-metal is:issue created:>2026-08-14 numerical OR pcc OR mismatch OR incorrect in:title'

run_q q8 'q=repo:vllm-project/vllm is:issue created:>2026-08-14 KV cache OR prefix cache OR speculative in:title'

echo DONE
