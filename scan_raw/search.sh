#!/bin/bash
set -u
OUTDIR="/workspace/inference_error_review/scan_raw"
SINCE="2026-07-28"

run_gh() {
  local name="$1"; local q="$2"; shift 2
  echo "=== $name ===" 
  curl -s -H "Accept: application/vnd.github+json" \
    "https://api.github.com/search/issues?q=${q}&sort=created&order=desc&per_page=15" \
    | jq -r '.items[]? | "\(.number)|\(.title)|\(.html_url)|\(.created_at)|\(.state)|\([.labels[].name] | join(","))"' \
    > "$OUTDIR/$name.txt" 2>/dev/null
  wc -l < "$OUTDIR/$name.txt"
  sleep 8
}

# 1. vllm correctness bugs last 7 days
run_gh "vllm_correctness" "repo:vllm-project/vllm+label:bug+created:>${SINCE}"
# 2. sglang correctness
run_gh "sglang_correctness" "repo:sgl-project/sglang+label:bug+created:>${SINCE}"
# 3. transformers correctness/wrong-output
run_gh "transformers_wrong" "repo:huggingface/transformers+wrong+inference+created:>${SINCE}"
# 4. TRT-LLM correctness
run_gh "trtllm_correctness" "repo:NVIDIA/TensorRT-LLM+correctness+created:>${SINCE}"
# 5. triton correctness
run_gh "triton_correctness" "repo:triton-lang/triton+correctness+created:>${SINCE}"
# 6. tt-metal numerical/correctness
run_gh "ttmetal_correctness" "repo:tenstorrent/tt-metal+correctness+created:>${SINCE}"
# 7. Cross-repo: silent data corruption / NaN / quantization
run_gh "xrepo_sd_nan" "silent+data+corruption+OR+NaN+inference+created:>${SINCE}"
# 8. Cross-repo: KV cache corruption
run_gh "xrepo_kv_cache" "KV+cache+corruption+created:>${SINCE}"
# 9. Cross-repo: quantization wrong result / bit flip
run_gh "xrepo_quant_bitflip" "quantization+wrong+result+OR+bit+flip+created:>${SINCE}"
# 10. vllm numerical/non-deterministic
run_gh "vllm_nondeterminism" "repo:vllm-project/vllm+numerical+OR+non-deterministic+created:>${SINCE}"
echo "DONE"
