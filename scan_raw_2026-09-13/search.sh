#!/bin/bash
S="https://api.github.com/search/issues"
D="2026-09-11"
run() { # name, query
  echo "=== $1: $2"
  curl -s --http1.1 -o "q_${1}.json" -w "%{http_code}\n" "$S?q=$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$2")&sort=created&order=desc&per_page=15"
  sleep 7.5
}
run 01_vllm_bug    "repo:vllm-project/vllm is:issue label:bug created:>=${D}"
run 02_vllm_wrong  "repo:vllm-project/vllm is:issue wrong output created:>=${D}"
run 03_sglang_wrong "repo:sgl-project/sglang is:issue created:>=${D} wrong OR incorrect OR corruption"
run 04_hf_wrong    "repo:huggingface/transformers is:issue created:>=${D} wrong OR incorrect OR NaN"
run 05_trtllm_wrong "repo:NVIDIA/TensorRT-LLM is:issue created:>=${D} wrong OR incorrect OR corruption"
run 06_triton_wrong "repo:triton-lang/triton is:issue created:>=${D} wrong OR incorrect OR NaN"
run 07_ttmetal_bug "repo:tenstorrent/tt-metal is:issue label:bug created:>=${D}"
run 08_flashinfer_wrong "repo:flashinfer-ai/flashinfer is:issue created:>=${D} wrong OR incorrect OR NaN"
run 09_vllm_quant  "repo:vllm-project/vllm is:issue created:>=${D} quantization OR quant"
run 10_sglang_kv   "repo:sgl-project/sglang is:issue created:>=${D} KV cache OR attention OR speculative"
