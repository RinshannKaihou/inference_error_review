#!/bin/bash
# Daily scan script for 2026-08-16 (retry with fixed jq filter file)
SINCE=$(date -u -d '7 days ago' '+%Y-%m-%d')
OUT=/workspace/inference_error_review/scan_raw_2026-08-16
FILT=$OUT/filter.jq

gh_search() {
  local name="$1"; local q="$2"
  echo "=== $name ==="
  curl -s "https://api.github.com/search/issues?q=$q&sort=created&order=desc&per_page=12" > "$OUT/$name.json"
  jq -r -f "$FILT" "$OUT/$name.json" > "$OUT/$name.txt"
  cat "$OUT/$name.txt"
  sleep 8
}

gh_search v1_vllm_bug "repo%3Avllm-project%2Fvllm+label%3Abug+created%3A%3E%3D$SINCE"
gh_search v2_sglang "repo%3Asgl-project%2Fsglang+correctness+OR+%22wrong+result%22+OR+NaN+created%3A%3E%3D$SINCE"
gh_search v3_trtllm "repo%3ANVIDIA%2FTensorRT-LLM+correctness+OR+%22wrong+output%22+OR+NaN+created%3A%3E%3D$SINCE"
gh_search v4_triton "repo%3Atriton-lang%2Ftriton+miscompile+OR+%22wrong+result%22+OR+numerical+OR+NaN+created%3A%3E%3D$SINCE"
gh_search v5_ttm "repo%3Atenstorrent%2Ftt-metal+numerical+OR+PCC+OR+correctness+created%3A%3E%3D$SINCE"
gh_search v6_vllm_kv "repo%3Avllm-project%2Fvllm+%22KV+cache%22+OR+quantization+OR+%22bit+flip%22+created%3A%3E%3D$SINCE"
gh_search v7_hf_tf "repo%3Ahuggingface%2Ftransformers+label%3Abug+created%3A%3E%3D$SINCE"
gh_search v8_sdc_global "%22silent+data+corruption%22+GPU+OR+LLM+created%3A%3E%3D$SINCE"
