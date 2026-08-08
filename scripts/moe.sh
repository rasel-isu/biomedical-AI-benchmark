MODEL="mixtral:8x7b-instruct-v0.1-q4_K_M"
HOST_ARGS=(--host http://127.0.0.1:11435)   # or HOST_ARGS=() for OpenAI/Azure

python -m moe.run_moe_eval \
  --model "$MODEL" \
  --setting zero_shot \
  --datasets ncbi_disease \
  --max_instances 5 \
  "${HOST_ARGS[@]}"