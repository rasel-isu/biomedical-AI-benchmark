eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate bio-nlp-next
HOST="http://127.0.0.1:11435"
HOST_ARGS=(--host "$HOST")

# python -m moe.run_moe_eval \
#   --model "$MODEL" \
#   --setting zero_shot \
#   --datasets ncbi_disease \
#   --max_instances 5 \
#   "${HOST_ARGS[@]}"
 

 python -m moe.run_moe_eval \
  --model "mixtral:8x7b-instruct-v0.1-q4_K_M" \
  --setting zero_shot \
  --datasets ncbi_disease \
  --max_instances 100 \
  --host "http://127.0.0.1:11435"