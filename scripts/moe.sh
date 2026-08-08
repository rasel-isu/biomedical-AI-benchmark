eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate bio-nlp-next



python -m moe.run_moe_eval \
  --model "$MODEL" \
  --setting zero_shot \
  --datasets ncbi_disease \
  --max_instances 5 \
  "${HOST_ARGS[@]}"