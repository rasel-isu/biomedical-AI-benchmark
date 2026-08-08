eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate bio-nlp-next

# ── Parse args ──
# Usage:
#   ./run.sh --model gpt-4                                 # OpenAI (default)
#   ./run.sh --model llama3.1 --host http://127.0.0.1:11435 # local Ollama
# MODEL="gpt-4"
# HOST=""

MODEL="llama3.2:3b-instruct-fp16"
HOST="http://127.0.0.1:11435"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

HOST_ARGS=()
if [[ -n "$HOST" ]]; then
  echo "Using local Ollama model '$MODEL' at $HOST (no API key needed)"
  HOST_ARGS=(--host "$HOST")
else
  echo "Using OpenAI model '$MODEL'"
  export OPENAI_API_KEY=
fi

python -m agentic.run_agentic_eval \
  --model "$MODEL" \
  --setting zero_shot \
  --datasets ncbi_disease bc5cdr_chem chemprot ddi hoc litcovid pubmedqa medqa ms2 pubmed_summ cochrane plos \
  "${HOST_ARGS[@]}"

# # Smoke test 
# python -m agentic.run_agentic_eval \
#   --model "$MODEL" \
#   --setting zero_shot \
#   --datasets ncbi_disease bc5cdr_chem chemprot ddi hoc litcovid pubmedqa medqa ms2 pubmed_summ cochrane plos \
#   --max_instances 10 \
#   "${HOST_ARGS[@]}"

# Smoke test NER
# python -m agentic.run_agentic_eval \
#   --model "$MODEL" \
#   --setting zero_shot \
#   --datasets ncbi_disease bc5cdr_chem \
#   --max_instances 5 \
#   "${HOST_ARGS[@]}"

# # Smoke test RE
# python -m agentic.run_agentic_eval \
#   --model "$MODEL" \
#   --setting zero_shot \
#   --datasets chemprot ddi \
#   --max_instances 5 \
#   "${HOST_ARGS[@]}"

# # Smoke test on MLC
# python -m agentic.run_agentic_eval \
#   --model "$MODEL" \
#   --setting zero_shot \
#   --datasets hoc litcovid \
#   --max_instances 5 \
#   "${HOST_ARGS[@]}"

# # Smoke test on QA
# python -m agentic.run_agentic_eval \
#   --model "$MODEL" \
#   --setting zero_shot \
#   --datasets pubmedqa medqa \
#   --max_instances 5 \
#   "${HOST_ARGS[@]}"


# # Smoke test on Text  summarization
# python -m agentic.run_agentic_eval \
#   --model "$MODEL" \
#   --setting zero_shot \
#   --datasets ms2 pubmed_summ \
#   --max_instances 5 \
#   "${HOST_ARGS[@]}"

# # Smoke test on Text simplification
# python -m agentic.run_agentic_eval \
#   --model "$MODEL" \
#   --setting zero_shot \
#   --datasets cochrane plos \
#   --max_instances 5 \
#   "${HOST_ARGS[@]}"

