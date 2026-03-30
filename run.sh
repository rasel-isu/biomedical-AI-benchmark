eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate bio-nlp-next
# conda create --prefix /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/envs/bio-nlp-next python=3.13.12 -y
# salloc --nodes=1 --gres=gpu --partition=scavenger --mem=100G  --time=2-12:00:00

pip install torch transformers datasets accelerate peft
pip install langchain openai anthropic
pip install evaluate rouge-score bert-score
pip install pandas numpy scikit-learn jupyter

export OPENAI_API_KEY=

python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting zero_shot \
  --datasets medqa \
  --max_instances 5

# Smoke test NER
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting zero_shot \
  --datasets ncbi_disease bc5cdr_chem \
  --max_instances 5

# Smoke test RE
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting zero_shot \
  --datasets chemprot ddi \
  --max_instances 5

# Smoke test on QA
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting zero_shot \
  --datasets pubmedqa medqa \
  --max_instances 5


# Smoke test on Gen
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting zero_shot \
  --datasets cochrane pubmed_summ \
  --max_instances 5

# Smoke test on MLC
python -m agentic.run_agentic_eval \
  --model gpt-4 \
  --setting zero_shot \
  --datasets hoc litcovid \
  --max_instances 5
