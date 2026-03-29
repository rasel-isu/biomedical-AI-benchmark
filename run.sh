eval $(/lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/bin/conda shell.bash hook)
source /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/etc/profile.d/conda.sh
conda activate bio-nlp-next
# conda create --prefix /lustre/hdd/LAS/qli-lab/rasel/apps/miniconda3/envs/bio-nlp-next python=3.13.12 -y
# salloc --nodes=1 --gres=gpu --partition=scavenger --mem=100G  --time=2-12:00:00

pip install torch transformers datasets accelerate peft
pip install langchain openai anthropic
pip install evaluate rouge-score bert-score
pip install pandas numpy scikit-learn jupyter