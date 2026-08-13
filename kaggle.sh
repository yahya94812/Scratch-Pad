#!/bin/bash

# all the setup stuff
export NANOCHAT_BASE_DIR="/kaggle/working/cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then # It checks whether the environment variable WANDB_RUN is empty or unset, and if so, assigns it the value "dummy".
    WANDB_RUN=dummy
fi

# train a tiny tokenizer on ~1M characters (a few seconds)
python -m nanochat.dataset -n 3
python -m scripts.tok_train --max-chars=1_900_000_000 --vocab-size=8000
python -m scripts.tok_eval

# train the model for T4 X 2
torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --depth=8 \
    --head-dim=64 \
    --window-pattern=S \
    --max-seq-len=256 \
    --device-batch-size=128 \
    --total-batch-size=65536 \
    --eval-every=2 \
    --eval-tokens=8192 \
    --core-metric-every=4 \
    --sample-every=5 \
    --num-iterations=10 \
    --run="$WANDB_RUN" \
    --save-every=5

# evaluating the model
python -m scripts.base_eval --device-batch-size=128 --split-tokens=256 --max-per-task=500

echo "Done!."
