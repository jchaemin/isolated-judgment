#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; R=run_package_script.py; M=Qwen/Qwen3.5-9B
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
D=${DATA_9B:-$PWD/data}
for t in arc gsm8k triviaqa mmlu; do python $R generate_answers.py --model $M --task $t --out $D/answers; done
for t in arc gsm8k triviaqa mmlu; do
  for c in formats negated sentence; do
    python $R extract_states.py $c --model $M --task $t
  done
done
for t in entailment safety; do for c in cross-task cross-task-negated; do python $R extract_states.py $c --model $M --task $t; done; done
