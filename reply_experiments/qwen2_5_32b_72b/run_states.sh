#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; R=run_package_script.py
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
D=${DATA_32B_72B:-$PWD/data}
for T in Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct; do
  for t in gsm8k mmlu arc triviaqa; do python $R $T generate_answers.py --model Qwen/$T --task $t --batch_size 16 --out $D/answers; done
  python $R $T grade_answers.py --answers $D/answers --models $T --tasks gsm8k mmlu arc triviaqa --out $D/grade_check_$T.csv
  python $R $T regrade_triviaqa.py --answers $D/answers --out $D/labels
  python $R $T cell_flags.py --data $D --out $D/cells
  for t in gsm8k mmlu arc triviaqa; do
    for c in formats negated sentence one-zero-correct-incorrect zero-false-means-correct no-means-correct more-wordings; do
      python $R $T extract_states.py $c --model Qwen/$T --task $t
    done
  done
  for t in entailment safety; do for c in cross-task cross-task-negated; do python $R $T extract_states.py $c --model Qwen/$T --task $t; done; done
done
