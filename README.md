# Isolating a Language Model’s Judgment with a Label-Free Subspace

Code and result files for every table, figure and number in the paper.

```
readout/            shared library: models, datasets, conditions and wordings; prompts; the data/ layout; data loading; estimators
preprocessing/      extract_states.py: hidden states for every condition (GPU)
                    extract_variants.py: states for the entailment and safety, answer-source, dataset-candidate and instruction-family experiments (GPU)
                    generate_answers.py, write_position_states.py: candidate answers and position states (GPU)
                    answer_tasks.py, grade_answers.py, regrade_triviaqa.py, select_task_items.py, cell_flags.py: grading, labels, task items, cell flags
experiments/        steering.py: steering interventions (GPU)
analysis/           readout.py (Section 4, App. B), subspace.py (Sections 5 and 7, App. C.1 to C.5),
                    generalization.py (App. C.5 to C.8), verdicts.py (Section 6, App. D)
                    tables.py, figure_2a.py: the printed numbers and Figure 2(a) (table labels in the output follow an earlier draft)
results/            the result files, one folder per analysis file
data/               prompts, candidate answers, TriviaQA labels, cell flags
reply_experiments/  experiments added for the revision
```

## Running

```
pip install -r requirements.txt
python preprocessing/extract_states.py negated --model Qwen/Qwen2.5-7B-Instruct --task gsm8k   # one family of states (GPU)
python analysis/subspace.py heldout
python analysis/subspace.py all           # every analysis in the file
bash reproduce.sh                         # every analysis and every table
```

Most analyses read states written by `preprocessing/extract_states.py` (listed in `hidden_states_manifest.tsv`). These read the output of another GPU step:

| Analysis | Reads the output of |
|---|---|
| `analysis/generalization.py tasks` | `preprocessing/extract_variants.py tasks` |
| `analysis/generalization.py other-model` | `preprocessing/extract_variants.py other-model` |
| `analysis/generalization.py dataset-candidates` | `preprocessing/extract_variants.py dataset-candidates` |
| `analysis/generalization.py instruction-family` | `preprocessing/extract_variants.py instruction-family` |
| `analysis/readout.py steering` | `experiments/steering.py` |

## Names in the result files

| Column | Values |
|---|---|
| `cell` | model and dataset, as in the paper's per-cell tables (`Qwen2.5-7B GSM8K`) |
| `condition` | `yes_no`, `true_false`, `a_b`, `b_a`, `sentence`, `one_zero`, `correct_incorrect` (`word_pair` for entailment and safety), `negated`, `no_means_correct`, `false_means_correct`, `zero_means_correct` (`..._positive` for entailment and safety) |
| `method` | `readout` (the Yes/No class mean), `isolated` (the isolated judgment), `fitted` (a class mean fitted on the condition), `pooled` (fitted on A/B and B/A), `tuned_probe`, `verdict` (the model's own response), and the variants named in each table |
| `wording` | `P01` to `P10`, the ten instruction wordings (`data/prompts/wordings.csv`) |

AUROC is computed against the benchmark label unless a column says otherwise. A TriviaQA answer is correct when a normalized alias is contained in the normalized first line of the answer; for Gemma-2-9B-IT and the larger and additional models, the first line must equal a normalized alias. With one thread per process (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`), a rerun reproduces every file in `results/` byte-identically.
