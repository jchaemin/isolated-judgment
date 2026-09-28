# Supplementary material

Code and result files for every table, figure and number in the paper.

```
readout/            shared library: models, datasets, conditions and wordings; prompts; the data/ layout; data loading; estimators
preprocessing/      extract_states.py: hidden states for every condition (GPU)
                    extract_variants.py: states for the entailment and safety, answer-source, dataset-candidate and instruction-family experiments (GPU)
experiments/        steering.py: steering interventions (GPU)
analysis/           readout.py (Section 4, App. B), subspace.py (Sections 5 and 7, App. C.1 to C.5),
                    generalization.py (App. C.5 to C.8), verdicts.py (Section 6, App. D)
results/            the result files, one folder per analysis file
data/prompts/       instruction wordings and response tokens
```

## Running

```
pip install -r requirements.txt
python preprocessing/extract_states.py negated --model Qwen/Qwen2.5-7B-Instruct --task gsm8k   # one family of states (GPU)
python analysis/subspace.py heldout      
python analysis/subspace.py all           # every analysis in the file
```


Most analyses read states written by `preprocessing/extract_states.py`. These read the output of another GPU step:

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

AUROC is computed against the benchmark label unless a column says otherwise. The LEACE and INLP rows of `subspace/erasure.csv` depend on an eigenvalue cutoff and an iterative solver, so they vary with the numerical environment (thread count, library versions). A rerun can differ by up to 0.03 in a single INLP value and by up to 0.004 in Table 13.
