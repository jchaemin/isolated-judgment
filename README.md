# Supplementary material

Code and result files for every table, figure and number in the paper. Each result file holds only the rows and columns the paper uses.

```
readout/            shared library: models, datasets, conditions and wordings; prompts; the data/ layout; data loading; estimators
preprocessing/      extract_states.py: hidden states for every condition (GPU)
                    extract_variants.py: states for the entailment and safety, answer-source, dataset-candidate and instruction-family experiments (GPU)
experiments/        steering.py: steering interventions (GPU)
analysis/           readout.py (Section 4, App. B), subspace.py (Sections 5 and 7, App. C.1 to C.5),
                    generalization.py (App. C.5 to C.8), verdicts.py (Section 6, App. D)
results/            the result files, one folder per analysis file
```

## Running

```
pip install -r requirements.txt
python preprocessing/extract_states.py negated --model Qwen/Qwen2.5-7B-Instruct --task gsm8k   # one family of states (GPU)
python analysis/subspace.py heldout       # one analysis
python analysis/subspace.py all           # every analysis in the file
```

Scripts run from this directory and read and write their inputs under `data/`, laid out as in `readout/storage.py`. The hidden states and model answers are too large to include. `python <file> --help` lists the commands of each file. Each analysis fixes its own wordings and rows, as described in Appendix A.

## Where each number comes from

| Paper | Result files | Command |
|---|---|---|
| Abstract, text | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| Figure 1 | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| §1, text | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| §3, text | `readout/regularization_path.csv` | `analysis/readout.py regularization` |
| Table 2 | `readout/position.csv`, `readout/steering.csv` | `analysis/readout.py position`, `analysis/readout.py steering` |
| §4, text | `generalization/dataset_candidates.csv`, `generalization/other_model_answers.csv`, `readout/format_first.csv`, `readout/position.csv`, `readout/steering.csv` | `analysis/generalization.py dataset-candidates`, `analysis/generalization.py other-model`, `analysis/readout.py format-first`, `analysis/readout.py position`, `analysis/readout.py steering` |
| §5, text | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| §5.1, text | `subspace/ranks.csv` | `analysis/subspace.py heldout` |
| Figure 2 | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| Table 3 | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| §5.2, text | `generalization/models/`, `generalization/tasks.csv`, `subspace/heldout.csv` | `analysis/generalization.py models`, `analysis/generalization.py tasks`, `analysis/subspace.py heldout` |
| §5.3, text | `generalization/instruction_family.csv`, `subspace/controls.csv`, `subspace/erasure.csv`, `subspace/heldout.csv`, `subspace/heldout_wordings.csv`, `subspace/principal_components.csv`, `subspace/verdict_free.csv` | `analysis/generalization.py instruction-family`, `analysis/subspace.py components`, `analysis/subspace.py controls`, `analysis/subspace.py erasure`, `analysis/subspace.py heldout`, `analysis/subspace.py verdict-free`, `analysis/subspace.py wordings` |
| Figure 3 | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| §6, text | `subspace/heldout.csv`, `verdicts/correct_token_rate.csv` | `analysis/subspace.py heldout`, `analysis/verdicts.py negation` |
| §6.1, text | `verdicts/negated_verdicts.csv`, `verdicts/self_labeled.csv` | `analysis/verdicts.py negation`, `analysis/verdicts.py self-labeled` |
| Figure 4 | `subspace/depth.csv` | `analysis/subspace.py depth` |
| §7.1, text | `readout/condition_identity.csv`, `subspace/depth.csv` | `analysis/readout.py format-first`, `analysis/subspace.py depth` |
| §7.2, text | `subspace/directions_removed.csv` | `analysis/subspace.py directions` |
| App. A, text | `subspace/erasure.csv`, `subspace/heldout.csv` | `analysis/subspace.py erasure`, `analysis/subspace.py heldout` |
| Table 4 | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| App. B.1, text | `readout/regularization_path.csv` | `analysis/readout.py regularization` |
| Table 5 | `readout/format_first.csv` | `analysis/readout.py format-first` |
| App. B.2, text | `readout/format_first.csv`, `readout/position.csv`, `readout/position_frozen.csv`, `readout/position_interval.json` | `analysis/readout.py format-first`, `analysis/readout.py position` |
| App. B.3, text | `readout/steering.csv`, `readout/steering_cells.csv`, `readout/steering_compliant.csv` | `analysis/readout.py steering` |
| Table 6 | `subspace/heldout.csv`, `subspace/ranks.csv`, `subspace/readout_share.csv` | `analysis/subspace.py heldout`, `analysis/subspace.py share` |
| App. C.1, text | `readout/condition_identity.csv`, `subspace/pair_sources.csv`, `subspace/ranks.csv` | `analysis/readout.py format-first`, `analysis/subspace.py heldout`, `analysis/subspace.py pair-sources` |
| Table 7 | `subspace/heldout.csv`, `subspace/heldout_intervals.csv` | `analysis/subspace.py heldout`, `analysis/subspace.py intervals` |
| Table 8 | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| Table 9 | `generalization/models/` | `analysis/generalization.py models` |
| App. C.2, text | `subspace/heldout_per_wording.csv` | `analysis/subspace.py per-wording` |
| Table 10 | `subspace/depth.csv` | `analysis/subspace.py depth` |
| Table 11 | `subspace/pair_sources.csv` | `analysis/subspace.py pair-sources` |
| Table 12 | `subspace/directions_removed.csv` | `analysis/subspace.py directions` |
| Table 13 | `subspace/erasure.csv` | `analysis/subspace.py erasure` |
| Table 14 | `subspace/verdict_free.csv` | `analysis/subspace.py verdict-free` |
| Table 15 | `subspace/heldout_wordings.csv` | `analysis/subspace.py wordings` |
| Table 16 | `generalization/instruction_family.csv` | `analysis/generalization.py instruction-family` |
| App. C.5, text | `subspace/controls.csv`, `subspace/principal_components.csv`, `subspace/readout_share.csv` | `analysis/subspace.py components`, `analysis/subspace.py controls`, `analysis/subspace.py share` |
| Table 17 | `generalization/tasks.csv` | `analysis/generalization.py tasks` |
| App. C.6, text | `generalization/tasks_per_model.json` | `analysis/generalization.py tasks` |
| Table 18 | `generalization/models/` | `analysis/generalization.py models` |
| Table 19 | `generalization/models/` | `analysis/generalization.py models` |
| Table 20 | `generalization/other_model_answers.csv` | `analysis/generalization.py other-model` |
| Table 21 | `generalization/dataset_candidates.csv` | `analysis/generalization.py dataset-candidates` |
| Table 22 | `subspace/heldout.csv` | `analysis/subspace.py heldout` |
| App. D.2, text | `subspace/heldout.csv`, `verdicts/drop_vs_rate.csv` | `analysis/subspace.py heldout`, `analysis/verdicts.py negation` |
| Table 23 | `verdicts/self_labeled.csv` | `analysis/verdicts.py self-labeled` |
| Table 24 | `verdicts/self_labeled.csv` | `analysis/verdicts.py self-labeled` |
| Table 25 | `verdicts/negated_verdicts.csv` | `analysis/verdicts.py negation` |

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
| `wording` | `P01` to `P10`, the ten instruction wordings (`readout/config.py`) |

AUROC is computed against the benchmark label unless a column says otherwise.
