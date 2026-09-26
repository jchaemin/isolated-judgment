"""The stored inputs under data/ (answers, prompts, cell flags, labels, hidden states, steering records) and the names of"""
import os

DATA = "data"

LAYOUT = dict(
    answers="answers/{model}/{task}.parquet",
    answer_hashes="answers/{model}/{task}_hashes.csv",
    wordings="prompts/wordings.csv",
    response_tokens="prompts/response_tokens.json",
    task_wordings="prompts/task_wordings.csv",
    task_tokens="prompts/task_tokens.json",
    task_items="prompts/{task}_items.json",
    cell_flags="cells/{model}/{task}.csv",
    triviaqa_labels="labels/{model}/triviaqa.csv",
    states="states/{model}/{task}/{condition}.npz",
    states_meta="states/{model}/{task}/{condition}.json",
    layers="states/{model}/{task}/layers/{condition}_{wording}.npz",
    depth="states/{model}/{task}/depth.npz",
    task_depth="states/{model}/{task}/depth/{condition}_{wording}.npz",
    format_first="states/{model}/{task}/format_first/{condition}_{wording}.npz",
    position="states/{model}/{task}/position/{wording}.npz",
    answers_of="states/{model}/{task}/answers_of/{source}.npz",
    final_layer="states/{model}/{task}/final_layer/{condition}_{wording}.npz",
    steering="steering/{model}/{task}/depth_{depth}.csv",
)


def path(kind, **fields):
    return os.path.join(DATA, LAYOUT[kind].format(**fields))


def key(array, condition=None, wording=None):
    return "_".join([array] + [x for x in (condition, wording) if x])
