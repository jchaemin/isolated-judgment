"""The printed numbers, recomputed from the result files. Usage: python analysis/tables.py --root results [--data data] [--out results/tables]"""
import argparse, json, os, re
import numpy as np, pandas as pd

ROOT = RES = DATA = OUT = None
ROWS = []
FOLLOW = None
CT5 = None
LABELS = {"fig2a": ["No means correct", "False means correct", "0 means correct", "Negated question"], "7": ["No means correct", "False means correct", "0 means correct", "negated question"], "8": ["Yes/No", "True/False", "A/B", "sentence", "1/0", "correct/incorrect", "B/A", "negated question", "0 means correct", "False means correct", "No means correct"], "9": ["Yes/No", "True/False", "A/B", "sentence", "1/0", "correct/incorrect", "B/A", "negated question", "0 means correct", "False means correct", "No means correct"], "3": ["Llama-3.1-8B ARC", "Llama-3.1-8B GSM8K", "Llama-3.1-8B MMLU", "Llama-3.1-8B TriviaQA", "Mistral-7B ARC", "Mistral-7B GSM8K", "Mistral-7B MMLU", "Mistral-7B TriviaQA", "Qwen2.5-14B GSM8K", "Qwen2.5-14B MMLU", "Qwen2.5-14B TriviaQA", "Qwen2.5-7B GSM8K", "Qwen2.5-7B MMLU", "Qwen2.5-7B TriviaQA"], "29": ["Llama-3.1-8B ARC", "Llama-3.1-8B GSM8K", "Llama-3.1-8B MMLU", "Llama-3.1-8B TriviaQA", "Mistral-7B ARC", "Mistral-7B GSM8K", "Mistral-7B MMLU", "Mistral-7B TriviaQA", "Qwen2.5-14B GSM8K", "Qwen2.5-14B MMLU", "Qwen2.5-14B TriviaQA", "Qwen2.5-7B GSM8K", "Qwen2.5-7B MMLU", "Qwen2.5-7B TriviaQA"], "5": ["Llama-3.1-8B ARC", "Llama-3.1-8B GSM8K", "Llama-3.1-8B MMLU", "Llama-3.1-8B TriviaQA", "Mistral-7B ARC", "Mistral-7B GSM8K", "Mistral-7B MMLU", "Mistral-7B TriviaQA", "Qwen2.5-14B GSM8K", "Qwen2.5-14B MMLU", "Qwen2.5-14B TriviaQA", "Qwen2.5-7B GSM8K", "Qwen2.5-7B MMLU", "Qwen2.5-7B TriviaQA"], "12": ["depth 0.25", "depth 0.50", "depth 0.75", "depth 0.90", "depth 1.00"], "13": ["No means correct", "False means correct", "0 means correct", "negated question"], "14": ["Yes/No", "True/False", "A/B", "sentence", "1/0", "correct/incorrect", "B/A", "negated question", "0 means correct", "False means correct", "No means correct"], "15": ["No means correct", "False means correct", "0 means correct", "negated question"], "16": ["No means correct", "False means correct", "0 means correct", "negated question"], "17": ["P01,P03,P05 \u2192 P07,P09 No means correct", "P01,P03,P05 \u2192 P07,P09 False means correct", "P01,P03,P05 \u2192 P07,P09 0 means correct", "P01,P03,P05 \u2192 P07,P09 negated question", "P07,P09 \u2192 P01,P03,P05 No means correct", "P07,P09 \u2192 P01,P03,P05 False means correct", "P07,P09 \u2192 P01,P03,P05 0 means correct", "P07,P09 \u2192 P01,P03,P05 negated question"], "18": ["Yes/No", "True/False", "A/B", "B/A", "sentence", "negated question", "No means correct"], "20": ["Yes/No", "True/False", "A/B", "B/A", "sentence", "negated question", "1/0", "word pair", "0 means positive", "False means positive", "No means positive"], "22": ["Qwen2.5-32B-Instruct", "Gemma-2-27B-IT"], "23": ["Mistral-Nemo-12B-Instruct", "Ministral-8B-Instruct", "OLMo-2-7B-Instruct"], "26": ["Qwen2.5-7B reads Llama-3.1-8B, TriviaQA", "Qwen2.5-7B reads Llama-3.1-8B, MMLU", "Llama-3.1-8B reads Mistral-7B, TriviaQA", "Llama-3.1-8B reads Mistral-7B, MMLU", "Mistral-7B reads Qwen2.5-14B, TriviaQA", "Mistral-7B reads Qwen2.5-14B, MMLU descr.", "Qwen2.5-14B reads Qwen2.5-7B, TriviaQA", "Qwen2.5-14B reads Qwen2.5-7B, MMLU"], "27": ["Yes/No", "True/False", "A/B", "B/A", "negated question", "No means correct", "sentence"], "30": ["Yes/No", "True/False", "A/B", "sentence", "1/0", "correct/incorrect", "B/A", "0 means correct", "False means correct", "No means correct"], "31": ["Llama-3.1-8B ARC", "Llama-3.1-8B GSM8K", "Llama-3.1-8B MMLU", "Llama-3.1-8B TriviaQA", "Mistral-7B GSM8K", "Mistral-7B MMLU", "Mistral-7B TriviaQA", "Qwen2.5-14B GSM8K", "Qwen2.5-14B MMLU", "Qwen2.5-14B TriviaQA", "Qwen2.5-7B GSM8K", "Qwen2.5-7B MMLU", "Qwen2.5-7B TriviaQA"], "4": ["Llama-3.1-8B ARC", "Llama-3.1-8B GSM8K", "Llama-3.1-8B MMLU", "Llama-3.1-8B TriviaQA", "Mistral-7B ARC", "Mistral-7B GSM8K", "Mistral-7B MMLU", "Mistral-7B TriviaQA", "Qwen2.5-14B ARC", "Qwen2.5-14B GSM8K", "Qwen2.5-14B MMLU", "Qwen2.5-14B TriviaQA", "Qwen2.5-7B ARC", "Qwen2.5-7B GSM8K", "Qwen2.5-7B MMLU", "Qwen2.5-7B TriviaQA"]}
CM = {'Yes/No':'yes_no','True/False':'true_false','A/B':'a_b','B/A':'b_a','sentence':'sentence','1/0':'one_zero','correct/incorrect':'correct_incorrect',
      'negated question':'negated','Negated question':'negated','0 means correct':'zero_means_correct','False means correct':'false_means_correct','No means correct':'no_means_correct',
      'word pair':'word_pair','0 means positive':'zero_means_positive','False means positive':'false_means_positive','No means positive':'no_means_positive'}
HO4 = ('negated', 'zero_means_correct', 'false_means_correct', 'no_means_correct')

def rd(f, **kw): return pd.read_csv(os.path.join(ROOT, f), **kw)

def add(loc, ctx, src, key, comp):
    ok = comp is not None and not (isinstance(comp, float) and np.isnan(comp))
    ROWS.append(dict(location=loc, context=ctx, value=(float(comp) if ok else np.nan), source_file=src, source_key=key))


MODELS4 = ['Qwen2.5-7B-Instruct', 'Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-14B-Instruct']
TASKS4 = ['gsm8k', 'mmlu', 'arc', 'triviaqa']
SHORT = {'Qwen2.5-7B-Instruct': 'Qwen2.5-7B', 'Llama-3.1-8B-Instruct': 'Llama-3.1-8B', 'Mistral-7B-Instruct-v0.3': 'Mistral-7B', 'Qwen2.5-14B-Instruct': 'Qwen2.5-14B'}
TNAME = {'gsm8k': 'GSM8K', 'mmlu': 'MMLU', 'arc': 'ARC', 'triviaqa': 'TriviaQA'}
H = C14 = None

def hv(df, method, cond, col='auroc', cells=None, agg='mean', cellcol='cell'):
    g = df[(df.method == method) & (df.condition == cond)]
    if cells is not None: g = g[g[cellcol].isin(cells)]
    return getattr(g[col], agg)() if len(g) else np.nan
def cond_of(label): return CM[label]
def labels(t): return LABELS[str(t)]
def method_key(m, col='auroc'): return 'method=readout; col=verdict_auroc' if m is None else f'method={m}; col={col}'

def cell_table():
    out = []
    for m in MODELS4:
        for t in TASKS4:
            a = pd.read_parquet(f'{DATA}/answers/{m}/{t}.parquet'); g = a[a.condition == 'greedy'].sort_values('qid')
            y = g.correct.values.astype(int)
            if t == 'triviaqa':
                lab = pd.read_csv(f'{DATA}/labels/{m}/triviaqa.csv'); assert (lab.original.values == y).all(); y = lab.regraded.values.astype(int)
            fl = pd.read_csv(f'{DATA}/cells/{m}/{t}.csv').primary
            assert fl.nunique() == 1
            neg = int((y == 0).sum())
            assert (neg >= 100) == bool(fl.iloc[0]), f'{m} {t}: primary flag does not follow the 100-negative rule'
            out.append(dict(cell=f'{SHORT[m]} {TNAME[t]}', rows=len(y), negatives=neg, positives=len(y) - neg, primary=bool(fl.iloc[0])))
    return pd.DataFrame(out)

def compliance_table():
    rows = []
    for m in MODELS4:
        for t in TASKS4:
            r = dict(cell=f'{SHORT[m]} {TNAME[t]}')
            for f, sl in (('yes_no', slice(0, 2)), ('a_b', slice(4, 6)), ('b_a', slice(4, 6))):
                v = []
                for p in ('P01', 'P03', 'P05', 'P07', 'P09'):
                    z = np.load(f'{DATA}/states/{m}/{t}/format_first/{f}_{p}.npz', allow_pickle=True); n = min(400, len(z['y']))
                    v.append(float(np.isin(z['t20i'][:n, 0], z['token_ids'][sl]).mean()))
                r[f'top1_in_response_tokens_{f}'] = float(np.mean(v))
            r['letters_ok'] = bool(min(r['top1_in_response_tokens_a_b'], r['top1_in_response_tokens_b_a']) >= 0.5)
            r['model'] = SHORT[m]; rows.append(r)
    d = pd.DataFrame(rows)
    d['model_follows'] = d.groupby('model').letters_ok.transform('all')
    return d

def setup(root, data):
    global ROOT, RES, DATA, H, C14, FOLLOW, CT5
    ROOT = RES = root; DATA = data
    H = rd('subspace/heldout.csv'); C14 = list(H.cell.unique())
    FOLLOW = CT5 = None
    if data and os.path.isdir(os.path.join(data, 'states')):
        CT5 = compliance_table(); FOLLOW = set(CT5[CT5.model_follows].cell)
    elif os.path.exists(os.path.join(OUT, 'table_4_choice.csv')):
        c = pd.read_csv(os.path.join(OUT, 'table_4_choice.csv')); FOLLOW = set(c[c.model_follows].cell)

def t_heldout():
    for lab in labels('fig2a'):
        c = cond_of(lab)
        for name, m, col in (('verdict','readout','verdict_auroc'), ('readout','readout','auroc'), ('isolated','isolated','auroc'), ('condition-fitted','fitted','auroc')):
            add(f'Figure 2(a) / {lab} / {name}', f'held-out {lab}', 'subspace/heldout.csv', f'condition={c}; method={m}; col={col}; mean over 14 cells', hv(H,m,c,col))
    for lab in labels(7):
        c = cond_of(lab)
        for name, m in zip(['verdict','readout','pooled','isolated','condition-fitted'], [None,'readout','pooled','isolated','fitted']):
            x = hv(H,'readout',c,'verdict_auroc') if m is None else hv(H,m,c)
            add(f'Table 7 / {lab} / {name}', f'held-out {lab}', 'subspace/heldout.csv', f'condition={c}; {method_key(m)}; mean over 14 cells', x)
    for lab in labels(8):
        c = cond_of(lab)
        for name, m in zip(['verdict','readout','pooled','tuned probe','isolated','condition-fitted'], [None,'readout','pooled','tuned_probe','isolated','fitted']):
            x = (np.nan if c == 'sentence' else hv(H,'readout',c,'verdict_auroc')) if m is None else hv(H,m,c)
            add(f'Table 8 / {lab} / {name}', f'every readout, target {lab}', 'subspace/heldout.csv', f'condition={c}; {method_key(m)}; mean over 14 cells', x)
    G = rd('generalization/models/gemma-2-9b.csv')
    for lab in labels(9):
        c = cond_of(lab)
        for name, m in zip(['verdict','readout','pooled','tuned probe','isolated','condition-fitted'], [None,'readout','pooled','tuned_probe','isolated','fitted']):
            x = (np.nan if c == 'sentence' else hv(G,'readout',c,'verdict_auroc')) if m is None else hv(G,m,c)
            add(f'Table 9 / {lab} / {name}', 'Gemma-2-9B-IT', 'generalization/models/gemma-2-9b.csv', f'condition={c}; {method_key(m)}; mean over 4 cells', x)
    for lab in labels(3):
        keys = [('readout','yes_no'),('readout','a_b'),('fitted','a_b'),('readout','b_a'),('fitted','b_a'),('readout','sentence'),('fitted','sentence')]
        names = ['Yes/No own','frozen->A/B','A/B fitted','frozen->B/A','B/A fitted','frozen->sentence','sentence fitted']
        for nm, (m,c) in zip(names, keys):
            x = H[(H.cell==lab)&(H.method==m)&(H.condition==c)].auroc
            add(f'Table 3 / {lab} / {nm}', 'per-cell format transfer', 'subspace/heldout.csv', f'cell={lab}; method={m}; condition={c}', x.iloc[0] if len(x) else np.nan)
    for lab in labels(29):
        specs = [('verdict, correct?','readout','yes_no','verdict_auroc'),('verdict, incorrect?','readout','negated','verdict_auroc'),('readout, correct?','readout','yes_no','auroc'),
                 ('readout, incorrect?','readout','negated','auroc'),('isolated, incorrect?','isolated','negated','auroc'),('fitted, incorrect?','fitted','negated','auroc')]
        for nm, m, c, col in specs:
            x = H[(H.cell==lab)&(H.method==m)&(H.condition==c)][col]
            add(f'Table 29 / {lab} / {nm}', 'negated question per cell', 'subspace/heldout.csv', f'cell={lab}; method={m}; condition={c}; col={col}', x.iloc[0])
    R = rd('subspace/ranks.csv'); S = rd('subspace/readout_share.csv')
    for lab in labels(5):
        rk = R[R.cell==lab]['rank']; sh = S[S.cell==lab].share
        add(f'Table 5 / {lab} / rank', 'split-half rank, median over folds x wordings', 'subspace/ranks.csv', f'cell={lab}; median(rank) over 25 fold-wording rows', rk.median())
        add(f'Table 5 / {lab} / norm in subspace', 'share of readout sq. norm in subspace (median)', 'subspace/readout_share.csv', f'cell={lab}; median(share) over fold x wording', sh.median())
        for nm, (m,c) in zip(['readout, B/A','isolated, B/A','readout, negated','isolated, negated'], [('readout','b_a'),('isolated','b_a'),('readout','negated'),('isolated','negated')]):
            x = H[(H.cell==lab)&(H.method==m)&(H.condition==c)].auroc
            add(f'Table 5 / {lab} / {nm}', 'per cell', 'subspace/heldout.csv', f'cell={lab}; method={m}; condition={c}', x.iloc[0])
    D = rd('subspace/depth.csv')
    cols = [('yes_no','readout'),('yes_no','isolated'),('b_a','readout'),('b_a','isolated'),('no_means_correct','readout'),('no_means_correct','isolated'),('negated','readout'),('negated','isolated')]
    for lab in labels(12):
        dp = float(lab.replace('depth ',''))
        for c, m in cols:
            x = D[(D.depth==dp)&(D.condition==c)&(D.method==m)].auroc
            add(f'Table 12 / {lab} / {c} {m}', 'readout by depth', 'subspace/depth.csv', f'depth={dp}; condition={c}; method={m}; mean over {len(x)} cells', x.mean())
    PS = rd('subspace/pair_sources.csv'); PS = PS[PS.quantity=='auroc']
    for lab in labels(13):
        c = cond_of(lab)
        for nm, m in zip(['readout','five formats only','negation pairs only','both','condition-fitted'], ['readout','isolated_format_pairs','isolated_negation_pairs','isolated','fitted']):
            x = PS[(PS.condition==c)&(PS.method==m)].value
            add(f'Table 13 / {lab} / {nm}', 'which pairs are needed', 'subspace/pair_sources.csv', f'condition={c}; method={m}; mean over 14 cells', x.mean())
    DR = rd('subspace/directions_removed.csv')
    for lab in labels(14):
        c = cond_of(lab)
        for nm, m in zip(['readout','rank 1','rank 3','full subspace (selected rank)','condition-fitted'], ['readout','negation_rank_1','negation_rank_3','subspace_full','fitted']):
            x = DR[(DR.condition==c)&(DR.method==m)].auroc
            add(f'Table 14 / {lab} / {nm}', 'directions removed (median)', 'subspace/directions_removed.csv', f'condition={c}; method={m}; median over 14 cells', x.median())
    E = rd('subspace/erasure.csv')
    for lab in labels(15):
        c = cond_of(lab)
        for nm, m in zip(['readout','after LEACE','after INLP','isolated'], ['readout','leace','inlp','isolated']):
            x = E[(E.condition==c)&(E.method==m)].auroc
            add(f'Table 15 / {lab} / {nm}', 'concept erasure (median)', 'subspace/erasure.csv', f'condition={c}; method={m}; median over 14 cells', x.median())

def t_part2():
    VF = rd('subspace/verdict_free.csv')
    for lab in labels(16):
        c = cond_of(lab)
        for nm, sb in zip(['readout','identical-verdict pairs','verdict sign regressed out','as in Table 7'], ['none','identical_verdict_pairs','verdict_regressed_out','all_pairs']):
            x = VF[(VF.condition==c)&(VF.subspace==sb)].auroc
            add(f'Table 16 / {lab} / {nm}', 'verdict-free subspace (median)', 'subspace/verdict_free.csv', f'condition={c}; subspace={sb}; median over 14 cells', x.median())
    W = rd('subspace/heldout_wordings.csv')
    for lab in labels(17):
        a, b, cn = re.match(r'(P[\d,P]+) → ([P\d,]+) (.*)', lab).groups(); split = f'{a} -> {b}'; c = cond_of(cn)
        g = W[(W.split==split)&(W.condition==c)]
        for nm, col in zip(['readout','other wordings','same wording','fitted'], ['readout','isolated_other_wordings','isolated_same_wording','fitted']):
            add(f'Table 17 / {split} / {cn} / {nm}', 'held-out wordings (median)', 'subspace/heldout_wordings.csv', f'split={split}; condition={c}; col={col}; median over 14 cells', g[col].median())
    IF = rd('generalization/instruction_family.csv')
    for lab in labels(18):
        c = cond_of(lab)
        for nm, m in zip(['readout','isolated','fitted on new family'], ['readout','isolated','fitted_new_family']):
            x = IF[(IF.condition==c)&(IF.method==m)].auroc
            add(f'Table 18 / {lab} / {nm} mean', 'new instruction family', 'generalization/instruction_family.csv', f'condition={c}; method={m}; mean over {len(x)} cells', x.mean())
            add(f'Table 18 / {lab} / {nm} median', 'new instruction family', 'generalization/instruction_family.csv', f'condition={c}; method={m}; median over {len(x)} cells', x.median())
    T = rd('generalization/tasks.csv')
    for lab in labels(20):
        c = cond_of(lab)
        for tk in ('entailment', 'safety'):
            g = T[(T.task==tk)&(T.condition==c)]
            for col in ['readout','isolated','fitted','verdict']:
                x = g[col].iloc[0] if len(g) else np.nan
                add(f'Table 20 / {tk} / {lab} / {col}', f'{tk} judgment task', 'generalization/tasks.csv', f'task={tk}; condition={c}; col={col}', x)
    mods = {'Qwen2.5-32B-Instruct':'qwen2.5-32b','Gemma-2-27B-IT':'gemma-2-27b','Mistral-Nemo-12B-Instruct':'mistral-nemo-12b','Ministral-8B-Instruct':'ministral-8b','OLMo-2-7B-Instruct':'olmo-2-7b'}
    def mcols(stem):
        M = rd(f'generalization/models/{stem}.csv'); out = {}
        for tk in ['TriviaQA','MMLU']:
            g = M[M.cell.str.endswith(tk)]
            f = lambda m,c: g[(g.method==m)&(g.condition==c)].auroc.iloc[0]
            yn = f('readout','yes_no')
            fvf = max(abs(f('readout',c)-f('fitted',c)) for c in ['a_b','b_a','true_false'])
            fall = yn - f('readout','negated')
            ivf = max(abs(f('isolated',c)-f('fitted',c)) for c in ['negated','no_means_correct'])
            out[tk] = (yn, fvf, fall, ivf)
        return out
    k18 = {'Yes/No readout': 'method=readout; condition=yes_no; mean over the TriviaQA and MMLU cells',
           'frozen vs fitted': 'max over a_b, b_a, true_false of |mean readout - mean fitted| (means over the TriviaQA and MMLU cells)',
           'fall under negation': 'mean over the TriviaQA and MMLU cells of (readout yes_no - readout negated)',
           'isolated vs fitted': 'max over negated, no_means_correct of |mean isolated - mean fitted| (means over the TriviaQA and MMLU cells)'}
    for lab in labels(22):
        stem = mods[lab]; M = rd(f'generalization/models/{stem}.csv'); out = mcols(stem)
        avg = np.mean([out['TriviaQA'], out['MMLU']], axis=0)
        f = lambda m,c: M[(M.method==m)&(M.condition==c)].auroc.mean()
        fvf = max(abs(f('readout',c)-f('fitted',c)) for c in ['a_b','b_a','true_false']); ivf = max(abs(f('isolated',c)-f('fitted',c)) for c in ['negated','no_means_correct'])
        for nm, x in zip(['Yes/No readout','frozen vs fitted','fall under negation','isolated vs fitted'], [avg[0], fvf, avg[2], ivf]):
            add(f'Table 22 / {lab} / {nm}', 'larger models', f'generalization/models/{stem}.csv', k18[nm], x)
    k19 = {'Yes/No': 'method=readout; condition=yes_no', 'frozen vs fitted': 'max over a_b, b_a, true_false of |readout - fitted|',
           'fall': 'readout yes_no - readout negated', 'isolated vs fitted': 'max over negated, no_means_correct of |isolated - fitted|'}
    for lab in labels(23):
        stem = mods[lab]; out = mcols(stem)
        for tk in ('TriviaQA', 'MMLU'):
            for j, nm in enumerate(['Yes/No','frozen vs fitted','fall','isolated vs fitted']):
                add(f'Table 23 / {lab} / {tk} / {nm}', 'additional models by cell', f'generalization/models/{stem}.csv', f'cell={tk}; {k19[nm]}', out[tk][j])
    O = rd('generalization/other_model_answers.csv')
    for lab in labels(26):
        key = lab.replace(' descr.','').replace(' (descr.)','').strip()
        m = O[O.cell.str.replace(',','').str.strip()==key.replace(',','').strip()]
        assert len(m) == 1, lab
        r = m.iloc[0]
        for nm, col in zip(['negatives','own-fitted','fitted here'], ['negatives','own_answer_readout','fitted_here']):
            add(f'Table 26 / {lab} / {nm}', 'another model\'s answers', 'generalization/other_model_answers.csv', f'cell={r.cell}; col={col}', float(r[col]))
    DC = rd('generalization/dataset_candidates.csv')
    for lab in labels(27):
        c = cond_of(lab)
        for m in ['readout','isolated','fitted','verdict']:
            x = DC[(DC.condition==c)&(DC.method==m)].auroc
            add(f'Table 27 / {lab} / {m} mean', 'dataset-written candidates', 'generalization/dataset_candidates.csv', f'condition={c}; method={m}; mean over {len(x)} cells', x.mean() if len(x) else np.nan)
            add(f'Table 27 / {lab} / {m} median', 'dataset-written candidates', 'generalization/dataset_candidates.csv', f'condition={c}; method={m}; median over {len(x)} cells', x.median() if len(x) else np.nan)
    SL = rd('verdicts/self_labeled.csv')
    for lab in labels(30):
        c = cond_of(lab)
        for m in ['readout','isolated']:
            x = SL[(SL.condition==c)&(SL.method==m)].auroc
            for nm, val in zip(['median','min','max'], [x.median(), x.min(), x.max()]):
                add(f'Table 30 / {lab} / {m} {nm}', 'readouts fitted to verdicts', 'verdicts/self_labeled.csv', f'condition={c}; method={m}; {nm} over {len(x)} cells', val)
    for lab in labels(31):
        for c in ('no_means_correct', 'false_means_correct'):
            for m in ['readout','isolated']:
                x = SL[(SL.cell==lab)&(SL.condition==c)&(SL.method==m)].auroc
                add(f'Table 31 / {lab} / {c} {"plain" if m=="readout" else "isolated"}', 'self-labeled per cell', 'verdicts/self_labeled.csv', f'cell={lab}; condition={c}; method={m}', x.iloc[0] if len(x) else np.nan)
    NV = rd('verdicts/negated_verdicts.csv')
    cm32 = NV.groupby('cell')[['vs_yes_no_verdict','vs_negated_verdict','vs_yes_no_verdict_disagreeing']].mean()
    for nm, col in zip(['Ordinary Yes/No verdict','Negated-prompt verdict','Ordinary verdict, disagreement only'], ['vs_yes_no_verdict','vs_negated_verdict','vs_yes_no_verdict_disagreeing']):
        x = cm32[col].dropna()
        add(f'Table 32 / {nm} / median AUROC', 'negated vs Yes/No verdicts', 'verdicts/negated_verdicts.csv', f'col={col}; median over cells of the cell mean over wordings', x.median())
        add(f'Table 32 / {nm} / cells', 'negated vs Yes/No verdicts', 'verdicts/negated_verdicts.csv', f'col={col}; cells with a defined value', float(len(x)))
    FF = rd('readout/format_first.csv')
    for lab in labels(4):
        r = FF[FF.cell==lab].iloc[0]
        for nm, col in zip(['stem','response position'], ['stem','response_position']):
            add(f'Table 4 / {lab} / {nm}', 'format first', 'readout/format_first.csv', f'cell={lab}; col={col}', float(r[col]))
    P = rd('readout/position.csv')
    for nm, col in (('Before judgment question','pre'),('After judgment question','stem'),('Before verdict','post')):
        add(f'Table 2 / {nm}', 'readout by prompt position', 'readout/position.csv', f'col={col}; mean over 14 cells', P[col].mean())
    ST = rd('readout/steering.csv'); ST = ST[ST.depth==0.75].set_index('format')
    for nm, f in (('Yes/No','yes_no'),('A/B','a_b'),('B/A','b_a')):
        add(f'Table 2 / logit change / {nm}', 'steering slope per unit', 'readout/steering.csv', f'depth=0.75; format={f}', ST.loc[f,'slope'])
    for nm, f in (('Yes/No','yes_no'),('A/B','a_b'),('B/A','b_a')):
        add(f'Table 2 / positive cells / {nm}', 'positive cells', 'readout/steering.csv', f'format={f}; cells_positive (of {int(ST.loc[f,"cells"])})', float(ST.loc[f,'cells_positive']))

def mean4(m, col='auroc', conds=HO4):
    return float(np.mean([hv(H, m, c, col) for c in conds]))
def prose():
    HF = 'subspace/heldout.csv'
    T = rd('generalization/tasks.csv'); tk = lambda t,c,col: T[(T.task==t)&(T.condition==c)][col].iloc[0]
    for loc, t, c, col in (('Abstract','entailment','negated','readout'),('Abstract','entailment','negated','isolated'),('Abstract','safety','negated','readout'),('Abstract','safety','negated','isolated')):
        add(f'{loc} (abstract)', f'negated {t} {col}', 'generalization/tasks.csv', f'task={t}; condition=negated; col={col}', tk(t,c,col))
    add('Intro', 'readout under ordinary question (Yes/No)', HF, 'method=readout; condition=yes_no; mean 14 cells', hv(H,'readout','yes_no'))
    add('Intro', 'readout under negated question', HF, 'method=readout; condition=negated; mean 14 cells', hv(H,'readout','negated'))
    add('Intro', 'isolated judgment under negated question', HF, 'method=isolated; condition=negated; mean 14 cells', hv(H,'isolated','negated'))
    add('Fig 1 caption', 'readout mean over four held-out conditions', HF, 'mean over 4 held-out conditions of 14-cell mean readout', mean4('readout'))
    add('Fig 1 caption', 'isolated mean over four held-out conditions', HF, 'mean over 4 held-out conditions of 14-cell mean isolated', mean4('isolated'))
    add('Fig 1 caption', 'mean fitted-minus-isolated gap over four held-out conditions', HF, 'mean over 4 held-out conditions of (fitted - isolated)', mean4('fitted') - mean4('isolated'))
    PO = rd('readout/position.csv'); PF = rd('readout/position_frozen.csv')
    for loc, col in (('Sec 4','pre'),('Sec 4','stem'),('Sec 4','post')):
        add(loc, f'readout AUROC {col}', 'readout/position.csv', f'col={col}; mean over 14 cells', PO[col].mean())
    add('Sec 4 / App B.2', 'count of cells with post > pre', 'readout/position.csv', 'count(post>pre)', float((PO.post>PO.pre).sum()))
    add('App B.2', 'paired gain post minus pre', 'readout/position.csv', 'mean(post_minus_pre)', PO.post_minus_pre.mean())
    ci = json.load(open(RES+'/readout/position_interval.json'))['post_minus_pre_interval']
    add('App B.2', 'interval lower', 'readout/position_interval.json', 'post_minus_pre_interval[0]', ci[0]); add('App B.2', 'interval upper', 'readout/position_interval.json', 'post_minus_pre_interval[1]', ci[1])
    for col in ('pre', 'stem', 'post'):
        add('App B.2', f'PRE direction and scaler carried unchanged, {col}', 'readout/position_frozen.csv', f'col={col}; mean over 14 cells', PF[col].mean())
    if FOLLOW is not None:
        FF = rd('readout/format_first.csv'); ff3 = FF[FF.cell.isin(FOLLOW) & FF.cell.isin(C14)]
        add('Sec 4 / App B.2', 'format-first stem, mean over the primary cells of the models that follow the format-first prompts', 'readout/format_first.csv', f'col=stem; mean over {len(ff3)} cells (table_4_choice.csv)', ff3.stem.mean())
        add('Sec 4 / App B.2', 'format-first response position, same cells', 'readout/format_first.csv', f'col=response_position; mean over {len(ff3)} cells (table_4_choice.csv)', ff3.response_position.mean())
    O = rd('generalization/other_model_answers.csv'); gapO = (O.own_answer_readout - O.fitted_here).abs()
    add('Sec 4', 'max |own-answer readout - readout fitted on these states|', 'generalization/other_model_answers.csv', 'max |own_answer_readout - fitted_here| over 8 rows', gapO.max())
    DC = rd('generalization/dataset_candidates.csv')
    for mm in ('min', 'max'):
        for m in ('readout', 'fitted'):
            g = DC[(DC.method==m)&(DC.condition.isin(['yes_no','true_false','a_b','b_a','sentence']))].groupby('condition').auroc.mean()
            x = g.min() if mm=='min' else g.max()
            add('Sec 4', f'dataset-written candidates, {m}, {mm} over 5 format conditions of the 8-cell mean', 'generalization/dataset_candidates.csv', f'method={m}; {mm} over conditions yes_no,true_false,a_b,b_a,sentence', x)
    ST = rd('readout/steering.csv').set_index(['depth','format']); SC = rd('readout/steering_compliant.csv'); SCl = rd('readout/steering_cells.csv')
    for f in ('yes_no', 'a_b', 'b_a'):
        add('Sec 4', f'steering logit change per unit, {f}', 'readout/steering.csv', f'depth=0.75; format={f}', ST.loc[(0.75,f),'slope'])
    for f in ('yes_no', 'a_b', 'b_a'):
        add('App B.3', f'steering slope {f}', 'readout/steering.csv', f'depth=0.75; format={f}', ST.loc[(0.75,f),'slope'])
    for f in ('yes_no', 'a_b', 'b_a'):
        add('App B.3', f'slope on rows that comply at every dose, {f}', 'readout/steering_compliant.csv', f'format={f}; mean over 14 cells', SC[SC.format==f].slope.mean())
    for f in ('yes_no', 'a_b', 'b_a'):
        add('App B.3', f'shuffled-label direction slope {f}', 'readout/steering.csv', f'depth=0.75; format={f}; shuffled_slope', ST.loc[(0.75,f),'shuffled_slope'])
    for f in ('yes_no', 'a_b', 'b_a'):
        add('App B.3', f'random direction slope {f}', 'readout/steering.csv', f'depth=0.75; format={f}; random_slope', ST.loc[(0.75,f),'random_slope'])
    for f in ('yes_no', 'a_b', 'b_a'):
        add('App B.3', f'percentage-point rise in "correct" rate dose -1 to +1, {f}', 'readout/steering_compliant.csv', f'format={f}; 100*mean(correct_rate_change) over 14 cells', 100*SC[SC.format==f].correct_rate_change.mean())
    for d, f in ((0.5,'a_b'), (0.5,'b_a'), (0.9,'b_a')):
        add('App B.3', f'slope at depth {d}, {f}', 'readout/steering.csv', f'depth={d}; format={f}', ST.loc[(d,f),'slope'])
    for f in ('yes_no', 'a_b', 'b_a'):
        add('App B.3', f'primary cells with positive slope, {f}', 'readout/steering_cells.csv', f'count(slope>0), format={f}', float((SCl[SCl.format==f].slope>0).sum()))
    def pair_abs(m1, m2, conds):
        return np.mean([ (H[(H.method==m1)&(H.condition==c)].set_index('cell').auroc - H[(H.method==m2)&(H.condition==c)].set_index('cell').auroc).abs().mean() for c in conds])
    add('Sec 5 / App B.1', 'mean |frozen Yes/No readout - fitted readout| on A/B, B/A, sentence', HF, 'mean over 3 formats and 14 cells of |readout - fitted|', pair_abs('readout','fitted',['a_b','b_a','sentence']))
    R = rd('subspace/ranks.csv')
    cm_ = R.groupby('cell')['rank'].median()
    add('Sec 5.1 / C.1', 'median over cells of selected rank', 'subspace/ranks.csv', 'median over cells of cell-median rank', cm_.median())
    add('App C.1', 'cell medians range from (min)', 'subspace/ranks.csv', 'min over cells of cell-median rank', cm_.min()); add('App C.1', 'cell medians range to (max)', 'subspace/ranks.csv', 'max over cells of cell-median rank', cm_.max())
    PS = rd('subspace/pair_sources.csv'); mr = PS[PS.quantity=='mean_rank'].groupby('method').value.mean()
    add('App C.1', 'dims from five response formats only', 'subspace/pair_sources.csv', 'quantity=mean_rank; method=format_pairs; mean over cells', mr['format_pairs'])
    add('App C.1', 'dims from negation pairs only', 'subspace/pair_sources.csv', 'quantity=mean_rank; method=negation_pairs; mean over cells', mr['negation_pairs'])
    CI = rd('readout/condition_identity.csv')
    add('App C.1', 'condition identity decodable, end of candidate', 'readout/condition_identity.csv', 'end_of_candidate mean 14 cells', CI.end_of_candidate.mean())
    add('App C.1 / Sec 7.1', 'condition identity decodable, after instruction', 'readout/condition_identity.csv', 'after_instruction mean 14 cells', CI.after_instruction.mean())
    for loc, m, c in (('Sec 5.2','readout','negated'),('Sec 5.2','isolated','negated'),('Sec 5.2','fitted','negated'),
                      ('Sec 5.2','readout','no_means_correct'),('Sec 5.2','readout','false_means_correct'),('Sec 5.2','readout','zero_means_correct'),
                      ('Sec 5.2','isolated','no_means_correct'),('Sec 5.2','isolated','false_means_correct'),('Sec 5.2','isolated','zero_means_correct')):
        add(loc, f'{m} {c}', HF, f'method={m}; condition={c}; mean 14 cells', hv(H,m,c))
    for loc, t, c, col in (('Sec 5.2','entailment','negated','readout'),('Sec 5.2','safety','negated','readout'),('Sec 5.2','entailment','negated','isolated'),('Sec 5.2','safety','negated','isolated'),('Sec 5.2','entailment','negated','fitted'),('Sec 5.2','safety','negated','fitted')):
        add(loc, f'{t} negated {col}', 'generalization/tasks.csv', f'task={t}; condition=negated; col={col}', tk(t,c,col))
    PC = rd('subspace/principal_components.csv')
    add('App C.5', 'mean absolute cosine to leading PCs', 'subspace/principal_components.csv', 'mean over 14 cells of mean_abs_cos', PC.mean_abs_cos.mean())
    add('App C.5', 'directions shared by subspace and leading PCs, max over cells', 'subspace/principal_components.csv', 'max overlap_directions', PC.overlap_directions.max())
    add('App C.5', 'readout direction sq. norm in subspace (median)', 'subspace/principal_components.csv', 'median share_in_subspace', PC.share_in_subspace.median())
    add('App C.5', 'sq. norm in principal components (median)', 'subspace/principal_components.csv', 'median share_in_components', PC.share_in_components.median())
    add('App C.5', 'removing PCs changes negated readout (median)', 'subspace/principal_components.csv', 'median negated_change_components', PC.negated_change_components.median())
    add('App C.5', 'removing subspace changes negated readout (median)', 'subspace/principal_components.csv', 'median negated_change_subspace', PC.negated_change_subspace.median())
    RS = rd('subspace/readout_share.csv'); cs = RS.groupby('cell').share.mean()
    add('App C.5', 'Yes/No class mean sq. norm in subspace, mean', 'subspace/readout_share.csv', 'mean over cells of cell-mean share', cs.mean())
    rng = np.random.default_rng(0); bs = [cs.values[rng.integers(0,len(cs),len(cs))].mean() for _ in range(10000)]
    add('App C.5', '95% interval lower, cell bootstrap of the mean share (10,000 resamples, seed 0)', 'subspace/readout_share.csv', '2.5 pct of cell bootstrap of mean share', np.percentile(bs,2.5))
    add('App C.5', '95% interval upper, cell bootstrap of the mean share (10,000 resamples, seed 0)', 'subspace/readout_share.csv', '97.5 pct of cell bootstrap of mean share', np.percentile(bs,97.5))
    CT = rd('subspace/controls.csv')
    add('App C.5', 'random subspace of the same rank removed: max |change| of the Yes/No readout', 'subspace/controls.csv', 'max over cells of |yes_no_random_removed - yes_no_readout|', (CT.yes_no_random_removed-CT.yes_no_readout).abs().max())
    add('App C.5', 'random subspace of the same rank removed: max |change| of the negated-question readout', 'subspace/controls.csv', 'max over cells of |negated_random_removed - negated_readout|', (CT.negated_random_removed-CT.negated_readout).abs().max())
    HI = rd('subspace/heldout_intervals.csv').set_index('condition').interval
    for c in ('negated', 'no_means_correct', 'false_means_correct', 'zero_means_correct'):
        ROWS.append(dict(location='Table 7 caption', context=f'cell-bootstrap interval of isolated-minus-fitted gap, {c}', value=HI[c], source_file='subspace/heldout_intervals.csv', source_key=f'condition={c}; col=interval'))
    d_v = np.mean([hv(H,'readout','yes_no','verdict_auroc') - hv(H,'readout',c,'verdict_auroc') for c in HO4])
    d_r = np.mean([hv(H,'readout','yes_no') - hv(H,'readout',c) for c in HO4])
    d_i = np.mean([hv(H,'isolated','yes_no') - hv(H,'isolated',c) for c in HO4])
    add('Sec 6 / Fig 3a', 'mean AUROC drop, verdict (Yes/No minus held-out, 4 conditions)', HF, 'mean over 4 conditions of [verdict(yes_no)-verdict(c)] (14-cell means)', d_v)
    add('Sec 6 / Fig 3a', 'mean AUROC drop, readout', HF, 'mean over 4 conditions of [readout(yes_no)-readout(c)]', d_r)
    add('Sec 6 / Fig 3a', 'mean AUROC drop, isolated judgment (isolated Yes/No minus isolated held-out)', HF, 'mean over 4 conditions of [isolated(yes_no)-isolated(c)]', d_i)
    add('Sec 6', 'max over 4 held-out conditions of |14-cell mean (fitted - isolated)|', HF, 'max over held-out conds of |fitted-isolated|', max(abs(hv(H,'fitted',c)-hv(H,'isolated',c)) for c in HO4))
    NV = rd('verdicts/negated_verdicts.csv')
    cm = NV.groupby('cell')[['verdicts_agree','vs_yes_no_verdict','vs_negated_verdict','vs_yes_no_verdict_disagreeing']].mean()
    add('Sec 6.2', 'isolated judgment predicts Yes/No verdict, median over cells (negated-question states)', 'verdicts/negated_verdicts.csv', 'median over 13 cells of cell-mean vs_yes_no_verdict', cm.vs_yes_no_verdict.median())
    add('Sec 6.2', 'median over cells, vs negated verdict', 'verdicts/negated_verdicts.csv', 'median of vs_negated_verdict', cm.vs_negated_verdict.median())
    add('Sec 6.2', 'isolated predicts Yes/No verdict on disagreeing examples', 'verdicts/negated_verdicts.csv', 'median over cells with defined value (12) of cell-mean vs_yes_no_verdict_disagreeing', cm.vs_yes_no_verdict_disagreeing.median())
    D = rd('subspace/depth.csv'); dv = lambda d,c,m: D[(D.depth==d)&(D.condition==c)&(D.method==m)].auroc.mean()
    for loc, d, c, m in (('Sec 7.1',0.5,'yes_no','readout'),('Sec 7.1',0.5,'no_means_correct','readout'),('Sec 7.1',0.5,'negated','readout'),
                         ('Sec 7.1',0.9,'yes_no','readout'),('Sec 7.1',0.9,'no_means_correct','readout'),('Sec 7.1',0.9,'negated','readout'),
                         ('Sec 7.1',0.75,'negated','isolated'),('Sec 7.1',1.0,'negated','isolated')):
        add(loc, f'depth {d} {c} {m}', 'subspace/depth.csv', f'depth={d}; condition={c}; method={m}; mean 14 cells', dv(d,c,m))
    add('App A', 'readout under negated question (mean)', HF, 'mean 14 cells', hv(H,'readout','negated'))
    add('App A', 'readout under negated question (median)', HF, 'median 14 cells', hv(H,'readout','negated',agg='median'))
    RP = rd('readout/regularization_path.csv').set_index('C')
    for C in (1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1.0):
        r = RP.loc[C]
        add('App B.1', f'logistic probe C={C} source', 'readout/regularization_path.csv', f'C={C}; source_auroc', r.source_auroc)
        add('App B.1', f'C={C} frozen A/B (= source - a_b_drop)', 'readout/regularization_path.csv', f'C={C}; source_auroc-a_b_drop', r.source_auroc-r.a_b_drop)
        add('App B.1', f'C={C} frozen B/A (= source - b_a_drop)', 'readout/regularization_path.csv', f'C={C}; source_auroc-b_a_drop', r.source_auroc-r.b_a_drop)
    PW = rd('subspace/heldout_per_wording.csv'); pv = PW.pivot_table(index=['cell','wording','condition'], columns='method', values='auroc').reset_index(); pv['gap']=pv.fitted-pv.isolated
    mm = pv.groupby(['condition','wording']).gap.median()
    add('App C.2', 'max over held-out condition x wording of |median (fitted - isolated)|', 'subspace/heldout_per_wording.csv', 'max over (condition, wording) of |median over cells of (fitted-isolated)|', mm.abs().max())
    TPM = json.load(open(RES+'/generalization/tasks_per_model.json'))
    for tkk in ('entailment','safety'):
        for key, loc in (('readout_vs_fitted','App C.6'),('isolated_vs_fitted','App C.6')):
            v_ = [TPM[tkk][m][key] for m in TPM[tkk]]
            add(loc, f'{tkk} {key} min over 4 models', 'generalization/tasks_per_model.json', f'{tkk}.*.{key} min', min(v_)); add(loc, f'{tkk} {key} max over 4 models', 'generalization/tasks_per_model.json', f'{tkk}.*.{key} max', max(v_))
    for model in ('Llama-3.1-8B', 'Mistral-7B', 'Qwen2.5-7B', 'Qwen2.5-14B'):
        for c in ('no_means_correct', 'false_means_correct', 'zero_means_correct'):
            g = H[(H.method=='readout')&(H.condition==c)&(H.model==model)]
            add('App D.2', f'verdict AUROC, {c}, {model}', HF, f'model={model}; condition={c}; mean of verdict_auroc over its cells ({len(g)})', g.verdict_auroc.mean())
    DV = rd('verdicts/drop_vs_rate.csv')
    from scipy.stats import spearmanr
    for c in ('negated', 'no_means_correct', 'false_means_correct'):
        g = DV[DV.condition==c]; r = spearmanr(g.readout_drop, g.correct_token_rate)[0]
        add('App D.2', f'Spearman readout drop vs correct-token rate, {c}', 'verdicts/drop_vs_rate.csv', f'condition={c}; spearman(readout_drop, correct_token_rate) over {len(g)} cells', r)
    extras()

def extras():
    if DATA and os.path.isdir(os.path.join(DATA, 'answers')):
        ct = cell_table(); ct.to_csv(os.path.join(OUT, 'cell_sizes.csv'), index=False)
        for _, r in ct.iterrows():
            add('App A (cell sizes and descriptive rule)', f'{r.cell}: negatives on the full cell ({r.rows} rows)', 'answers/*.parquet (+ labels for TriviaQA)', 'count(label==0) on the full cell, greedy answers', float(r.negatives))
        add('App A (cell sizes and descriptive rule)', 'number of cells with fewer than 100 negatives', 'answers/*.parquet', 'count(negatives<100) over 16 cells', float((ct.negatives < 100).sum()))
        add('App A (cell sizes and descriptive rule)', 'rows per cell, min', 'answers/*.parquet', 'min over 16 cells of rows', float(ct.rows.min()))
        add('App A (cell sizes and descriptive rule)', 'rows per cell, max', 'answers/*.parquet', 'max over 16 cells of rows', float(ct.rows.max()))
    if CT5 is not None:
        ct5 = CT5.copy(); ct5['primary'] = ct5.cell.isin(C14); ct5['in_ten_cell_set'] = ct5.primary & ct5.model_follows
        ct5.to_csv(os.path.join(OUT, 'table_4_choice.csv'), index=False)
        add('Table 4 / ten-cell set', 'number of primary cells whose model follows the format-first prompts', 'states/*/format_first/*.npz', 'count over 16 cells of (primary and model_follows)', float(ct5.in_ten_cell_set.sum()))
        for m in ct5.model.unique():
            g = ct5[ct5.model == m]
            add('Table 4 / ten-cell set', f'{m}: lowest letter-format compliance over its cells', 'states/*/format_first/*.npz', 'min over cells of min(top1 A/B, top1 B/A)', float(g[['top1_in_response_tokens_a_b', 'top1_in_response_tokens_b_a']].min().min()))

def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--data', default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    OUT = a.out or os.path.join(a.root, 'tables'); os.makedirs(OUT, exist_ok=True)
    setup(a.root, a.data)
    t_heldout(); t_part2(); prose()
    d = pd.DataFrame(ROWS)[['location', 'context', 'value', 'source_file', 'source_key']]
    d.to_csv(os.path.join(OUT, 'numbers.csv'), index=False)
    tid = d.location.str.extract(r'^Table (\d+)')[0]
    for t in sorted(tid.dropna().unique(), key=int):
        d[tid == t].to_csv(os.path.join(OUT, f'table_{t}.csv'), index=False)
    print(f'{len(d)} numbers written to {OUT}')

if __name__ == '__main__':
    main()
