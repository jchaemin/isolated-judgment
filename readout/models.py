"""Model loading, chat template check, final norm, state capture, and shared-prefix key-value reuse."""
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = torch.bfloat16
LARGE_4BIT = {"Qwen2.5-32B-Instruct": "unsloth/Qwen2.5-32B-Instruct-bnb-4bit", "gemma-2-27b-it": "unsloth/gemma-2-27b-it-bnb-4bit"}


def load_model(model_name, device=None):
    if device is None:
        device = os.environ.get("READOUT_DEVICE", "auto")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    kw = dict(dtype=DTYPE, device_map=device)
    if device == "auto" and torch.cuda.is_available():
        head = 1500
        mm = {}
        for i in range(torch.cuda.device_count()):
            free = torch.cuda.mem_get_info(i)[0] // 2**20
            mm[i] = f"{max(free - head, 0)}MiB"
        kw["max_memory"] = mm
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    model.eval()
    return model, tok


def load_large(model_name, device=None):
    if device is None:
        device = os.environ.get("READOUT_DEVICE", "auto")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(LARGE_4BIT[model_name.split("/")[-1]], device_map=device)
    model.eval()
    return model, tok


def chat_model(model_name):
    T = model_name.split("/")[-1]
    model, tok = load_large(model_name) if T in LARGE_4BIT else load_model(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def uses_chat_template(tok):
    return getattr(tok, "chat_template", None) is not None


def final_norm(model):
    m = model
    for _ in range(6):
        for attr in ("norm", "final_layernorm", "ln_f"):
            if hasattr(m, attr) and hasattr(getattr(m, attr), "forward"):
                return getattr(m, attr)
        if hasattr(m, "model"):
            m = m.model
        elif hasattr(m, "base_model"):
            m = m.base_model
        else:
            break
    raise AttributeError("could not locate final norm; inspect model.model")


def wrap(tok, text, chat=True):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) if chat else text + "\nAnswer:"


def decide_norm(h, lg_ref, Wu, nrm):
    d_raw = float(((h.to(Wu.device).to(Wu.dtype) @ Wu.T).float() - lg_ref.to(Wu.device)).abs().max())
    d_n = float(((nrm(h.to(next(nrm.parameters()).device)).to(Wu.device).to(Wu.dtype) @ Wu.T).float() - lg_ref.to(Wu.device)).abs().max())
    return d_n < d_raw


def apply_state(h, use_norm, Wu, nrm):
    g = nrm(h.to(next(nrm.parameters()).device)) if use_norm else h
    return g.to(Wu.device).float()


def state(o, h, Wu, nrm):
    ref = o.logits[:, -1, :].float(); use_norm = decide_norm(h, ref, Wu, nrm); g = apply_state(h, use_norm, Wu, nrm)
    return g, (g.to(Wu.dtype) @ Wu.T).float()


def forward(net, Wu, nrm, tok, texts, batch_size, tid):
    n = len(texts); G = np.zeros((n, Wu.shape[1]), np.float16); Z = np.zeros((n, len(tid)), np.float32)
    t20i = np.zeros((n, 20), np.int32); tid_t = torch.tensor(tid)
    for i in range(0, n, batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(net.device)
        with torch.no_grad():
            o = net(**enc, output_hidden_states=True)
            g, lg = state(o, o.hidden_states[-1][:, -1, :], Wu, nrm)
            pr = torch.softmax(lg, -1); ti = torch.topk(pr, 20, dim=-1).indices; s = slice(i, i + g.shape[0])
            G[s] = g.cpu().numpy().astype(np.float16); Z[s] = lg[:, tid_t.to(lg.device)].cpu().numpy(); t20i[s] = ti.cpu().numpy()
        del enc, o, g, lg, pr
    torch.cuda.empty_cache()
    return G, Z, t20i


def plan(tok, texts, extra=None):
    keys = list(texts); B = len(texts[keys[0]])
    ids = {k: [tok(t).input_ids for t in texts[k]] for k in keys}; cuts = []
    for i in range(B):
        seqs = [ids[k][i] for k in keys]; m = min(len(s) for s in seqs); c = 0
        while c < m and all(s[c] == seqs[0][c] for s in seqs):
            c += 1
        cuts.append(c)
    suf = {}
    for k in keys:
        tails = {tuple(ids[k][i][cuts[i]:]) for i in range(B)}
        if len(tails) != 1:
            return None
        t = list(next(iter(tails)))
        if len(t) < 1:
            return None
        suf[k] = t + list((extra or {}).get(k, []))
    if min(cuts) < 1:
        return None
    return dict(prefix=[ids[keys[0]][i][:cuts[i]] for i in range(B)], suffix=suf)


class Shared:
    def __init__(self, model, tok, pl):
        self.model, self.pl = model, pl; dev = model.device; B = len(pl["prefix"]); Lp = max(len(x) for x in pl["prefix"]); pad = tok.pad_token_id
        P = torch.full((B, Lp), pad, dtype=torch.long); M = torch.zeros((B, Lp), dtype=torch.long)
        for i, x in enumerate(pl["prefix"]):
            P[i, Lp - len(x):] = torch.tensor(x); M[i, Lp - len(x):] = 1
        self.P, self.M, self.Lp, self.B = P.to(dev), M.to(dev), Lp, B
        with torch.no_grad():
            o = model(input_ids=self.P, attention_mask=self.M, use_cache=True, logits_to_keep=1); self.cache = o.past_key_values; del o

    def extend(self, key):
        s = self.pl["suffix"][key]; self.cache.crop(self.Lp); dev = self.P.device
        Tt = torch.tensor([s] * self.B, dtype=torch.long, device=dev); Mfull = torch.cat([self.M, torch.ones((self.B, len(s)), dtype=torch.long, device=dev)], 1)
        if len(s) > 1:
            with torch.no_grad():
                self.model(input_ids=Tt[:, :-1], attention_mask=Mfull[:, :-1], past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        return Tt[:, -1:], Mfull, self.Lp + len(s) - 1

    def full(self, key):
        s = self.pl["suffix"][key]; self.cache.crop(self.Lp); dev = self.P.device
        Tt = torch.tensor([s] * self.B, dtype=torch.long, device=dev); Mfull = torch.cat([self.M, torch.ones((self.B, len(s)), dtype=torch.long, device=dev)], 1)
        with torch.no_grad():
            return self.model(input_ids=Tt, attention_mask=Mfull, past_key_values=self.cache, use_cache=True, logits_to_keep=1)


def encode_full(tok, texts, extra, device):
    enc = tok(texts, return_tensors="pt", padding=True); ids_, am_ = enc.input_ids, enc.attention_mask
    if extra:
        e = torch.tensor([extra] * ids_.shape[0], dtype=ids_.dtype); ids_ = torch.cat([ids_, e], 1); am_ = torch.cat([am_, torch.ones_like(e)], 1)
    return ids_.to(device), am_.to(device)


def identity_bad(cs):
    cs = np.asarray(cs, float)
    return bool(np.median(cs) < 0.999 or cs.min() < 0.95 or (len(cs) >= 100 and (cs < 0.99).mean() > 0.05))


def cosine(A, B):
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)


def save(path, compressed=False, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path + ".tmp.npz"
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays); os.replace(tmp, path)
