"""
Novius - a ChatGPT-style AI chat app in pure Python + NumPy.

A local web UI (no frameworks, stdlib only) for a family of four
GPT-style transformer models trained from scratch on English text:

  Avion  V1.0 - the very fast thinker (smallest net, instant replies)
  Qlux   V1.0 - quick thinker (small net, brief thinking phase)
  Crusta V1.0 - balanced thinker (medium net, medium thinking phase)
  Hyron  V1.0 - the most powerful thinker (largest net, deepest thinking)

Every reply is generated token-by-token - no predefined answers. The
"thinking power" of each model is real: bigger network, more context
compute, and a longer thinking phase (the model first generates thought
tokens, then answers conditioned on them).

Files (created next to this script):
  corpus.txt            training text (auto-downloaded on first train)
  weights_avion.txt     trained weights per model (plain-text JSON)
  weights_qlux.txt
  weights_crusta.txt
  weights_hyron.txt

Usage:
  python novius.py train            train all models (~1 hour total on CPU)
  python novius.py train hyron      train a single model
  python novius.py                  start the Novius web UI (http://127.0.0.1:8763)
  python novius.py gen hyron "hi"   one-shot generation in the terminal

Requires:  pip install numpy
"""

import json
import math
import os
import re
import sys
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")
CORPUS_FILE = os.path.join(HERE, "corpus.txt")
PORT = 8763

BASE_CONFIG = {
    "vocab_size": 512,
    "block_size": 64,
}

MODEL_ZOO = {
    "avion": {
        "label": "Avion V1.0",
        "tagline": "The very fast thinker",
        "badge": "Fastest",
        "config": {"d_model": 64, "n_heads": 4, "n_layers": 1, "d_ff": 256},
        "train_steps": 3000,
        "gen": {"max_new_tokens": 80, "temperature": 0.9, "top_k": 30,
                "think_tokens": 0},
    },
    "qlux": {
        "label": "Qlux V1.0",
        "tagline": "Quick thinker",
        "badge": "Quick",
        "config": {"d_model": 96, "n_heads": 4, "n_layers": 2, "d_ff": 384},
        "train_steps": 4000,
        "gen": {"max_new_tokens": 100, "temperature": 0.85, "top_k": 40,
                "think_tokens": 12},
    },
    "crusta": {
        "label": "Crusta V1.0",
        "tagline": "Balanced thinker",
        "badge": "Balanced",
        "config": {"d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 512},
        "train_steps": 5500,
        "gen": {"max_new_tokens": 120, "temperature": 0.8, "top_k": 40,
                "think_tokens": 24},
    },
    "hyron": {
        "label": "Hyron V1.0",
        "tagline": "The most powerful thinker",
        "badge": "Most powerful",
        "config": {"d_model": 192, "n_heads": 6, "n_layers": 3, "d_ff": 768},
        "train_steps": 3000,
        "gen": {"max_new_tokens": 150, "temperature": 0.75, "top_k": 50,
                "think_tokens": 48},
    },
}

BATCH_SIZE = 16
LR = 3e-3
WARMUP = 200
MIN_LR_FRAC = 0.1
CLIP = 1.0
SEED = 42


def weights_path(name):
    return os.path.join(HERE, f"weights_{name}.txt")


# --------------------------------------------------------------------------
# BPE tokenizer
# --------------------------------------------------------------------------

PRETOK = re.compile(r" ?\w+| ?[^\w\s]|\s+")


class BPETokenizer:
    """Byte-pair encoding learned from the training corpus."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = []
        self.tok2id = {}
        self._cache = {}

    def train(self, text):
        counts = Counter(PRETOK.findall(text))
        words = [(list(w), c) for w, c in counts.items()]
        base = sorted({ch for syms, _ in words for ch in syms})
        self.vocab = list(base)
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        while len(self.vocab) < self.vocab_size:
            pairs = Counter()
            for syms, c in words:
                for a, b in zip(syms, syms[1:]):
                    pairs[(a, b)] += c
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new = best[0] + best[1]
            self.merges.append(best)
            self.tok2id[new] = len(self.vocab)
            self.vocab.append(new)
            merged = []
            for syms, c in words:
                out, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                        out.append(new)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                merged.append((out, c))
            words = merged
        self._cache = {}

    def _encode_word(self, w):
        if w in self._cache:
            return self._cache[w]
        syms = list(w)
        while len(syms) > 1:
            best_rank, best_i = None, None
            for i, (a, b) in enumerate(zip(syms, syms[1:])):
                r = self.tok2id.get(a + b)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        ids = [self.tok2id[s] for s in syms if s in self.tok2id]
        self._cache[w] = ids
        return ids

    def encode(self, text):
        ids = []
        for w in PRETOK.findall(text):
            ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids if 0 <= i < len(self.vocab))

    def payload(self):
        return {"vocab": self.vocab, "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_payload(cls, payload):
        tok = cls(len(payload["vocab"]))
        tok.vocab = list(payload["vocab"])
        tok.tok2id = {t: i for i, t in enumerate(tok.vocab)}
        tok.merges = [tuple(m) for m in payload["merges"]]
        tok._cache = {}
        return tok


# --------------------------------------------------------------------------
# Neural network pieces
# --------------------------------------------------------------------------

def gelu(x):
    u = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(u))


def gelu_grad(x):
    c = math.sqrt(2.0 / math.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layernorm_forward(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    inv = 1.0 / np.sqrt((xc ** 2).mean(-1, keepdims=True) + eps)
    xhat = xc * inv
    return g * xhat + b, (xhat, inv, g)


def layernorm_backward(dout, cache):
    xhat, inv, g = cache
    dxhat = dout * g
    m1 = dxhat.mean(-1, keepdims=True)
    m2 = (dxhat * xhat).mean(-1, keepdims=True)
    dx = inv * (dxhat - m1 - xhat * m2)
    axes = tuple(range(dout.ndim - 1))
    return dx, (dout * xhat).sum(axes), dout.sum(axes)


class GPT:
    def __init__(self, config, seed=SEED):
        self.config = dict(config)
        V, T = config["vocab_size"], config["block_size"]
        D, H, L, F = (config["d_model"], config["n_heads"],
                      config["n_layers"], config["d_ff"])
        self.T, self.H, self.L, self.dh = T, H, L, D // H
        rng = np.random.default_rng(seed)
        p = {"tok_emb": rng.normal(0, 0.02, (V, D)),
             "wpe": rng.normal(0, 0.01, (T, D)),
             "lnfg": np.ones(D), "lnfb": np.zeros(D),
             "Wout": rng.normal(0, 0.02, (D, V))}
        for i in range(L):
            pre = f"l{i}_"
            p[pre + "ln1g"], p[pre + "ln1b"] = np.ones(D), np.zeros(D)
            p[pre + "ln2g"], p[pre + "ln2b"] = np.ones(D), np.zeros(D)
            for nm in ("Wq", "Wk", "Wv", "Wo"):
                p[pre + nm] = rng.normal(0, 0.02, (D, D))
                p[pre + "b" + nm[1]] = np.zeros(D)
            p[pre + "W1"] = rng.normal(0, 0.02, (D, F))
            p[pre + "b1"] = np.zeros(F)
            p[pre + "W2"] = rng.normal(0, 0.02, (F, D))
            p[pre + "b2"] = np.zeros(D)
        self.p = p
        self.mask = np.triu(np.full((T, T), -1e9), k=1)

    def _block_forward(self, i, x):
        p, pre = self.p, f"l{i}_"
        B, T, D = x.shape
        H, dh = self.H, self.dh
        a, c_ln1 = layernorm_forward(x, p[pre + "ln1g"], p[pre + "ln1b"])
        aflat = a.reshape(B * T, D)
        q = (aflat @ p[pre + "Wq"] + p[pre + "bq"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = (aflat @ p[pre + "Wk"] + p[pre + "bk"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = (aflat @ p[pre + "Wv"] + p[pre + "bv"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        att = softmax((q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dh) + self.mask[:T, :T])
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B * T, D)
        out = (y @ p[pre + "Wo"] + p[pre + "bo"]).reshape(B, T, D)
        x2 = x + out
        b, c_ln2 = layernorm_forward(x2, p[pre + "ln2g"], p[pre + "ln2b"])
        bflat = b.reshape(B * T, D)
        z1 = bflat @ p[pre + "W1"] + p[pre + "b1"]
        g = gelu(z1)
        z2 = (g @ p[pre + "W2"] + p[pre + "b2"]).reshape(B, T, D)
        cache = {"x": x, "ln1": c_ln1, "aflat": aflat, "q": q, "k": k, "v": v,
                 "att": att, "y": y, "ln2": c_ln2, "bflat": bflat, "z1": z1, "g": g}
        return x2 + z2, cache

    def forward(self, idx):
        B, T = idx.shape
        p = self.p
        x = p["tok_emb"][idx] + p["wpe"][None, :T, :]
        cache = {"idx": idx, "blocks": []}
        for i in range(self.L):
            x, c = self._block_forward(i, x)
            cache["blocks"].append(c)
        h, c_lnf = layernorm_forward(x, p["lnfg"], p["lnfb"])
        cache["lnf"], cache["h"] = c_lnf, h
        return h.reshape(B * T, -1) @ p["Wout"], cache

    def _block_backward(self, i, dx3, c):
        p, pre, grads = self.p, f"l{i}_", {}
        B, T, D = dx3.shape
        H, dh = self.H, self.dh
        dz2 = dx3.reshape(B * T, D)
        grads[pre + "W2"] = c["g"].T @ dz2
        grads[pre + "b2"] = dz2.sum(0)
        dz1 = (dz2 @ p[pre + "W2"].T) * gelu_grad(c["z1"])
        grads[pre + "W1"] = c["bflat"].T @ dz1
        grads[pre + "b1"] = dz1.sum(0)
        dbflat = (dz1 @ p[pre + "W1"].T).reshape(B, T, D)
        dx2, dg2, db2g = layernorm_backward(dbflat, c["ln2"])
        grads[pre + "ln2g"], grads[pre + "ln2b"] = dg2, db2g
        dx2 = dx2 + dx3
        dout = dx2.reshape(B * T, D)
        grads[pre + "Wo"] = c["y"].T @ dout
        grads[pre + "bo"] = dout.sum(0)
        dy = (dout @ p[pre + "Wo"].T).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        datt = dy @ c["v"].transpose(0, 1, 3, 2)
        dv = c["att"].transpose(0, 1, 3, 2) @ dy
        datt = c["att"] * (datt - (datt * c["att"]).sum(-1, keepdims=True))
        scale = 1.0 / math.sqrt(dh)
        dq = (datt @ c["k"]) * scale
        dk = (datt.transpose(0, 1, 3, 2) @ c["q"]) * scale

        def flat(m):
            return m.transpose(0, 2, 1, 3).reshape(B * T, D)

        dqf, dkf, dvf = flat(dq), flat(dk), flat(dv)
        for nm, dm in (("Wq", dqf), ("Wk", dkf), ("Wv", dvf)):
            grads[pre + nm] = c["aflat"].T @ dm
            grads[pre + "b" + nm[1]] = dm.sum(0)
        daflat = (dqf @ p[pre + "Wq"].T + dkf @ p[pre + "Wk"].T
                  + dvf @ p[pre + "Wv"].T).reshape(B, T, D)
        da, dg1, db1g = layernorm_backward(daflat, c["ln1"])
        grads[pre + "ln1g"], grads[pre + "ln1b"] = dg1, db1g
        return da + dx2, grads

    def backward(self, dlogits, cache):
        p, grads = self.p, {}
        B, T, D = cache["h"].shape
        hflat = cache["h"].reshape(B * T, D)
        grads["Wout"] = hflat.T @ dlogits
        dh = (dlogits @ p["Wout"].T).reshape(B, T, D)
        dx, dg, db = layernorm_backward(dh, cache["lnf"])
        grads["lnfg"], grads["lnfb"] = dg, db
        dx = dx.reshape(B, T, D)
        for i in reversed(range(self.L)):
            dx, gi = self._block_backward(i, dx, cache["blocks"][i])
            grads.update(gi)
        grads["wpe"] = dx.sum(0)
        dtok = np.zeros_like(p["tok_emb"])
        np.add.at(dtok, cache["idx"], dx)
        grads["tok_emb"] = dtok
        return grads

    def save(self, tokenizer, path):
        payload = {"config": self.config,
                   "tokenizer": tokenizer.payload(),
                   "params": {k: v.round(6).tolist() for k, v in self.p.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = BPETokenizer.from_payload(payload["tokenizer"])
        model = cls(payload["config"])
        model.p = {k: np.array(v) for k, v in payload["params"].items()}
        return model, tok


class Adam:
    def __init__(self, params):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        gnorm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        if gnorm > CLIP:
            for g in grads.values():
                g *= CLIP / (gnorm + 1e-12)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mhat = self.m[k] / (1 - b1 ** self.t)
            vhat = self.v[k] / (1 - b2 ** self.t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def ensure_corpus():
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            return f.read()
    print("Downloading training corpus ...")
    import urllib.request
    text = urllib.request.urlopen(DATA_URL, timeout=60).read().decode("utf-8")
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def loss_and_grads(model, x, y):
    logits, cache = model.forward(x)
    n = logits.shape[0]
    logits = logits - logits.max(-1, keepdims=True)
    e = np.exp(logits)
    probs = e / e.sum(-1, keepdims=True)
    tgt = y.ravel()
    loss = -np.log(probs[np.arange(n), tgt] + 1e-12).mean()
    dlogits = probs
    dlogits[np.arange(n), tgt] -= 1.0
    dlogits /= n
    grads = model.backward(dlogits, cache)
    return float(loss), grads


def train_model(name):
    spec = MODEL_ZOO[name]
    text = ensure_corpus()
    print(f"[{spec['label']}] corpus {len(text):,} chars, training BPE ...")
    tok = BPETokenizer(BASE_CONFIG["vocab_size"])
    tok.train(text)
    ids = []
    for w in PRETOK.findall(text):
        ids.extend(tok._encode_word(w))
    data = np.array(ids, dtype=np.int32)

    config = dict(BASE_CONFIG, **spec["config"])
    model = GPT(config)
    opt = Adam(model.p)
    rng = np.random.default_rng(SEED)
    steps, T, B = spec["train_steps"], config["block_size"], BATCH_SIZE
    print(f"[{spec['label']}] {sum(v.size for v in model.p.values()):,} params, "
          f"{steps} steps (batch {B}, context {T}) ...")
    for step in range(1, steps + 1):
        warm = min(step / WARMUP, 1.0)
        progress = max(step - WARMUP, 0) / max(steps - WARMUP, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        lr = LR * warm * (MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * cosine)
        ix = rng.integers(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix])
        y = np.stack([data[i + 1:i + T + 1] for i in ix])
        loss, grads = loss_and_grads(model, x, y)
        opt.step(model.p, grads, lr)
        if step % 200 == 0 or step == 1:
            print(f"[{spec['label']}] step {step:5d}  loss {loss:.4f}  lr {lr:.5f}")
        if step % 1000 == 0:
            model.save(tok, weights_path(name))
    model.save(tok, weights_path(name))
    size = os.path.getsize(weights_path(name)) / 1e6
    print(f"[{spec['label']}] done - loss {loss:.4f}, saved {path_mb(size)}")
    return model, tok


def path_mb(size):
    return f"weights file ({size:.1f} MB)"


def train(names):
    for name in names:
        if name not in MODEL_ZOO:
            print(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_ZOO)}")
            return
    for name in names:
        if os.path.exists(weights_path(name)):
            print(f"[{MODEL_ZOO[name]['label']}] already trained, skipping.")
            continue
        train_model(name)


# --------------------------------------------------------------------------
# Generation engine
# --------------------------------------------------------------------------

class Engine:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def models(self):
        return [{"id": k, "label": s["label"], "tagline": s["tagline"],
                 "badge": s["badge"],
                 "ready": os.path.exists(weights_path(k))}
                for k, s in MODEL_ZOO.items()]

    def load(self, name):
        with self._lock:
            if name not in self._cache:
                self._cache[name] = GPT.load(weights_path(name))
            return self._cache[name]

    def chat(self, name, prompt, emit):
        spec = MODEL_ZOO[name]
        if not os.path.exists(weights_path(name)):
            emit({"type": "error",
                  "text": f"{spec['label']} is not trained yet. "
                          f"Run:  python novius.py train {name}"})
            return
        emit({"type": "status", "text": f"Loading {spec['label']} ..."})
        model, tok = self.load(name)
        gen = spec["gen"]
        block = model.config["block_size"]

        thoughts = ""
        if gen["think_tokens"] > 0:
            emit({"type": "status", "text": f"{spec['label']} is thinking ..."})
            thoughts = self._generate(model, tok, prompt, block,
                                      gen["think_tokens"],
                                      gen["temperature"] + 0.15,
                                      max(gen["top_k"], 60), None)
            if thoughts.strip():
                emit({"type": "think", "text": thoughts})

        emit({"type": "status", "text": ""})
        context = prompt + ("\n" + thoughts if thoughts.strip() else "")
        self._stream(model, tok, context, block, gen, emit)

    def _run(self, model, tok, context, temperature, top_k, rng):
        ids = tok.encode(context)[-model.config["block_size"]:] or [0]
        ctx = np.array(ids)[None, :]
        logits, _ = model.forward(ctx)
        logits = logits[-1] / max(temperature, 1e-3)
        probs = softmax(logits)
        if top_k and top_k < len(probs):
            keep = np.argpartition(probs, -top_k)[-top_k:]
            filtered = np.zeros_like(probs)
            filtered[keep] = probs[keep]
            probs = filtered / filtered.sum()
        return ids, int(rng.choice(len(probs), p=probs))

    def _generate(self, model, tok, context, block, max_new, temperature,
                  top_k, seed):
        rng = np.random.default_rng(seed)
        ids = tok.encode(context)[-block:] or [0]
        out = list(ids)
        for _ in range(max_new):
            _, nxt = self._run(model, tok, tok.decode(out[-block:]),
                               temperature, top_k, rng)
            out.append(nxt)
        return tok.decode(out[len(ids):])

    def _stream(self, model, tok, context, block, gen, emit):
        rng = np.random.default_rng()
        ids = tok.encode(context)[-block:] or [0]
        out = list(ids)
        for _ in range(gen["max_new_tokens"]):
            _, nxt = self._run(model, tok, tok.decode(out[-block:]),
                               gen["temperature"], gen["top_k"], rng)
            out.append(nxt)
            piece = tok.decode([nxt])
            if piece:
                emit({"type": "token", "text": piece})


ENGINE = Engine()


# --------------------------------------------------------------------------
# Web UI
# --------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Novius</title>
<style>
  :root { --text:#e8f1f5; --dim:#8fa8b8; --blue:#3b82f6; --green:#34d399;
          --line:rgba(125,180,220,.14); --panel:rgba(10,16,24,.55); }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:'Segoe UI',system-ui,sans-serif; background:#010409;
         color:var(--text); height:100vh; display:flex; overflow:hidden; }
  .aurora { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
  .aurora i { position:absolute; border-radius:50%; filter:blur(90px); }
  .a1 { width:55vw; height:55vw; top:-18vw; left:-12vw;
        background:radial-gradient(circle, rgba(37,99,235,.55), transparent 65%);
        animation:drift1 24s ease-in-out infinite alternate; }
  .a2 { width:45vw; height:45vw; top:-12vw; right:-10vw;
        background:radial-gradient(circle, rgba(16,185,129,.45), transparent 65%);
        animation:drift2 30s ease-in-out infinite alternate; }
  .a3 { width:50vw; height:50vw; bottom:-20vw; left:18vw;
        background:radial-gradient(circle, rgba(45,212,191,.38), transparent 65%);
        animation:drift3 27s ease-in-out infinite alternate; }
  .a4 { width:35vw; height:35vw; bottom:-8vw; right:6vw;
        background:radial-gradient(circle, rgba(59,130,246,.35), transparent 65%);
        animation:drift4 33s ease-in-out infinite alternate; }
  @keyframes drift1 { to { transform:translate(9vw,7vh) scale(1.18); } }
  @keyframes drift2 { to { transform:translate(-7vw,9vh) scale(1.22); } }
  @keyframes drift3 { to { transform:translate(6vw,-8vh) scale(1.15); } }
  @keyframes drift4 { to { transform:translate(-8vw,-6vh) scale(1.2); } }
  ::-webkit-scrollbar { width:8px; }
  ::-webkit-scrollbar-thumb { background:rgba(90,150,190,.25); border-radius:8px; }
  aside { width:270px; background:rgba(4,8,14,.72);
          backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
          padding:14px; display:flex; flex-direction:column; gap:8px;
          border-right:1px solid var(--line); z-index:2; }
  .logo { font-size:20px; font-weight:700; padding:6px 8px 14px; letter-spacing:.4px;
          background:linear-gradient(90deg,#60a5fa,#34d399);
          -webkit-background-clip:text; background-clip:text; color:transparent; }
  .newchat { background:var(--panel); border:1px solid var(--line); color:var(--text);
             padding:10px 12px; border-radius:10px; cursor:pointer; text-align:left;
             font-size:14px; }
  .newchat:hover { background:rgba(59,130,246,.15); border-color:rgba(59,130,246,.4); }
  .models { margin-top:10px; display:flex; flex-direction:column; gap:8px;
            overflow-y:auto; }
  .mtitle { font-size:11px; text-transform:uppercase; letter-spacing:1px;
            color:var(--dim); padding:4px 8px; }
  .mcard { border:1px solid var(--line); background:transparent; color:var(--text);
           border-radius:12px; padding:10px 12px; cursor:pointer; text-align:left; }
  .mcard:hover { background:rgba(56,189,248,.08); }
  .mcard.sel { border-color:rgba(52,211,153,.55);
               background:linear-gradient(90deg, rgba(59,130,246,.14), rgba(52,211,153,.10)); }
  .mcard .nm { font-weight:600; font-size:14px; display:flex;
               justify-content:space-between; align-items:center; }
  .mcard .tg { font-size:12px; color:var(--dim); margin-top:3px; }
  .badge { font-size:10px; padding:2px 8px; border-radius:99px;
           background:linear-gradient(90deg,#3b82f6,#34d399); color:#fff;
           font-weight:600; }
  .mcard.off .badge { background:#3a4450; }
  .foot { margin-top:auto; font-size:11px; color:var(--dim); padding:8px;
          line-height:1.5; }
  main { flex:1; display:flex; flex-direction:column; z-index:1; }
  header { padding:14px 22px; border-bottom:1px solid var(--line); font-size:14px;
           color:var(--dim); background:rgba(4,8,14,.5);
           backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); }
  header b { color:var(--text); }
  #chat { flex:1; overflow-y:auto; padding:24px 0; }
  .inner { max-width:760px; margin:0 auto; padding:0 22px; display:flex;
           flex-direction:column; gap:22px; }
  .row { display:flex; gap:14px; }
  .row.user { justify-content:flex-end; }
  .avatar { width:30px; height:30px; border-radius:8px;
            background:linear-gradient(135deg,#3b82f6,#34d399);
            display:flex; align-items:center; justify-content:center; font-weight:700;
            font-size:14px; flex-shrink:0; }
  .row.user .bubble { background:rgba(30,45,60,.65); border:1px solid rgba(125,180,220,.15);
                      border-radius:18px; padding:12px 16px;
                      max-width:75%; white-space:pre-wrap; line-height:1.55; }
  .row.ai .content { max-width:calc(100% - 44px); line-height:1.65;
                     white-space:pre-wrap; }
  .status { color:var(--dim); font-size:13px; font-style:italic; }
  details.think { margin-bottom:10px; font-size:13px; }
  details.think summary { cursor:pointer; color:var(--dim); font-style:italic; }
  details.think .tbody { color:var(--dim); white-space:pre-wrap; border-left:2px solid
                         rgba(52,211,153,.5); padding:6px 0 6px 12px; margin-top:6px; }
  .cursor { display:inline-block; width:8px; height:16px;
            background:linear-gradient(180deg,#3b82f6,#34d399);
            animation:blink 1s steps(2) infinite; vertical-align:-2px; }
  @keyframes blink { 50% { opacity:0; } }
  .composer { padding:14px 22px 22px; }
  .box { max-width:760px; margin:0 auto; display:flex; gap:10px; align-items:flex-end;
         background:rgba(8,14,22,.75); backdrop-filter:blur(16px);
         -webkit-backdrop-filter:blur(16px); border:1px solid var(--line);
         border-radius:26px; padding:10px 10px 10px 18px; }
  textarea { flex:1; background:transparent; border:none; outline:none; resize:none;
             color:var(--text); font:inherit; max-height:140px; line-height:1.5; }
  #send { width:38px; height:38px; border-radius:50%; border:none; cursor:pointer;
          background:linear-gradient(135deg,#3b82f6,#34d399); color:#fff;
          font-size:17px; flex-shrink:0; }
  #send:disabled { background:#2a3947; }
  .hint { text-align:center; font-size:11px; color:var(--dim); margin-top:8px; }
  .welcome { text-align:center; margin-top:12vh; }
  .welcome h1 { font-size:30px; font-weight:700;
                background:linear-gradient(90deg,#60a5fa,#34d399);
                -webkit-background-clip:text; background-clip:text; color:transparent; }
  .welcome p { color:var(--dim); margin-top:10px; font-size:14px; }
</style>
</head>
<body>
<div class="aurora"><i class="a1"></i><i class="a2"></i><i class="a3"></i><i class="a4"></i></div>
<aside>
  <div class="logo">Novi<span>us</span></div>
  <button class="newchat" id="newchat">+ New chat</button>
  <div class="mtitle">Models</div>
  <div class="models" id="models"></div>
  <div class="foot">Novius runs four transformer models trained from scratch
    in NumPy. Every word is generated live - nothing is scripted.</div>
</aside>
<main>
  <header>Chatting with <b id="curmodel">Crusta V1.0</b></header>
  <div id="chat"><div class="inner" id="feed">
    <div class="welcome"><h1>Welcome to Novi<span>us</span></h1>
    <p>Pick a model on the left, then say anything.</p></div>
  </div></div>
  <div class="composer"><div class="box">
    <textarea id="input" rows="1" placeholder="Message Novius..."></textarea>
    <button id="send" title="Send">&#8593;</button>
  </div>
  <div class="hint">Novius is a tiny research model trained on Shakespeare -
  expect poetry, not facts. Enter to send.</div></div>
</main>
<script>
const feed = document.getElementById('feed');
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const curModel = document.getElementById('curmodel');
let model = 'crusta';
let busy = false;

async function loadModels() {
  const models = await (await fetch('/api/models')).json();
  const box = document.getElementById('models');
  box.innerHTML = '';
  for (const m of models) {
    const b = document.createElement('button');
    b.className = 'mcard' + (m.id === model ? ' sel' : '') + (m.ready ? '' : ' off');
    b.innerHTML = `<div class="nm">${m.label}
      <span class="badge">${m.ready ? m.badge : 'not trained'}</span></div>
      <div class="tg">${m.tagline}</div>`;
    b.onclick = () => { model = m.id; curModel.textContent = m.label;
      loadModels(); };
    box.appendChild(b);
  }
}

function scroll() { chat.scrollTop = chat.scrollHeight; }

function addUser(text) {
  const w = document.querySelector('.welcome'); if (w) w.remove();
  const row = document.createElement('div');
  row.className = 'row user';
  const b = document.createElement('div');
  b.className = 'bubble'; b.textContent = text;
  row.appendChild(b); feed.appendChild(row); scroll();
}

function addAI() {
  const row = document.createElement('div');
  row.className = 'row ai';
  row.innerHTML = `<div class="avatar">N</div>
    <div class="content"><div class="status"></div>
    <details class="think" style="display:none">
      <summary>Thought process</summary><div class="tbody"></div></details>
    <span class="reply"></span><span class="cursor"></span></div>`;
  feed.appendChild(row); scroll();
  return row;
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true; sendBtn.disabled = true; input.value = '';
  addUser(text);
  const row = addAI();
  const status = row.querySelector('.status');
  const think = row.querySelector('.think');
  const tbody = row.querySelector('.tbody');
  const reply = row.querySelector('.reply');
  let got = false;
  try {
    const resp = await fetch('/api/chat', { method:'POST',
      body: JSON.stringify({ model, prompt: text }) });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream:true });
      let i;
      while ((i = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line) continue;
        const evt = JSON.parse(line);
        if (evt.type === 'status') { status.textContent = evt.text; }
        else if (evt.type === 'think') { think.style.display = 'block';
          tbody.textContent += evt.text; }
        else if (evt.type === 'token') { got = true;
          reply.textContent += evt.text; scroll(); }
        else if (evt.type === 'error') { status.textContent = evt.text; }
      }
    }
  } catch (e) { status.textContent = 'Connection error: ' + e; }
  row.querySelector('.cursor').remove();
  if (!got && !think.open && !tbody.textContent)
    status.textContent = status.textContent || '(no output)';
  busy = false; sendBtn.disabled = false; input.focus();
}

sendBtn.onclick = send;
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});
document.getElementById('newchat').onclick = () => {
  feed.innerHTML = `<div class="welcome"><h1>Welcome to Novi<span>us</span></h1>
    <p>Pick a model on the left, then say anything.</p></div>`;
};
loadModels();
input.focus();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        elif self.path == "/api/models":
            body = json.dumps(ENGINE.models()).encode("utf-8")
            self._send(200, "application/json", body)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/api/chat":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        name = req.get("model", "crusta")
        prompt = str(req.get("prompt", "")).strip()
        if name not in MODEL_ZOO or not prompt:
            self._send(400, "application/json",
                       json.dumps({"type": "error", "text": "bad request"}).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(event):
            try:
                self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (ConnectionAbortedError, BrokenPipeError):
                raise GenerationCancelled()

        try:
            ENGINE.chat(name, prompt, emit)
        except GenerationCancelled:
            return

    def do_HEAD(self):
        self._send(200, "text/html", b"")


class GenerationCancelled(Exception):
    pass


def serve():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Novius is running at {url}  (Ctrl+C to stop)")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nNovius stopped.")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "train":
        names = sys.argv[2:] or list(MODEL_ZOO)
        train(names)
    elif cmd == "gen":
        name = sys.argv[2] if len(sys.argv) > 2 else "crusta"
        prompt = sys.argv[3] if len(sys.argv) > 3 else ""
        events = []
        ENGINE.chat(name, prompt, events.append)
        for e in events:
            if e["type"] in ("think", "token"):
                print(e["text"], end="")
        print()
    else:
        serve()


if __name__ == "__main__":
    main()
