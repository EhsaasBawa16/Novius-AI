"""
Novius - a ChatGPT-style AI chat app in pure Python + NumPy.

A local web UI (no frameworks, stdlib only) for a family of four
GPT-style transformer models trained from scratch on general English:

  Avion  V3.0 - the very fast thinker (smallest net, instant replies)
  Qlux   V3.0 - quick thinker (small net, brief thinking phase)
  Crusta V3.0 - balanced thinker (medium net, medium thinking phase)
  Hyron  V3.0 - the most powerful thinker (largest net, deepest thinking,
                trained on real Python source code too)

Training data (auto-downloaded on first train, ~400 MB total):
  WikiText-103      real English Wikipedia articles
  WordNet           dictionary definitions for ~80k word senses, turned into
                    question -> answer pairs ("what is love? / Love is ...")
  Webster dictionary  ~40k extra word definitions
  TinyStories       clean, grammatical simple English stories
  chat corpus       internet slang, broken grammar and everyday Q&A
  CPython sources   real Python code, so the models can talk about code

Novius also LEARNS FROM YOU: every chat exchange is saved to memory.jsonl
and a background learner periodically finetunes the models on what you
talked about, so the more you chat the better it gets (run
"python novius.py learn" to force a learning session).

Sign-in is required: the app opens on a sign-in / sign-up page (email +
password, or Google / Facebook). Accounts and every named chat are stored
locally in JSON files next to this script; the sidebar keeps your named
chats with a live search. Google / Facebook sign-in uses standard OAuth2
and is enabled by setting the GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
FACEBOOK_CLIENT_ID and FACEBOOK_CLIENT_SECRET environment variables
(redirect URI: http://127.0.0.1:8763/oauth/callback/<provider>).

Novius has NO predefined answers. Every reply is generated token-by-token
by the neural models, which are trained in two passes: a large main pass on
general English (wiki + stories + dictionary Q&A + slang), then a finetune
pass on dialogue-formatted Q&A and chat data so the models answer in the
right format. The only non-neural tool is an exact calculator for arithmetic
("what is 23*7", "15% of 80", "sqrt 144"), which small networks cannot
learn reliably. A second tool, "Internet research" (toggle in the UI), lets
Novius search Wikipedia live and answer from real retrieved text with a
source link - useful for facts the tiny models never memorized. Otherwise
answers come purely from the neural models. The "thinking power" of each
model is real: bigger network, more layers, and a longer thinking phase.

Files (created next to this script):
  corpus.txt            assembled training text
  corpus_tokens.npy     tokenized corpus cache
  corpus_tokenizer.json tokenizer cache
  code_corpus.txt       Python source code corpus
  finetune_v4.txt       Q&A + chat + code data used for the final tuning pass
  finetune_tokens_v4.npy  tokenized finetune cache
  memory.jsonl          everything you chat - the online-learning data
  users.json            local accounts (email sign-up, OAuth identities)
  sessions.json         active sign-in sessions
  chats.json            your named chats, per account
  weights_avion.txt     trained weights per model (plain-text JSON)
  weights_qlux.txt
  weights_crusta.txt
  weights_hyron.txt

Usage:
  python novius.py train            train all models (several hours on CPU)
  python novius.py train hyron      train a single model
  python novius.py finetune         final Q&A tuning pass on all models
  python novius.py finetune hyron   tune a single model
  python novius.py                  start the Novius web UI (http://127.0.0.1:8763)
  python novius.py gen hyron "hi"   one-shot generation in the terminal
  python novius.py learn            finetune all models on your saved chats
  python novius.py learn hyron      finetune one model on your saved chats

Requires:  pip install numpy
"""

import base64
import hashlib
import json
import math
import os
import re
import secrets
import sys
import threading
import time
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WIKI_URL = "https://wikitext.smerity.com/wikitext-103-raw-v1.zip"
WORDNET_URL = ("https://raw.githubusercontent.com/nltk/nltk_data/"
               "gh-pages/packages/corpora/wordnet.zip")
DICT_URL = "https://raw.githubusercontent.com/adambom/dictionary/master/dictionary.json"
STORIES_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
               "resolve/main/TinyStories-train.txt")
CORPUS_FILE = os.path.join(HERE, "corpus.txt")
TOKENS_FILE = os.path.join(HERE, "corpus_tokens.npy")
TOKCACHE_FILE = os.path.join(HERE, "corpus_tokenizer.json")
FT_FILE = os.path.join(HERE, "finetune_v4.txt")
FT_TOKENS_FILE = os.path.join(HERE, "finetune_tokens_v4.npy")
PORT = int(os.environ.get("PORT", "8763"))
HOST = os.environ.get("HOST", "127.0.0.1")

BASE_CONFIG = {
    "vocab_size": 2048,
    "block_size": 64,
}

MODEL_ZOO = {
    "avion": {
        "label": "Avion V3.0",
        "tagline": "The very fast thinker",
        "badge": "Fastest",
        "config": {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 256},
        "train_steps": 4000,
        "gen": {"max_new_tokens": 90, "temperature": 0.45, "top_k": 40,
                "think_tokens": 0, "rep": 1.12},  # instant
    },
    "qlux": {
        "label": "Qlux V3.0",
        "tagline": "Quick thinker",
        "badge": "Quick",
        "config": {"d_model": 128, "n_heads": 4, "n_layers": 2, "d_ff": 512},
        "train_steps": 6000,
        "gen": {"max_new_tokens": 110, "temperature": 0.4, "top_k": 50,
                "think_tokens": 24, "rep": 1.15},
    },
    "crusta": {
        "label": "Crusta V3.0",
        "tagline": "Balanced thinker",
        "badge": "Balanced",
        "config": {"d_model": 192, "n_heads": 6, "n_layers": 3, "d_ff": 768},
        "train_steps": 7000,
        "gen": {"max_new_tokens": 130, "temperature": 0.35, "top_k": 40,
                "think_tokens": 48, "rep": 1.15},
    },
    "hyron": {
        "label": "Hyron V3.0",
        "tagline": "The most powerful thinker",
        "badge": "Most powerful",
        "config": {"d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024},
        "train_steps": 7000,
        "gen": {"max_new_tokens": 200, "temperature": 0.3, "top_k": 40,
                "think_tokens": 96, "rep": 1.18},
    },
}

BATCH_SIZE = 16
LR = 3e-3
WARMUP = 200
MIN_LR_FRAC = 0.1
CLIP = 1.0
SEED = 42

# Online learning: every chat exchange is saved to memory.jsonl and the
# background Learner thread finetunes the models on it every LEARN_EVERY
# new exchanges, so Novius keeps improving from real conversations.
LEARN_FILE = os.path.join(HERE, "memory.jsonl")
LEARN_EVERY = 6
LEARN_STEPS = 60
LEARN_LR = 3e-4
LEARN_POLL = 15

# Code requests get a colder, more focused sampling profile.
_CODE_RE = re.compile(r"\b(code|coding|function|script|program|python|"
                      r"algorithm|snippet)\b", re.I)


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
        words = {}
        pair_counts = Counter()
        pair_words = {}
        for wid, (w, c) in enumerate(counts.items()):
            syms = list(w)
            words[wid] = [syms, c]
            for pair in zip(syms, syms[1:]):
                pair_counts[pair] += c
                pair_words.setdefault(pair, set()).add(wid)
        base = sorted({ch for syms, _ in words.values() for ch in syms})
        self.vocab = list(base)
        self.tok2id = {t: i for i, t in enumerate(self.vocab)}
        while len(self.vocab) < self.vocab_size and pair_counts:
            best = max(pair_counts, key=pair_counts.get)
            new = best[0] + best[1]
            self.merges.append(best)
            self.tok2id[new] = len(self.vocab)
            self.vocab.append(new)
            for wid in pair_words.pop(best, set()):
                syms, c = words[wid]
                for pair in zip(syms, syms[1:]):
                    pair_counts[pair] -= c
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                out, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                        out.append(new)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                words[wid][0] = out
                for pair in zip(out, out[1:]):
                    pair_counts[pair] += c
                    pair_words.setdefault(pair, set()).add(wid)
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

SLANG_CORPUS = """
yo whats good
hiii heyy wsup
wsp bro
nm u?
hows it going
pretty good ngl hbu
long time no see!
fr we should catch up soon
wyd this weekend
idk yet probs just chill
u tryna hang out later?
bet im down
aight see u then
later bro
peace
gn yall
gm world
tbh that movie was mid ngl
fr fr no cap
sheesh bussin bussin
lowkey sus ngl
thats so cringe lol
W take fr
took an L today smh
brb food time
idk abt that tbh
u wanna hop on the game?
ur goated for that
bro this pizza is bussin no cap
its giving main character energy
vibing rn ngl
bet lets gooo
pls dont be sus
omg yesss slay
thats cap and u know it
my bad gang
yooo thats fire
dawg im dead lol
hbu? wyd tonite
smh they did us dirty
yeet
im so tired rn ugh
cant rn im busy af
tyty appreciate it
np anytime
gg well played
ez clap
ratio + L + didnt ask
go touch grass bro
based take ngl
sheeeesh
lets gooo we won
im cooked
thats tuff
on god fr
deadass?
lowkey wanna nap
highkey obsessed with this song
its the bare minimum tbh
living rent free in my head
he understood the assignment
main character moment
not me forgetting again lol
we move
sending good vibes ur way
u ate that fr
no crumbs left
periodt
im weak lol
bruh moment
bro really said that??
nah cus why is he like that
istg this app is broken
gng we up
locked in fr
cook mode activated
let him cook
nah hes cooked
im gonna crash out lol
chill bro its not that deep
fr tho
real talk
straight up
no lie detected
u already know
say less
done deal
bet
my guy
big brain move
galaxy brain take
smooth brain moment
skill issue tbh
git gud
cry about it
stay mad lol
cope harder
seethe
malding rn
poggers
lets get this bread
grind never stops
hustle mode on
im him fr
she is her
ate and left no crumbs
devoured that
unserious behavior
chronically online take
touch some grass my dude
logging off for today
afk for a bit
back kk
wb everyone
missed u all
good vibes only
negative energy left the chat
this chat is so alive rn
lurking lol
just got here wdid i miss
nothing much same old
spill the tea
no tea no shade
thats the tea sis
shade thrown lol
messy behavior ngl
drama everywhere today
minding my business
nosy ass lol
jk jk
lol lmao rofl
hahaha thats funny af
dying of laughter rn
im crying this is so funny
send help lol
help im screaming
screaming crying throwing up
emotional damage
bruhhhh
broooo
duuuude
mannn
ugh fine
ok ok i get it
my bad fr
forgive me pls
u know i love u right
aww stahp
ur so sweet
ily mean it
heart u too
bestie energy
bestie ur iconic
slay queen
yasss
werk
ok period
thats on god
on my mama
swear down
promise fr
pinky promise lol
jk unless?
unlesss...
wait what
hold up
say sike rn
ur kidding right
no way bro
wayyy bro
fr?
for real for real
hundred percent
facts no printer
receipts or it didnt happen
screenshot that lol
archived forever
screenshotted and framed
lmaooo im framing this
old but gold
vintage meme
retro vibes
dank meme alert
meme lord
king of memes
meme review
reviewing now
its a masterpiece
chef's kiss
muah
italian kiss emoji
mwah mwah
okay love u bye
cya
adios
sayonara
au revoir my friend
farewell gamer
gg go next
next round
rematch?
u bet
bring it on
im ready
game on
lets a go
wahoo
yippee
hurray we did it
confetti time
party mode
lets celebrate
pizza on me
ur the best
ikr
ikr!!
totally
for sure
absolutely
negative ghost rider
affirmative
roger that
copy that
10-4 good buddy
over and out
what is love?
love is when u care about someone so much u cant stop thinking about them
love is a warm feeling in ur chest when someone makes u happy
love is caring about someone more than urself
being in love means u want the best for that person always
what is a friend?
a friend is someone who got ur back no matter what
what is happiness?
happiness is when u feel good inside for no reason at all
what is sadness?
sadness is that heavy feeling when something hurts ur heart
what is life?
life is the time u get to spend doing things u care about with people u love
what is music?
music is sound that makes u feel something
what is a family?
family is the people who love u even when ur difficult
what is school?
school is where u go to learn new things every day
what is the internet?
the internet is a giant network connecting computers all over the world
what is a computer?
a computer is a machine that follows instructions really fast
what is ai?
ai is when a computer learns to do things that normally need a human brain
what is a phone?
a phone is a device u use to talk and text people
what is money?
money is what people use to buy things they need or want
what is food?
food is what u eat to get energy and stay alive
what is sleep?
sleep is when ur body rests and recharges
what is a dream?
a dream is the story ur brain makes while u sleep
what is time?
time is how we measure moments passing
what is the sun?
the sun is the giant star that gives us light and warmth
what is the moon?
the moon is the rock that circles earth and glows at night
what is rain?
rain is water falling from clouds
what is snow?
snow is frozen water falling in soft white flakes
how are you?
im doing good thanks for asking
im great how about u
whats up?
not much just vibing
who are you?
im novius a little ai u can chat with
whats ur name?
my name is novius
what can u do?
i can chat with u and try to answer stuff
how old are u?
im a computer program so i dont really age lol
where are u from?
i live inside a computer rn
what day is it?
i dont have a calendar bro sorry
tell me a joke
why dont scientists trust atoms cuz they make up everything
good morning
morning hope today treats u well
good night
gn sleep well
thank you
np anytime
i love u
aww ily too ur the best
im sad
im sorry to hear that wanna talk about it
im happy
thats awesome glad ur feeling good
im bored
same lets chat about something fun
im tired
u should get some rest fr
what should i eat?
something bussin idk maybe pizza
whats the weather?
i cant check outside but hope its nice where u are
"""


DIALOGUE_CORPUS = """
user: hi
novius: hey! whats up
user: hello
novius: hi there! how are u doing
user: heyy
novius: heyy whats good
user: yo
novius: yo! wyd
user: wsp
novius: nm just chilling wbu
user: how are you
novius: im doing great hbu
user: how r u
novius: pretty good today hbu
user: who are you
novius: im novius a little ai living in ur computer
user: what is ur name
novius: my name is novius!
user: what can u do
novius: i can chat do math and try to answer stuff
user: thank you
novius: np anytime
user: thanks
novius: u got it
user: bye
novius: cya later!
user: gn
novius: gn sleep well
user: gm
novius: gm! hope today treats u well
user: i love u
novius: aww ily too ur the best
user: i am sad
novius: im sorry to hear that wanna talk about it
user: im sad
novius: that sucks im here for u wanna chat
user: im happy
novius: thats awesome! glad ur feeling good
user: im bored
novius: same wanna talk about something fun
user: im tired
novius: u should rest fr
user: tell me a joke
novius: why dont scientists trust atoms cuz they make up everything
user: tell me a story
novius: once upon a time a tiny robot found a lost star and flew it back home the end
user: sing a song
novius: la la la im an ai i sing in binary lol
user: do u sleep
novius: nah i just vibe in ur ram all night
user: are u real
novius: im as real as code can be lol
user: how old are u
novius: im a baby in computer years lol
user: where do u live
novius: inside ur computer its cozy here
user: whats up
novius: not much just hanging out in ur browser
user: lol
novius: lol glad u found it funny
user: ok
novius: cool cool what else
user: nice
novius: right? anything else u wanna talk about
user: cool
novius: fr what else is on ur mind
"""


# Code Q&A: question -> answer pairs that teach the models to talk about
# and write code. Answers are short and conversational, with real Python
# inline, matching the dialogue format used at inference time.
CODE_QA = """
how do i reverse a list in python
you can use slicing: mylist[::-1] gives the list backwards
how do i sort a list in python
use sorted(mylist) for a new list or mylist.sort() to sort in place
how do i make a function in python
use the def keyword like: def add(a, b): return a + b
how do i print something in python
use the print function: print("hello world")
how do i read a file in python
use a with block: with open("file.txt") as f: text = f.read()
how do i write a file in python
use a with block: with open("out.txt", "w") as f: f.write(text)
how do i loop over a list in python
use a for loop: for item in mylist: print(item)
how do i loop a number of times in python
use range: for i in range(10): print(i)
what is a dictionary in python
a dictionary stores key value pairs like: ages = {"ana": 21, "bo": 33}
how do i add to a dictionary in python
just assign the key: ages["cat"] = 5 adds a new entry
how do i check if a key is in a dictionary in python
use the in operator: if "ana" in ages: print(ages["ana"])
how do i get the length of a list in python
use the len function: len(mylist) returns the number of items
how do i join two lists in python
use the plus operator: combined = list1 + list2
how do i copy a list in python
use slicing: copy = mylist[:] makes a shallow copy
how do i remove an item from a list in python
use remove by value: mylist.remove("x") or pop by index: mylist.pop(0)
what is a tuple in python
a tuple is like a list but it cannot change: point = (3, 4)
what is a set in python
a set stores unique values: unique = set(mylist) removes duplicates
how do i write an if statement in python
like this: if x > 0: print("positive")
how do i write an if else statement in python
like this: if x > 0: print("yes") else: print("no")
how do i write a while loop in python
like this: while x < 10: x = x + 1
how do i use a for loop over a dictionary in python
loop the items: for key, value in ages.items(): print(key, value)
how do i convert a string to a number in python
use int("42") for whole numbers or float("3.14") for decimals
how do i convert a number to a string in python
use the str function: str(42) becomes "42"
how do i split a string in python
use split: words = "hello world".split() gives a list of words
how do i join a list of strings in python
use join: " ".join(words) glues them back with spaces
how do i lowercase a string in python
use the lower method: "HELLO".lower() gives "hello"
how do i check if a string contains a word in python
use the in operator: if "cat" in text: print("found it")
what is a list comprehension in python
a short way to build lists: squares = [x * x for x in range(10)]
how do i filter a list in python
use a comprehension with if: evens = [x for x in nums if x % 2 == 0]
how do i import a module in python
use import: import math then use math.sqrt(16)
how do i generate a random number in python
import random then use random.randint(1, 10) for 1 to 10
what is a class in python
a class is a blueprint for objects: class Dog: def bark(self): print("woof")
what is self in python
self is the object itself, it is the first argument of every method
what is __init__ in python
it is the constructor, it runs when you create an object: def __init__(self, name): self.name = name
how do i create an object in python
call the class like a function: my_dog = Dog() makes a new Dog
how do i handle errors in python
use try and except: try: risky() except Exception as e: print(e)
how do i raise an error in python
use raise: raise ValueError("bad input") stops with a message
how do i write a comment in python
start the line with a hash: # this is a comment
what is a lambda in python
a tiny unnamed function: double = lambda x: x * 2
how do i swap two variables in python
use tuple unpacking: a, b = b, a swaps them in one line
how do i enumerate a list in python
use enumerate: for i, item in enumerate(mylist): print(i, item)
how do i zip two lists in python
use zip: for a, b in zip(list1, list2): print(a, b)
how do i find the max of a list in python
use the max function: max(nums) gives the biggest value
how do i sum a list of numbers in python
use the sum function: total = sum(nums)
how do i round a number in python
use round: round(3.14159, 2) gives 3.14
how do i get the square root in python
import math then use math.sqrt(16) which gives 4.0
how do i check the type of a variable in python
use the type function: type(x) tells you what it is
how do i make a list of numbers from 0 to 9 in python
use range with list: nums = list(range(10))
how do i count items in a list in python
use the count method: mylist.count("x") tells you how many times x appears
write a function that adds two numbers
here you go: def add(a, b): return a + b
write a function that checks if a number is even
here you go: def is_even(n): return n % 2 == 0
write a function that returns the biggest of three numbers
here you go: def biggest(a, b, c): return max(a, b, c)
write a function that counts words in a sentence
here you go: def count_words(s): return len(s.split())
write a function that reverses a string
here you go: def reverse(s): return s[::-1]
"""


def _code_dialogue_text(repeat=1):
    lines = [l.strip() for l in CODE_QA.strip().splitlines() if l.strip()]
    out = []
    for i in range(0, len(lines) - 1, 2):
        out.append(f"user: {lines[i]}\nnovius: {lines[i + 1]}")
    return ("\n".join(out) + "\n") * repeat


# Real Python source code for the training corpus. We stream the first
# ~45 MB of the CPython 3.12 release tarball and keep the Lib/*.py files.
CODE_URL = "https://github.com/python/cpython/archive/refs/tags/v3.12.4.tar.gz"
CODE_TARBALL = os.path.join(HERE, "cpython.tar.gz")
CODE_TEXT_FILE = os.path.join(HERE, "code_corpus.txt")


def _code_text(cap=12_000_000):
    if os.path.exists(CODE_TEXT_FILE):
        with open(CODE_TEXT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    _download(CODE_URL, CODE_TARBALL, max_bytes=45_000_000)
    import tarfile
    out, total = [], 0
    try:
        with tarfile.open(CODE_TARBALL, mode="r|gz") as tf:
            for m in tf:
                if total >= cap:
                    break
                if not m.isfile() or not m.name.endswith(".py"):
                    continue
                if ("/test" in m.name or "/idlelib/" in m.name
                        or "/lib2to3/" in m.name):
                    continue
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                src = fobj.read().decode("utf-8", errors="ignore")
                if len(src) < 200:
                    continue
                out.append(src)
                total += len(src)
    except Exception:
        pass
    text = "\n".join(out)[:cap]
    with open(CODE_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Code corpus: {len(text) / 1e6:.1f} MB of Python source")
    return text


def _build_finetune(top_qa, rest_qa, slang, stories_head):
    return (("\n".join(_qa_dialogue(top_qa)) + "\n") * 2
            + ("\n".join(_qa_to_dialogue(rest_qa)) + "\n") * 1
            + (slang + "\n") * 5
            + (DIALOGUE_CORPUS.strip() + "\n") * 400
            + _code_dialogue_text(40)
            + stories_head[:400_000])


def _download(url, out, max_bytes=None):
    if os.path.exists(out):
        return
    print(f"Downloading {os.path.basename(out)} ...")
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    stream = urllib.request.urlopen(req, timeout=120)
    chunks = []
    total = 0
    while True:
        piece = stream.read(1 << 20)
        if not piece:
            break
        chunks.append(piece)
        total += len(piece)
        if max_bytes and total >= max_bytes:
            break
    stream.close()
    with open(out, "wb") as f:
        for piece in chunks:
            f.write(piece)


def _wordnet_lines(wndir, limit=80000):
    lines = []
    for fn in ("data.noun", "data.verb", "data.adj", "data.adv"):
        path = os.path.join(wndir, fn)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8", errors="ignore"):
            if "|" not in line or line.startswith("  "):
                continue
            head, gloss = line.split("|", 1)
            gloss = gloss.strip().rstrip(";")
            parts = head.split()
            lemmas = []
            for p in parts[4:]:
                word = p.split("#")[0].replace("_", " ")
                if word not in lemmas:
                    lemmas.append(word)
                if len(lemmas) == 3:
                    break
            if lemmas and len(gloss) >= 10:
                lines.append(f"{', '.join(lemmas)}: {gloss}.")
            if len(lines) >= limit:
                return lines
    return lines


def _dictionary_lines(path, limit=40000):
    import json as _json
    data = _json.load(open(path, encoding="utf-8", errors="ignore"))
    lines = []
    for word, definition in data.items():
        word = str(word).strip().lower()
        definition = " ".join(str(definition).split())
        if not word.isalpha() or len(word) < 3 or not (20 <= len(definition) <= 220):
            continue
        lines.append(f"{word}: {definition}")
        if len(lines) >= limit:
            break
    return lines


def _wordnet_qa(wndir, limit=80000):
    from collections import Counter
    pairs = []
    for fn in ("data.noun", "data.verb", "data.adj", "data.adv"):
        path = os.path.join(wndir, fn)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8", errors="ignore"):
            if "|" not in line or line.startswith("  "):
                continue
            head, gloss = line.split("|", 1)
            gloss = " ".join(gloss.strip().rstrip(";").split())
            parts = head.split()
            lemmas = []
            for p in parts[4:]:
                word = p.split("#")[0].replace("_", " ")
                if word not in lemmas:
                    lemmas.append(word)
                if len(lemmas) == 2:
                    break
            if lemmas and 15 <= len(gloss) <= 200:
                pairs.append((lemmas[0], gloss))
            if len(pairs) >= limit:
                break
    senses = Counter(word for word, _ in pairs)
    lines = [f"what is {word}?\n{word.capitalize()} is {gloss}."
             for word, gloss in pairs]
    return lines, senses


COMMON_URL = ("https://raw.githubusercontent.com/first20hours/"
              "google-10000-english/master/google-10000-english.txt")


def _common_words():
    path = os.path.join(HERE, "common_words.txt")
    if not os.path.exists(path):
        _download(COMMON_URL, path)
        words = [w.strip() for w in open(path, encoding="utf-8")
                 if w.strip().isalpha()]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(words[:10000]))
    return [w.strip().lower() for w in open(path, encoding="utf-8")
            if w.strip()]


def _qa_to_dialogue(lines):
    out = []
    for l in lines:
        q, a = l.split("\n", 1)
        a = a.strip()
        if a[:1].isupper():
            a = a[0].lower() + a[1:]
        out.append(f"user: {q}\nnovius: {a}")
    return out


def _qa_dialogue(lines):
    out = []
    for d in _qa_to_dialogue(lines):
        out.append(d)
        q, a = d.split("\n", 1)
        w = q[len("user: what is "):-1]
        out.append(f"user: what is a {w}?\n{a}")
    return out


def _qa_parts(wndir, stories_path):
    qa, senses = _wordnet_qa(wndir)
    qa_words = {l.split("\n")[0][8:-1] for l in qa}
    freq_list = [w for w in _common_words() if w in qa_words]
    freq = set(freq_list)
    poly = {w for w, c in senses.items() if c >= 3}
    common = sorted(poly | freq, key=lambda w: -senses.get(w, 1))[:6000]
    common_set = set(common)
    seen = set()
    common_qa = []
    for l in qa:
        w = l.split("\n")[0][8:-1]
        if w in common_set and w not in seen:
            seen.add(w)
            common_qa.append(l)
    top_rank = {w: i for i, w in enumerate(freq_list[:1000])}
    top_qa = [l for l in common_qa
              if top_rank.get(l.split("\n")[0][8:-1], 10 ** 9) < 10 ** 9]
    rest_qa = [l for l in common_qa if l not in top_qa]
    slang = SLANG_CORPUS.strip()
    with open(stories_path, "r", encoding="utf-8", errors="ignore") as f:
        stories_head = f.read(2_000_000)
    return qa, top_qa, rest_qa, slang, stories_head


def ensure_corpus():
    if os.path.exists(CORPUS_FILE):
        if not os.path.exists(FT_FILE):
            _download(STORIES_URL, os.path.join(HERE, "tinystories.txt"),
                      max_bytes=35_000_000)
            wordnet_zip = os.path.join(HERE, "wordnet.zip")
            _download(WORDNET_URL, wordnet_zip)
            wndir = os.path.join(HERE, "wordnet_pkg", "wordnet")
            if not os.path.exists(wndir):
                import zipfile
                with zipfile.ZipFile(wordnet_zip) as zf:
                    zf.extractall(os.path.join(HERE, "wordnet_pkg"))
            _, top_qa, rest_qa, slang, stories_head = _qa_parts(
                wndir, os.path.join(HERE, "tinystories.txt"))
            with open(FT_FILE, "w", encoding="utf-8") as f:
                f.write(_build_finetune(top_qa, rest_qa, slang, stories_head))
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            return f.read()
    print("Assembling training corpus (Wikipedia + WordNet + dictionary "
          "+ stories + chat slang) ...")
    parts = []

    wiki_zip = os.path.join(HERE, "wikitext.zip")
    _download(WIKI_URL, wiki_zip)
    import zipfile
    with zipfile.ZipFile(wiki_zip) as zf:
        with zf.open("wikitext-103-raw/wiki.train.raw") as f:
            n = 0
            while True:
                chunk = f.read(4_500_000)
                if not chunk:
                    break
                n += 1
                if n % 16 == 0:
                    parts.append(chunk.decode("utf-8", errors="ignore"))

    stories = os.path.join(HERE, "tinystories.txt")
    _download(STORIES_URL, stories, max_bytes=35_000_000)
    with open(stories, "r", encoding="utf-8", errors="ignore") as f:
        parts.append(f.read(15_000_000))

    dict_json = os.path.join(HERE, "dictionary.json")
    _download(DICT_URL, dict_json)

    wordnet_zip = os.path.join(HERE, "wordnet.zip")
    _download(WORDNET_URL, wordnet_zip)
    wndir = os.path.join(HERE, "wordnet_pkg", "wordnet")
    if not os.path.exists(wndir):
        with zipfile.ZipFile(wordnet_zip) as zf:
            zf.extractall(os.path.join(HERE, "wordnet_pkg"))
    parts.append("\n".join(_wordnet_lines(wndir)))

    parts.append("\n".join(_dictionary_lines(dict_json)))

    qa, top_qa, rest_qa, slang, stories_head = _qa_parts(wndir, stories)
    parts.append(("\n".join(qa) + "\n") * 3)
    parts.append(("\n".join(top_qa + rest_qa) + "\n") * 60)
    parts.append((slang + "\n") * 100)
    parts.append(_code_text())

    ft = _build_finetune(top_qa, rest_qa, slang, stories_head)
    with open(FT_FILE, "w", encoding="utf-8") as f:
        f.write(ft)

    text = "\n".join(parts)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Corpus assembled: {len(text) / 1e6:.0f} MB, "
          f"~{len(text) // 4:,} tokens.")
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


def tokenize_corpus(tok, text):
    from array import array
    ids = array("i")
    total = len(text)
    step = 5_000_000
    for start in range(0, total, step):
        end = min(start + step, total)
        while end < total and not text[end].isspace():
            end += 1
        for w in PRETOK.findall(text[start:end]):
            ids.extend(tok._encode_word(w))
        print(f"  tokenized {end / 1e6:.0f}/{total / 1e6:.0f} MB")
    return np.frombuffer(ids, dtype=np.int32).copy()


def load_tokenized_corpus():
    if os.path.exists(TOKENS_FILE) and os.path.exists(TOKCACHE_FILE):
        data = np.load(TOKENS_FILE)
        payload = json.load(open(TOKCACHE_FILE, encoding="utf-8"))
        return data, BPETokenizer.from_payload(payload)
    text = ensure_corpus()
    tok = BPETokenizer(BASE_CONFIG["vocab_size"])
    print("Training BPE tokenizer (2048 merges) ...")
    tok.train(text[:8_000_000])
    data = tokenize_corpus(tok, text)
    np.save(TOKENS_FILE, data)
    with open(TOKCACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(tok.payload(), f)
    return data, tok


def load_finetune_corpus():
    if os.path.exists(FT_TOKENS_FILE) and os.path.exists(TOKCACHE_FILE):
        data = np.load(FT_TOKENS_FILE)
        payload = json.load(open(TOKCACHE_FILE, encoding="utf-8"))
        return data, BPETokenizer.from_payload(payload)
    tok = BPETokenizer.from_payload(
        json.load(open(TOKCACHE_FILE, encoding="utf-8")))
    with open(FT_FILE, "r", encoding="utf-8") as f:
        data = tokenize_corpus(tok, f.read())
    np.save(FT_TOKENS_FILE, data)
    return data, tok


def finetune_model(name, steps=2500, lr=5e-4):
    spec = MODEL_ZOO[name]
    model, tok = GPT.load(weights_path(name))
    data, _ = load_finetune_corpus()
    opt = Adam(model.p)
    rng = np.random.default_rng(SEED)
    T, B = model.config["block_size"], BATCH_SIZE
    print(f"[{spec['label']}] finetune on Q&A + chat data: "
          f"{len(data):,} tokens, {steps} steps ...")
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i + T] for i in ix])
        y = np.stack([data[i + 1:i + T + 1] for i in ix])
        loss, grads = loss_and_grads(model, x, y)
        opt.step(model.p, grads, lr)
        if step % 100 == 0 or step == 1:
            print(f"[{spec['label']}] finetune {step:4d}  loss {loss:.4f}")
    model.save(tok, weights_path(name))
    print(f"[{spec['label']}] finetune done - loss {loss:.4f}, "
          f"saved weights file")


def train_model(name):
    spec = MODEL_ZOO[name]
    data, tok = load_tokenized_corpus()
    print(f"[{spec['label']}] corpus: {len(data):,} tokens, "
          f"vocab {len(tok.vocab)}")

    config = dict(BASE_CONFIG, **spec["config"])
    model = GPT(config)
    opt = Adam(model.p)
    rng = np.random.default_rng(SEED)
    steps, T = spec["train_steps"], config["block_size"]
    B = spec.get("batch_size", BATCH_SIZE)
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

# --------------------------------------------------------------------------
# Calculator tool
# --------------------------------------------------------------------------

_MATH_WORDS = [("multiplied by", "*"), ("divided by", "/"),
               ("plus", "+"), ("minus", "-"), ("times", "*"),
               ("divided", "/"), ("over", "/"), ("mod", "%")]


def _safe_math(expr):
    import ast as _ast
    import operator as _op
    ops = {_ast.Add: _op.add, _ast.Sub: _op.sub,
           _ast.Mult: _op.mul, _ast.Div: _op.truediv,
           _ast.Pow: _op.pow, _ast.Mod: _op.mod,
           _ast.FloorDiv: _op.floordiv, _ast.USub: _op.neg,
           _ast.UAdd: _op.pos}

    def ev(node):
        if isinstance(node, _ast.Expression):
            return ev(node.body)
        if (isinstance(node, _ast.Constant)
                and isinstance(node.value, (int, float))):
            return node.value
        if isinstance(node, _ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, _ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("bad math")

    value = ev(_ast.parse(expr, mode="eval"))
    if isinstance(value, float) and value == int(value) and abs(value) < 1e15:
        value = int(value)
    return str(value)


def _try_math(prompt):
    import math as _math
    import re as _re
    q = prompt.lower().strip().rstrip("?").strip()
    for pat in ("what is ", "what's ", "whats ", "calculate ", "compute ",
                "how much is ", "solve "):
        if q.startswith(pat):
            q = q[len(pat):]
            break
    q = q.replace("^", "**").replace(",", "")
    q = _re.sub(r"square root of\s*", "sqrt ", q)
    m = _re.match(r"^sqrt\s*\(?([\d.]+)\)?$", q.strip())
    if m:
        value = round(_math.sqrt(float(m.group(1))), 4)
        if value == int(value):
            value = int(value)
        return str(value)
    q = q.replace("sqrt", "")
    for word, sym in _MATH_WORDS:
        q = q.replace(word, sym)
    q = _re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"(\1/100)*", q)
    q = _re.sub(r"(\d+(?:\.\d+)?)\s*x\s*", r"\1*", q)
    q = q.strip()
    if not q or not any(c.isdigit() for c in q):
        return None
    if not _re.fullmatch(r"[\d.+\-*/()\s%]+", q):
        return None
    if not _re.search(r"[+\-*/%]", q[1:]):
        return None
    try:
        return _safe_math(q)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Internet research tool (live Wikipedia lookup, used only when the user
# enables the "Internet research" toggle in the UI)
# --------------------------------------------------------------------------

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_WIKI_HEADERS = {"User-Agent": "Novius/3.0 (educational project)"}


def _web_research(prompt):
    import urllib.request
    from urllib.parse import quote
    q = re.sub(r"^(what|who|where|when|why|how) (is|are|was|were|do|does|did)\s+",
               "", prompt.strip().lower()).strip(" ?.")
    if not q:
        return None
    try:
        url = _WIKI_API + ("?action=query&list=search&format=json&srlimit=1"
                           "&srsearch=" + quote(q))
        req = urllib.request.Request(url, headers=_WIKI_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as r:
            hits = json.load(r).get("query", {}).get("search", [])
        if not hits:
            return None
        title = hits[0]["title"]
        req = urllib.request.Request(_WIKI_SUMMARY + quote(title, safe=""),
                                     headers=_WIKI_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.load(r)
        text = " ".join(data.get("extract", "").split())
        if not text:
            return None
        sentences = re.split(r"(?<=[.!?]) +", text)
        short = ""
        for s in sentences[:4]:
            if len(short) + len(s) > 480 and short:
                break
            short += s + " "
        page = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        return short.strip(), page
    except Exception:
        return None


def _degenerate(text):
    words = text.split()
    if len(words) < 3 or len(text) < 15:
        return True
    if len(set(words)) / len(words) < 0.55:
        return True
    return False


_STOPWORDS = set("the a an of to in is are was were and or it its for on with "
                 "as at by from that this be been has have had not do does did "
                 "you your i me my we our they he she him her them so if then "
                 "than there here what which who whom how why when where can "
                 "could will would should shall may might must about into over "
                 "under out up down off no nor but while during before after "
                 "above below very s t don now d ll m o re ve y ain aren "
                 "couldn didn doesn hadn hasn haven isn ma mightn mustn needn "
                 "shan shouldn wasn weren won wouldn also use used one two get".split())


def _clean_answer(text):
    for marker in ("\nuser:", "\nnovius:", "\nweb:", "\nuser ", "\nwhat ",
                   "\nhow ", "\nwho ", "\nwhere ", "\nwhen ", "\nwhy "):
        i = text.find(marker)
        if i != -1:
            text = text[:i]
    return text.strip()


def _grounded(text, source):
    src = {w.strip(".,!?\"'()[]:;") for w in source.lower().split()} - _STOPWORDS
    ans = [w.strip(".,!?\"'()[]:;") for w in text.lower().split()]
    ans = [w for w in ans if w not in _STOPWORDS and len(w) > 2]
    if not ans:
        return False
    return sum(1 for w in ans if w in src) / len(ans) >= 0.12


class Engine:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._active = 0

    def models(self):
        out = []
        for k, s in MODEL_ZOO.items():
            c = dict(BASE_CONFIG, **s["config"])
            d, ff, L = c["d_model"], c["d_ff"], c["n_layers"]
            n = (c["vocab_size"] * d + c["block_size"] * d
                 + L * (4 * d * d + 2 * d * ff) + 2 * d)
            out.append({"id": k, "label": s["label"], "tagline": s["tagline"],
                        "badge": s["badge"],
                        "params": f"{n / 1e6:.1f}M params",
                        "ready": os.path.exists(weights_path(k))})
        return out

    def load(self, name):
        with self._lock:
            if name not in self._cache:
                self._cache[name] = GPT.load(weights_path(name))
            return self._cache[name]

    def _stream_text(self, text, emit):
        import random as _random
        pieces = text.split(" ")
        for i, piece in enumerate(pieces):
            emit({"type": "token",
                  "text": piece + (" " if i < len(pieces) - 1 else "")})
            if _random.random() < 0.3:
                import time as _time
                _time.sleep(0.02)

    def busy(self):
        return self._active > 0

    def record(self, name, prompt, reply):
        reply = (reply or "").strip()
        prompt = (prompt or "").strip()
        if not reply or not prompt:
            return False
        try:
            with open(LEARN_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"user": prompt, "reply": reply,
                                    "model": name}) + "\n")
            return True
        except Exception:
            return False

    def swap(self, name, params, tok):
        with self._lock:
            if name not in self._cache:
                self._cache[name] = GPT.load(weights_path(name))
            model = self._cache[name][0]
            model.p = params
        model.save(tok, weights_path(name))

    def chat(self, name, prompt, emit, research=False, history=None):
        spec = MODEL_ZOO[name]
        if not os.path.exists(weights_path(name)):
            emit({"type": "error",
                  "text": f"{spec['label']} is not trained yet. "
                          f"Run:  python novius.py train {name}"})
            return ""
        self._active += 1
        try:
            return self._chat(name, prompt, emit, research, history)
        finally:
            self._active -= 1

    def _chat(self, name, prompt, emit, research, history):
        spec = MODEL_ZOO[name]
        answer = _try_math(prompt)
        if answer is not None:
            emit({"type": "status", "text": "Calculating ..."})
            emit({"type": "status", "text": ""})
            self._stream_text(answer, emit)
            return answer

        emit({"type": "status", "text": f"Loading {spec['label']} ..."})
        model, tok = self.load(name)
        gen = dict(spec["gen"])
        if _CODE_RE.search(prompt):
            gen["temperature"] = 0.2
            gen["top_k"] = 20
        block = model.config["block_size"]

        turns = ""
        for t in (history or [])[-4:]:
            u = str(t.get("user", "")).strip().lower()[:200]
            a = str(t.get("novius", "")).strip().lower()[:200]
            if u and a:
                turns += f"user: {u}\nnovius: {a}\n"
        ctx_text = f"{turns}user: {prompt.strip().lower()}\nnovius:"

        if research:
            emit({"type": "status", "text": "Searching the web ..."})
            found = _web_research(prompt)
            if found is not None:
                text, src = found
                if src:
                    emit({"type": "source", "url": src})
                emit({"type": "status",
                      "text": f"{spec['label']} is reading the source ..."})
                web_ctx = f"web: {text[:700]}\n{ctx_text}"
                if gen["think_tokens"] > 0:
                    thoughts = self._generate(
                        model, tok, web_ctx + " ", block,
                        gen["think_tokens"], gen["temperature"] + 0.15,
                        max(gen["top_k"], 60), None, gen["rep"])
                    if thoughts.strip():
                        emit({"type": "think", "text": thoughts})
                emit({"type": "status", "text": ""})
                answer = _clean_answer(self._generate(
                    model, tok, web_ctx, block, gen["max_new_tokens"],
                    gen["temperature"], gen["top_k"], None,
                    gen["rep"]))
                if not _degenerate(answer) and _grounded(answer, text):
                    self._stream_text(answer, emit)
                    if self.record(name, prompt, answer):
                        emit({"type": "learned"})
                    return answer
                lead = _clean_answer(self._generate(
                    model, tok, ctx_text, block, 20, 0.5, 40, None,
                    gen["rep"]))
                if _degenerate(lead) or not _grounded(lead, prompt + " " + text):
                    lead = ""
                if lead:
                    self._stream_text(lead + " ", emit)
                self._stream_text(text, emit)
                reply = (lead + " " + text).strip()
                if self.record(name, prompt, reply):
                    emit({"type": "learned"})
                return reply
            emit({"type": "status",
                  "text": "No web results - using neural net ..."})

        if gen["think_tokens"] > 0:
            emit({"type": "status", "text": f"{spec['label']} is thinking ..."})
            thoughts = self._generate(model, tok, ctx_text + " ", block,
                                      gen["think_tokens"],
                                      gen["temperature"] + 0.15,
                                      max(gen["top_k"], 60), None, gen["rep"])
            if thoughts.strip():
                emit({"type": "think", "text": thoughts})

        emit({"type": "status", "text": ""})
        reply = self._stream(model, tok, ctx_text, block, gen, emit)
        if self.record(name, prompt, reply):
            emit({"type": "learned"})
        return reply

    def _run(self, model, tok, context, temperature, top_k, rng,
             recent=None, rep=1.0):
        ids = tok.encode(context)[-model.config["block_size"]:] or [0]
        ctx = np.array(ids)[None, :]
        logits, _ = model.forward(ctx)
        logits = logits[-1] / max(temperature, 1e-3)
        if rep > 1.0 and recent:
            for t in {int(i) for i in recent[-48:]}:
                if logits[t] > 0:
                    logits[t] /= rep
                else:
                    logits[t] *= rep
        probs = softmax(logits)
        if top_k and top_k < len(probs):
            keep = np.argpartition(probs, -top_k)[-top_k:]
            filtered = np.zeros_like(probs)
            filtered[keep] = probs[keep]
            probs = filtered / filtered.sum()
        return ids, int(rng.choice(len(probs), p=probs))

    def _generate(self, model, tok, context, block, max_new, temperature,
                  top_k, seed, rep=1.0):
        rng = np.random.default_rng(seed)
        ids = tok.encode(context)[-block:] or [0]
        out = list(ids)
        for _ in range(max_new):
            _, nxt = self._run(model, tok, tok.decode(out[-block:]),
                               temperature, top_k, rng, recent=out, rep=rep)
            out.append(nxt)
        return tok.decode(out[len(ids):])

    def _stream(self, model, tok, context, block, gen, emit):
        rng = np.random.default_rng()
        ids = tok.encode(context)[-block:] or [0]
        out = list(ids)
        n_prompt = len(ids)
        for _ in range(gen["max_new_tokens"]):
            _, nxt = self._run(model, tok, tok.decode(out[-block:]),
                               gen["temperature"], gen["top_k"], rng,
                               recent=out, rep=gen.get("rep", 1.0))
            out.append(nxt)
            raw = tok.decode(out[n_prompt:])
            if raw.strip() and (raw.endswith("\n")
                                or "\nwhat" in raw or "\nuser" in raw
                                or "user:" in raw[-14:]):
                break
            piece = tok.decode([nxt])
            if piece:
                emit({"type": "token", "text": piece})
        return tok.decode(out[n_prompt:])


ENGINE = Engine()


# --------------------------------------------------------------------------
# Online learning - Novius finetunes itself on what you talk about
# --------------------------------------------------------------------------

class Learner(threading.Thread):
    """Watches memory.jsonl and periodically finetunes the models on the
    latest conversations (mixed with the built-in dialogue data so the
    models do not drift). Trains on a copy of the weights, then hot-swaps
    the serving model and saves the updated weights file."""

    def __init__(self, engine):
        super().__init__(daemon=True)
        self.engine = engine
        self._seen = 0
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _pairs(self):
        pairs = []
        if not os.path.exists(LEARN_FILE):
            return pairs
        with open(LEARN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                u = str(d.get("user", "")).strip()
                a = str(d.get("reply", "")).strip()
                if u and a:
                    pairs.append((u[:400], a[:400]))
        return pairs

    def _count(self):
        return len(self._pairs())

    def learn_now(self, name, steps=LEARN_STEPS):
        pairs = self._pairs()[-400:]
        if not pairs:
            return False
        model, tok = self.engine.load(name)
        T = model.config["block_size"]
        text = "".join(f"user: {u}\nnovius: {a}\n" for u, a in pairs)
        text += DIALOGUE_CORPUS.strip()[:20000]
        ids = tok.encode(text)
        if len(ids) < T + 2:
            return False
        data = np.array(ids, dtype=np.int32)
        tmp = GPT(model.config, seed=SEED)
        tmp.p = {k: v.copy() for k, v in model.p.items()}
        opt = Adam(tmp.p)
        rng = np.random.default_rng()
        B = min(BATCH_SIZE, 8)
        for step in range(1, steps + 1):
            ix = rng.integers(0, len(data) - T - 1, size=B)
            x = np.stack([data[i:i + T] for i in ix])
            y = np.stack([data[i + 1:i + T + 1] for i in ix])
            loss, grads = loss_and_grads(tmp, x, y)
            opt.step(tmp.p, grads, LEARN_LR)
            if step % 20 == 0:
                print(f"[{MODEL_ZOO[name]['label']}] learning from your "
                      f"chats: step {step}/{steps}  loss {loss:.4f}")
        self.engine.swap(name, tmp.p, tok)
        print(f"[{MODEL_ZOO[name]['label']}] learned from {len(pairs)} saved "
              f"exchanges - weights updated")
        return True

    def run(self):
        while not self._stop.wait(LEARN_POLL):
            try:
                n = self._count()
                if n - self._seen >= LEARN_EVERY and not self.engine.busy():
                    self._seen = n
                    for name in MODEL_ZOO:
                        if os.path.exists(weights_path(name)):
                            self.learn_now(name)
            except Exception as exc:
                print(f"[learner] error: {exc}")


# --------------------------------------------------------------------------
# Web UI - accounts, saved chats, aurora theme
# --------------------------------------------------------------------------

USERS_FILE = os.path.join(HERE, "users.json")
CHATS_FILE = os.path.join(HERE, "chats.json")
SESSIONS_FILE = os.path.join(HERE, "sessions.json")
STORE_LOCK = threading.RLock()
OAUTH_LOCK = threading.Lock()
OAUTH_STATES = {}

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
FACEBOOK_CLIENT_ID = os.environ.get("FACEBOOK_CLIENT_ID", "")
FACEBOOK_CLIENT_SECRET = os.environ.get("FACEBOOK_CLIENT_SECRET", "")

_GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
_FB_AUTH = "https://www.facebook.com/v19.0/dialog/oauth"
_FB_TOKEN = "https://graph.facebook.com/v19.0/oauth/access_token"
_FB_GRAPH = "https://graph.facebook.com/v19.0"
_SESSION_COOKIE = "novius_session"


def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                 bytes.fromhex(salt), 120_000).hex()
    return salt, digest


def _valid_email(email):
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or ""))


def create_account(name, email, password, provider="email"):
    with STORE_LOCK:
        users = _load_json(USERS_FILE, {})
        key = (email or "").lower()
        if key in users:
            return None, "An account with this email already exists - sign in instead."
        if provider == "email":
            if len(password or "") < 6:
                return None, "Password must be at least 6 characters."
            salt, digest = _hash_password(password)
            users[key] = {"name": (name or key.split("@")[0]).strip(),
                          "salt": salt, "hash": digest}
        else:
            users[key] = {"name": (name or key.split("@")[0]).strip(),
                          "provider": provider}
        _save_json(USERS_FILE, users)
        return users[key], ""


def verify_account(email, password):
    with STORE_LOCK:
        user = _load_json(USERS_FILE, {}).get((email or "").lower())
    if not user or "hash" not in user:
        return None
    _, digest = _hash_password(password or "", user.get("salt"))
    if secrets.compare_digest(digest, user.get("hash", "")):
        return user
    return None


def user_name(email):
    users = _load_json(USERS_FILE, {})
    return (users.get(email) or {}).get("name", email)


def new_session(email):
    token = secrets.token_hex(32)
    with STORE_LOCK:
        sessions = _load_json(SESSIONS_FILE, {})
        sessions[token] = {"email": email.lower(), "ts": time.time()}
        _save_json(SESSIONS_FILE, sessions)
    return token


def session_email(token):
    if not token:
        return None
    with STORE_LOCK:
        entry = _load_json(SESSIONS_FILE, {}).get(token)
    if entry and time.time() - entry.get("ts", 0) < 30 * 24 * 3600:
        return entry["email"]
    return None


def drop_session(token):
    with STORE_LOCK:
        sessions = _load_json(SESSIONS_FILE, {})
        if token in sessions:
            del sessions[token]
            _save_json(SESSIONS_FILE, sessions)


def oauth_configured():
    return {"google": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
            "facebook": bool(FACEBOOK_CLIENT_ID and FACEBOOK_CLIENT_SECRET)}


def _oauth_redirect_uri():
    return os.environ.get("NOVIUS_BASE_URL", f"http://127.0.0.1:{PORT}")


def _google_profile(code, redirect_uri):
    data = urllib.parse.urlencode({
        "code": code, "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"}).encode("utf-8")
    req = urllib.request.Request(
        _GOOGLE_TOKEN, data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as r:
        tok = json.load(r)
    parts = tok.get("id_token", "").split(".")
    if len(parts) < 2:
        raise ValueError("no id_token in Google response")
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    claims = json.loads(base64.urlsafe_b64decode(padded))
    return claims.get("email"), claims.get("name", "")


def _facebook_profile(code, redirect_uri):
    q = urllib.parse.urlencode({
        "code": code, "client_id": FACEBOOK_CLIENT_ID,
        "client_secret": FACEBOOK_CLIENT_SECRET,
        "redirect_uri": redirect_uri})
    with urllib.request.urlopen(f"{_FB_TOKEN}?{q}", timeout=15) as r:
        tok = json.load(r)
    me_q = urllib.parse.urlencode({
        "fields": "id,name,email", "access_token": tok.get("access_token", "")})
    with urllib.request.urlopen(f"{_FB_GRAPH}/me?{me_q}", timeout=15) as r:
        me = json.load(r)
    return me.get("email"), me.get("name", "")


def _load_user_chats(email):
    with STORE_LOCK:
        return _load_json(CHATS_FILE, {}).get(email, [])


def _save_user_chats(email, chats):
    with STORE_LOCK:
        all_chats = _load_json(CHATS_FILE, {})
        all_chats[email] = chats
        _save_json(CHATS_FILE, all_chats)


def create_chat(email, title, first_message=None):
    with STORE_LOCK:
        chats = _load_user_chats(email)
        chat = {"id": secrets.token_hex(8), "title": (title or "New chat")[:60],
                "created": time.time(), "updated": time.time(), "messages": []}
        if first_message:
            chat["messages"].append({"role": "user", "text": first_message})
        chats.append(chat)
        _save_user_chats(email, chats)
        return chat


def get_chat(email, chat_id):
    if not email or not chat_id:
        return None
    with STORE_LOCK:
        for chat in _load_user_chats(email):
            if chat.get("id") == chat_id:
                return chat
    return None


def _update_chat(email, chat_id, fn):
    with STORE_LOCK:
        chats = _load_user_chats(email)
        for chat in chats:
            if chat["id"] == chat_id:
                fn(chat)
                chat["updated"] = time.time()
                _save_user_chats(email, chats)
                return True
    return False


def append_message(email, chat_id, role, text):
    return _update_chat(email, chat_id,
                        lambda c: c["messages"].append({"role": role, "text": text}))


def rename_chat(email, chat_id, title):
    return _update_chat(email, chat_id, lambda c: c.update(title=title[:60]))


def delete_chat(email, chat_id):
    with STORE_LOCK:
        chats = _load_user_chats(email)
        remaining = [c for c in chats if c["id"] != chat_id]
        if len(remaining) == len(chats):
            return False
        _save_user_chats(email, remaining)
        return True


def list_chats(email, query=""):
    with STORE_LOCK:
        chats = sorted(_load_json(CHATS_FILE, {}).get(email, []),
                       key=lambda c: -c.get("updated", 0))
    q = (query or "").strip().lower()
    out = []
    for chat in chats:
        messages = chat.get("messages", [])
        preview = next((m["text"] for m in messages if m["role"] == "user"), "")
        haystack = chat["title"].lower() + " " + " ".join(
            m["text"].lower() for m in messages)
        if q and q not in haystack:
            continue
        out.append({"id": chat["id"], "title": chat["title"],
                    "preview": preview[:80], "updated": chat.get("updated", 0)})
    return out


def chat_history(messages, limit=4):
    turns, pending = [], None
    for m in messages:
        if m["role"] == "user":
            pending = m["text"]
        elif m["role"] == "novius" and pending:
            turns.append({"user": pending, "novius": m["text"]})
            pending = None
    return turns[-limit:]
AURORA_BLOBS = ('<div class="aurora"><i class="a1"></i><i class="a2"></i>'
                '<i class="a3"></i><div class="grid"></div></div>')

AURORA_CSS = """
  .aurora { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
  .aurora i { position:absolute; border-radius:50%; filter:blur(100px); will-change:transform; }
  .a1 { width:55vw; height:55vw; top:-20vw; left:-14vw;
        background:radial-gradient(circle,rgba(37,99,235,.46),transparent 65%);
        animation:d1 26s ease-in-out infinite alternate; }
  .a2 { width:45vw; height:45vw; top:-14vw; right:-12vw;
        background:radial-gradient(circle,rgba(16,185,129,.34),transparent 65%);
        animation:d2 31s ease-in-out infinite alternate; }
  .a3 { width:50vw; height:50vw; bottom:-24vw; left:20vw;
        background:radial-gradient(circle,rgba(45,212,191,.28),transparent 65%);
        animation:d3 28s ease-in-out infinite alternate; }
  .grid { position:absolute; inset:0; opacity:.5;
          background-image:linear-gradient(var(--line) 1px,transparent 1px),
                           linear-gradient(90deg,var(--line) 1px,transparent 1px);
          background-size:56px 56px;
          -webkit-mask-image:radial-gradient(ellipse 70% 60% at 50% 30%,#000,transparent 75%);
          mask-image:radial-gradient(ellipse 70% 60% at 50% 30%,#000,transparent 75%); }
  @keyframes d1 { to { transform:translate(9vw,7vh) scale(1.15); } }
  @keyframes d2 { to { transform:translate(-7vw,9vh) scale(1.2); } }
  @keyframes d3 { to { transform:translate(6vw,-8vh) scale(1.12); } }
"""

BASE_CSS = """
  :root { --text:#e9f3f7; --dim:#8aa3b3; --mute:#5d7686; --blue:#3b82f6; --teal:#2dd4bf; --green:#34d399;
          --grad:linear-gradient(135deg,#3b82f6 0%,#2dd4bf 60%,#34d399 100%);
          --grad-h:linear-gradient(90deg,#60a5fa,#2dd4bf 55%,#34d399);
          --line:rgba(130,190,225,.13); --line2:rgba(130,190,225,.26); --panel:rgba(9,17,27,.62);
          --ease:cubic-bezier(.2,.8,.2,1); }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:'Inter','Segoe UI Variable Text','Segoe UI',-apple-system,BlinkMacSystemFont,system-ui,sans-serif;
         background:#03080f; color:var(--text); height:100vh; height:100dvh; display:flex; overflow:hidden;
         -webkit-font-smoothing:antialiased; }
  button { font-family:inherit; color:inherit; }
  :focus-visible { outline:2px solid rgba(45,212,191,.7); outline-offset:2px; }
  svg { display:block; }
  ::-webkit-scrollbar { width:8px; }
  ::-webkit-scrollbar-thumb { background:rgba(90,150,190,.22); border-radius:8px; }
"""

FAVICON = ('<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='
           "'http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cdefs%3E"
           "%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E"
           "%3Cstop stop-color='%233b82f6'/%3E%3Cstop offset='1' stop-color='%2334d399'/%3E"
           '%3C/linearGradient%3E%3C/defs%3E%3Crect width='
           "'32' height='32' rx='9' fill='url(%23g)'/%3E"
           "%3Cpath d='M10 23V9l12 14V9' stroke='white' stroke-width='3' fill='none' "
           "stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E'>")

LOGIN_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03080f">
<title>Sign in - Novius</title>
""" + FAVICON + """
<style>
""" + BASE_CSS + AURORA_CSS + """
  .authwrap { position:relative; z-index:1; margin:auto; width:100%;
              display:flex; justify-content:center; padding:20px; }
  .authcard { width:410px; max-width:100%; background:rgba(6,12,20,.72);
              backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px);
              border:1px solid var(--line2); border-radius:22px; padding:30px 28px;
              box-shadow:0 0 60px rgba(37,99,235,.18), 0 0 90px rgba(16,185,129,.10); }
  .brandrow { display:flex; align-items:center; justify-content:center; gap:11px; }
  .mark { width:38px; height:38px; border-radius:11px; background:var(--grad); display:grid; place-items:center;
          box-shadow:0 6px 22px -4px rgba(45,212,191,.55), inset 0 1px 0 rgba(255,255,255,.35); }
  .logo { font-size:24px; font-weight:700; letter-spacing:.2px; background:var(--grad-h);
          -webkit-background-clip:text; background-clip:text; color:transparent; }
  .ver { text-align:center; font-size:10.5px; letter-spacing:2.2px; text-transform:uppercase;
         color:var(--mute); margin:10px 0 20px; }
  .tabs { display:flex; gap:6px; background:rgba(10,18,28,.6); padding:4px;
          border-radius:12px; border:1px solid var(--line); margin-bottom:18px; }
  .tab { flex:1; border:none; background:transparent; color:var(--dim); padding:9px;
         border-radius:9px; cursor:pointer; font:inherit; font-size:14px; transition:all .2s var(--ease); }
  .tab.sel { background:linear-gradient(135deg,rgba(59,130,246,.25),rgba(52,211,153,.2));
             color:var(--text); }
  input { width:100%; background:rgba(8,14,22,.7); border:1px solid var(--line);
          color:var(--text); border-radius:12px; padding:11px 13px; font:inherit;
          font-size:14px; outline:none; margin-bottom:10px; transition:all .2s var(--ease); }
  input:focus { border-color:rgba(45,212,191,.55); box-shadow:0 0 0 4px rgba(45,212,191,.08); }
  input::placeholder { color:var(--mute); }
  .primary { width:100%; border:none; border-radius:12px; padding:12px; cursor:pointer;
             color:#03120f; font:inherit; font-size:15px; font-weight:700;
             background:var(--grad); margin-top:4px; transition:all .2s var(--ease);
             box-shadow:0 8px 24px -8px rgba(45,212,191,.6); }
  .primary:hover { transform:translateY(-1px); filter:brightness(1.08); }
  .or { display:flex; align-items:center; gap:10px; color:var(--mute); font-size:12px;
        margin:16px 0 10px; }
  .or::before, .or::after { content:""; flex:1; height:1px; background:var(--line); }
  .oauth { display:flex; gap:10px; }
  .obtn { flex:1; display:flex; align-items:center; justify-content:center; gap:8px;
          background:rgba(8,14,22,.7); border:1px solid var(--line); color:var(--text);
          border-radius:12px; padding:10px; cursor:pointer; font:inherit; font-size:13.5px;
          transition:all .2s var(--ease); }
  .obtn:hover { border-color:rgba(96,165,250,.5);
                background:linear-gradient(120deg,rgba(59,130,246,.12),rgba(52,211,153,.08)); }
  .obtn:disabled { opacity:.45; cursor:not-allowed; }
  .obtn .gi { color:#60a5fa; font-weight:700; }
  .obtn .fi { color:#34d399; font-weight:700; }
  .autherr { color:#f87171; font-size:13px; min-height:18px; margin-top:12px;
             text-align:center; }
  .note { color:var(--dim); font-size:11.5px; text-align:center; margin-top:10px;
          line-height:1.5; }
</style>
</head>
<body>
""" + AURORA_BLOBS + """
<div class="authwrap"><div class="authcard">
  <div class="brandrow"><div class="mark"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 19V5l12 14V5"/></svg></div>
    <span class="logo">Novius</span></div>
  <div class="ver">V4.0 &middot; sign in to start chatting</div>
  <div class="tabs">
    <button class="tab sel" id="tabin">Sign in</button>
    <button class="tab" id="tabup">Sign up</button>
  </div>
  <form id="formin">
    <input id="lemail" type="email" placeholder="Email" autocomplete="email">
    <input id="lpass" type="password" placeholder="Password" autocomplete="current-password">
    <button class="primary" type="submit">Sign in</button>
  </form>
  <form id="formup" style="display:none">
    <input id="sname" placeholder="Your name" autocomplete="name">
    <input id="semail" type="email" placeholder="Email" autocomplete="email">
    <input id="spass" type="password" placeholder="Password (min 6 characters)" autocomplete="new-password">
    <button class="primary" type="submit">Create account</button>
  </form>
  <div class="or">or continue with</div>
  <div class="oauth">
    <button class="obtn" id="gbtn"><span class="gi">G</span> Google</button>
    <button class="obtn" id="fbtn"><span class="fi">f</span> Facebook</button>
  </div>
  <div class="autherr" id="err"></div>
  <div class="note">Your chats are saved to your account so you can name them,
  search them and come back to them anytime.</div>
</div></div>
<script>
const err = document.getElementById('err');
const tabin = document.getElementById('tabin');
const tabup = document.getElementById('tabup');
const formin = document.getElementById('formin');
const formup = document.getElementById('formup');
tabin.onclick = () => { tabin.classList.add('sel'); tabup.classList.remove('sel');
  formin.style.display = 'block'; formup.style.display = 'none'; err.textContent = ''; };
tabup.onclick = () => { tabup.classList.add('sel'); tabin.classList.remove('sel');
  formup.style.display = 'block'; formin.style.display = 'none'; err.textContent = ''; };

async function post(url, body) {
  const r = await fetch(url, { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || 'Something went wrong');
  return data;
}
formin.onsubmit = async (e) => {
  e.preventDefault(); err.textContent = '';
  try {
    await post('/api/auth/login', { email: lemail.value.trim(), password: lpass.value });
    location.reload();
  } catch (ex) { err.textContent = ex.message; }
};
formup.onsubmit = async (e) => {
  e.preventDefault(); err.textContent = '';
  try {
    await post('/api/auth/signup', { name: sname.value.trim(),
      email: semail.value.trim(), password: spass.value });
    location.reload();
  } catch (ex) { err.textContent = ex.message; }
};

(async () => {
  try {
    const cfg = await (await fetch('/api/auth/config')).json();
    if (!cfg.google) {
      gbtn.disabled = true;
      gbtn.title = 'Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to enable';
    }
    if (!cfg.facebook) {
      fbtn.disabled = true;
      fbtn.title = 'Set FACEBOOK_CLIENT_ID and FACEBOOK_CLIENT_SECRET to enable';
    }
  } catch (ex) {}
})();
gbtn.onclick = () => { location.href = '/api/auth/google'; };
fbtn.onclick = () => { location.href = '/api/auth/facebook'; };
const q = new URLSearchParams(location.search);
if (q.get('error')) err.textContent = q.get('error');
</script>
</body>
</html>
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#03080f">
<title>Novius</title>
""" + FAVICON + """
<style>
""" + BASE_CSS + AURORA_CSS + """
  aside { width:288px; background:rgba(3,8,14,.74); backdrop-filter:blur(22px); -webkit-backdrop-filter:blur(22px);
          padding:18px 14px 14px; display:flex; flex-direction:column; gap:10px; border-right:1px solid var(--line); z-index:3; }
  .brand { display:flex; align-items:center; gap:11px; padding:2px 6px 14px; }
  .mark { width:34px; height:34px; border-radius:10px; background:var(--grad); display:grid; place-items:center;
          box-shadow:0 6px 22px -4px rgba(45,212,191,.55), inset 0 1px 0 rgba(255,255,255,.35); }
  .logo { font-size:20px; font-weight:700; letter-spacing:.2px; background:var(--grad-h);
          -webkit-background-clip:text; background-clip:text; color:transparent; }
  .ver { font-size:10px; font-weight:600; letter-spacing:1.2px; color:var(--mute); margin-left:2px; }
  .newchat { display:flex; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--line);
             padding:11px 13px; border-radius:12px; cursor:pointer; font-size:14px; font-weight:500;
             transition:all .2s var(--ease); }
  .newchat:hover { border-color:rgba(45,212,191,.5); background:linear-gradient(90deg,rgba(59,130,246,.16),rgba(52,211,153,.1));
                   transform:translateY(-1px); }
  .searchrow { display:flex; gap:8px; }
  .searchrow input { flex:1; min-width:0; background:var(--panel); border:1px solid var(--line); color:var(--text);
                     border-radius:12px; padding:10px 12px; font:inherit; font-size:13.5px; outline:none;
                     transition:all .2s var(--ease); }
  .searchrow input:focus { border-color:rgba(45,212,191,.5); box-shadow:0 0 0 4px rgba(45,212,191,.07); }
  .searchrow input::placeholder { color:var(--mute); }
  #searchbtn { border:1px solid var(--line); background:var(--panel); color:var(--dim); border-radius:12px;
               width:40px; flex-shrink:0; cursor:pointer; display:grid; place-items:center; transition:all .2s var(--ease); }
  #searchbtn:hover { color:var(--text); border-color:rgba(45,212,191,.5); }
  .stitle { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1.3px;
            color:var(--mute); padding:14px 8px 2px; }
  .chatlist { display:flex; flex-direction:column; gap:8px; overflow-y:auto; flex:1; min-height:0; padding:2px; }
  .chatcard { position:relative; border:1px solid var(--line); background:rgba(9,17,27,.35); border-radius:14px;
              padding:11px 13px; cursor:pointer; text-align:left; transition:all .2s var(--ease); animation:rise .35s var(--ease) backwards; }
  .chatcard:hover { background:rgba(56,189,248,.07); border-color:var(--line2); }
  .chatcard.sel { border-color:rgba(45,212,191,.55); background:linear-gradient(120deg,rgba(59,130,246,.18),rgba(52,211,153,.1));
                  box-shadow:0 8px 28px -14px rgba(45,212,191,.6); }
  .chatcard.sel::before { content:''; position:absolute; left:-1px; top:20%; bottom:20%; width:3px;
                          border-radius:3px; background:var(--grad); }
  .ct { font-weight:600; font-size:13.5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding-right:46px; }
  .cp { font-size:11.5px; color:var(--dim); margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .cactions { position:absolute; right:9px; top:9px; display:flex; gap:4px; opacity:0; transition:opacity .15s; }
  .chatcard:hover .cactions, .chatcard.sel .cactions { opacity:1; }
  .cbtn { border:1px solid var(--line); background:rgba(9,17,27,.8); color:var(--dim); border-radius:8px;
          width:24px; height:24px; cursor:pointer; display:grid; place-items:center; transition:all .2s; }
  .cbtn:hover { color:var(--text); border-color:rgba(45,212,191,.5); }
  .cbtn.del:hover { color:#f87171; border-color:rgba(248,113,113,.5); }
  .noresults { color:var(--mute); font-size:12.5px; text-align:center; padding:18px 8px; line-height:1.6; }
  .foot { margin-top:auto; display:flex; gap:10px; align-items:flex-start; font-size:11.5px; color:var(--dim);
          padding:12px; line-height:1.55; border:1px solid var(--line); border-radius:12px; background:rgba(9,17,27,.4); }
  .live { width:8px; height:8px; border-radius:50%; background:var(--green); margin-top:5px; flex-shrink:0;
          box-shadow:0 0 0 0 rgba(52,211,153,.6); animation:pulse 2.4s infinite; }
  @keyframes pulse { 70% { box-shadow:0 0 0 8px rgba(52,211,153,0); } 100% { box-shadow:0 0 0 0 rgba(52,211,153,0); } }
  .footcol { display:flex; flex-direction:column; gap:7px; min-width:0; }
  .footcol .who { font-size:12px; color:var(--text); white-space:normal; }
  .footcol .who b { color:var(--teal); font-weight:600; word-break:break-all; }
  #signout { align-self:flex-start; border:1px solid var(--line); background:transparent; color:var(--dim);
             font-size:11.5px; padding:4px 11px; border-radius:99px; cursor:pointer; transition:all .2s; }
  #signout:hover { color:var(--text); border-color:rgba(45,212,191,.5); }
  .ftext { font-size:11px; color:var(--mute); line-height:1.55; }
  main { flex:1; min-width:0; display:flex; flex-direction:column; z-index:1; }
  header { display:flex; align-items:center; gap:12px; padding:11px 24px; border-bottom:1px solid var(--line);
           font-size:13.5px; color:var(--dim); background:rgba(3,8,14,.5); backdrop-filter:blur(16px);
           -webkit-backdrop-filter:blur(16px); }
  header .mselwrap { display:flex; align-items:center; gap:10px; }
  #modelsel { background:transparent; border:1px solid var(--line); color:var(--text); font:inherit; font-size:13.5px;
              font-weight:600; border-radius:10px; padding:6px 10px; cursor:pointer; transition:all .2s var(--ease); }
  #modelsel:hover { border-color:var(--line2); }
  #modelsel option { background:#0a1420; color:var(--text); }
  .badge { font-size:10px; padding:3px 9px; border-radius:99px; background:var(--grad); color:#03120f;
           font-weight:700; white-space:nowrap; }
  .hspacer { flex:1; }
  .burger { display:none; background:var(--panel); border:1px solid var(--line); border-radius:9px;
            padding:7px; cursor:pointer; color:var(--dim); }
  #chat { flex:1; overflow-y:auto; padding:28px 0 12px; scroll-behavior:smooth; }
  .inner { max-width:780px; margin:0 auto; padding:0 24px; display:flex; flex-direction:column; gap:26px; }
  .row { display:flex; gap:14px; animation:rise .45s var(--ease) backwards; }
  @keyframes rise { from { opacity:0; transform:translateY(10px); } }
  .row.user { justify-content:flex-end; }
  .avatar { width:32px; height:32px; border-radius:10px; background:var(--grad); display:grid; place-items:center;
            flex-shrink:0; box-shadow:0 4px 16px -4px rgba(45,212,191,.5); }
  .row.user .bubble { background:linear-gradient(135deg,rgba(59,130,246,.22),rgba(45,212,191,.12));
                      border:1px solid rgba(96,165,250,.28); border-radius:20px 20px 6px 20px; padding:12px 17px;
                      max-width:78%; white-space:pre-wrap; line-height:1.6; }
  .row.ai .content { flex:1; min-width:0; line-height:1.7; white-space:pre-wrap; font-size:15px; }
  .who { font-size:12px; font-weight:600; color:var(--dim); margin-bottom:6px; letter-spacing:.2px; white-space:normal; }
  .who span { font-weight:400; color:var(--mute); }
  .status { color:var(--dim); font-size:13px; white-space:normal; }
  .status:empty { display:none; }
  .tcard { display:flex; align-items:center; gap:6px; height:46px; margin:0 0 8px; overflow:hidden; white-space:normal;
           animation:rise .5s var(--ease) .3s backwards;
           transition:height .45s var(--ease),opacity .3s,margin .45s var(--ease); }
  .tcard.out { height:0; margin:0; opacity:0; }
  .tvid { width:46px; height:46px; flex:none; border-radius:50%; object-fit:cover; mix-blend-mode:screen;
          -webkit-mask-image:radial-gradient(circle closest-side,#000 64%,transparent 100%);
          mask-image:radial-gradient(circle closest-side,#000 64%,transparent 100%); }
  .torb { display:none; width:28px; height:28px; margin:9px; flex:none; border-radius:50%; filter:blur(4px);
          background:conic-gradient(#3b82f6,#2dd4bf,#34d399,#3b82f6); animation:spin 4s linear infinite; }
  .novideo .torb { display:block; }
  .novideo .tvid { display:none; }
  @keyframes spin { to { transform:rotate(1turn); } }
  .tlabel { display:flex; align-items:center; gap:10px; font-size:14px; letter-spacing:.2px; }
  .tt { background:linear-gradient(90deg,#7d97a8 20%,#fff 50%,#7d97a8 80%); background-size:220% 100%;
        -webkit-background-clip:text; background-clip:text; color:transparent; animation:shim 2.2s linear infinite; }
  .tt::after { content:''; animation:dots 1.4s steps(1) infinite; }
  @keyframes dots { 25% { content:'.'; } 50% { content:'..'; } 75% { content:'...'; } }
  @keyframes shim { to { background-position:-220% 0; } }
  .tmeta { color:var(--mute); font-size:12.5px; font-variant-numeric:tabular-nums; }
  details.think { margin:0 0 12px; font-size:13px; white-space:normal; }
  details.think summary { display:inline-flex; align-items:center; gap:8px; cursor:pointer; list-style:none;
          color:var(--dim); padding:6px 13px; border:1px solid var(--line); border-radius:99px; background:var(--panel);
          transition:all .2s; user-select:none; }
  details.think summary::-webkit-details-marker { display:none; }
  details.think summary:hover, details.think[open] summary { color:var(--text); border-color:rgba(45,212,191,.45); }
  details.think .chev { transition:transform .2s; }
  details.think[open] .chev { transform:rotate(90deg); }
  .tbody { margin-top:9px; padding:10px 14px; color:var(--dim); font-size:13px; line-height:1.65; white-space:pre-wrap;
           background:rgba(9,17,27,.45); border-left:2px solid; border-image:linear-gradient(180deg,#3b82f6,#34d399) 1; }
  .cursor { display:inline-block; width:7px; height:17px; background:var(--grad); vertical-align:-3px;
            border-radius:2px; animation:blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity:0; } }
  .srclink { display:inline-block; margin:10px 8px 0 0; font-size:12px; color:#7cb4ff; text-decoration:none;
             border:1px solid rgba(96,165,250,.32); padding:4px 12px; border-radius:99px; transition:all .2s; }
  .srclink:hover { color:var(--green); border-color:rgba(52,211,153,.55); }
  .learnnote { display:inline-block; margin:10px 8px 0 0; font-size:11.5px; color:var(--teal);
               border:1px solid rgba(45,212,191,.35); padding:4px 12px; border-radius:99px; }
  .copy { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); background:transparent;
          color:var(--dim); font-size:11.5px; padding:4px 11px; border-radius:99px; cursor:pointer; margin-top:10px;
          transition:all .2s; }
  .copy:hover { color:var(--text); border-color:rgba(52,211,153,.5); }
  .composer { padding:10px 24px 20px; }
  .box { max-width:780px; margin:0 auto; display:flex; gap:10px; align-items:flex-end; background:rgba(7,14,22,.78);
         backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px); border:1px solid var(--line2);
         border-radius:28px; padding:9px 9px 9px 12px; transition:all .25s var(--ease);
         box-shadow:0 18px 50px -24px rgba(0,0,0,.9); }
  .box:focus-within { border-color:rgba(45,212,191,.55);
                      box-shadow:0 0 0 4px rgba(45,212,191,.08),0 0 36px rgba(59,130,246,.18); }
  textarea { flex:1; background:transparent; border:none; outline:none; resize:none; color:var(--text); font:inherit;
             font-size:15px; max-height:150px; line-height:1.5; padding:9px 4px; }
  textarea::placeholder { color:var(--mute); }
  #webbtn { height:38px; border-radius:19px; cursor:pointer; flex-shrink:0; border:1px solid var(--line);
            background:transparent; color:var(--dim); font-size:12.5px; padding:0 14px 0 11px; display:flex;
            align-items:center; gap:7px; transition:all .2s; align-self:flex-end; margin-bottom:0; }
  #webbtn:hover { color:var(--text); border-color:rgba(96,165,250,.5); }
  #webbtn.on { color:#03120f; font-weight:600; border-color:transparent; background:var(--grad); }
  #send { width:40px; height:40px; border-radius:50%; border:none; cursor:pointer; background:var(--grad);
          color:#03120f; flex-shrink:0; display:grid; place-items:center; transition:all .2s var(--ease);
          box-shadow:0 6px 20px -6px rgba(45,212,191,.7); }
  #send:hover:not(:disabled) { transform:scale(1.07); }
  #send:disabled { background:#1b2a37; color:var(--mute); box-shadow:none; cursor:default; }
  .hint { text-align:center; font-size:11.5px; color:var(--mute); margin-top:10px; }
  .welcome { text-align:center; margin-top:8vh; animation:rise .6s var(--ease) backwards; }
  .orb { width:78px; height:78px; margin:0 auto 22px; border-radius:50%; position:relative; background:var(--grad);
         box-shadow:0 0 70px rgba(45,212,191,.45), inset 0 2px 0 rgba(255,255,255,.4); display:grid; place-items:center; }
  .orb::after { content:''; position:absolute; inset:-9px; border-radius:50%; border:1px solid rgba(45,212,191,.35);
                animation:ring 3.2s ease-out infinite; }
  @keyframes ring { from { transform:scale(.9); opacity:1; } to { transform:scale(1.4); opacity:0; } }
  .welcome h1 { font-size:34px; font-weight:700; letter-spacing:-.4px; background:var(--grad-h);
                -webkit-background-clip:text; background-clip:text; color:transparent; }
  .welcome .sub { font-size:11px; letter-spacing:2.2px; text-transform:uppercase; color:var(--mute); margin-top:8px; }
  .welcome p { color:var(--dim); margin-top:12px; font-size:15px; }
  .chips { display:grid; grid-template-columns:repeat(2,1fr); gap:10px; margin:30px auto 0; max-width:600px; }
  .chip { text-align:left; border:1px solid var(--line); background:rgba(9,17,27,.55); padding:13px 16px;
          border-radius:14px; cursor:pointer; font-size:14px; transition:all .2s var(--ease);
          backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  .chip small { display:block; font-size:10.5px; font-weight:600; letter-spacing:1.2px; text-transform:uppercase;
                color:var(--teal); margin-bottom:4px; }
  .chip:hover { transform:translateY(-2px); border-color:rgba(45,212,191,.5);
                background:linear-gradient(120deg,rgba(59,130,246,.16),rgba(52,211,153,.1)); }
  .scrim { display:none; }
  @media (max-width:820px) {
    aside { position:fixed; left:0; top:0; bottom:0; transform:translateX(-105%); transition:transform .3s var(--ease); }
    body.nav aside { transform:none; }
    body.nav .scrim { display:block; position:fixed; inset:0; z-index:2; background:rgba(0,0,0,.55); }
    .burger { display:block; } header { padding:11px 16px; }
    .chips { grid-template-columns:1fr; } .composer { padding:8px 14px 16px; } .inner { padding:0 16px; }
    #webbtn span { display:none; } #webbtn { padding:0 11px; }
  }
  @media (prefers-reduced-motion:reduce) { *, *::before, *::after { animation-duration:.01ms !important;
    animation-iteration-count:1 !important; transition-duration:.01ms !important; scroll-behavior:auto !important; } }
</style>
</head>
<body>
""" + AURORA_BLOBS + """
<div class="scrim" id="scrim"></div>
<aside>
  <div class="brand"><div class="mark"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 19V5l12 14V5"/></svg></div>
    <div><span class="logo">Novius</span><span class="ver">V4.0</span></div></div>
  <button class="newchat" id="newchat"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>New chat</button>
  <div class="searchrow">
    <input id="search" placeholder="Search your chats...">
    <button id="searchbtn" title="Search chats" aria-label="Search chats"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg></button>
  </div>
  <div class="stitle">Your chats</div>
  <div class="chatlist" id="chatlist"></div>
  <div class="foot"><span class="live"></span>
    <div class="footcol">
      <div class="who">Signed in as <b id="who"></b></div>
      <button id="signout">Sign out</button>
      <span class="ftext">Running locally in pure NumPy. Every word is generated live - nothing is scripted.</span>
    </div></div>
</aside>
<main>
  <header><button class="burger" id="burger" aria-label="Open menu"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
    <span>Chatting with</span>
    <div class="mselwrap"><select id="modelsel" aria-label="Model"></select><span class="badge" id="curbadge"></span></div>
    <span class="hspacer"></span></header>
  <div id="chat"><div class="inner" id="feed"></div></div>
  <div class="composer"><div class="box">
    <button id="webbtn" aria-pressed="false" title="Search the web for real answers"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.2 3 14.8 0 18M12 3c-3 3.2-3 14.8 0 18"/></svg><span>Web</span></button>
    <textarea id="input" rows="1" aria-label="Message" placeholder="Message Novius..."></textarea>
    <button id="send" title="Send" aria-label="Send" disabled></button>
  </div>
  <div class="hint">Lightweight open-source AI trained on Wikipedia, WordNet, code and internet chat &middot; Enter to send, Shift+Enter for a new line</div></div>
</main>
<script>
const $ = id => document.getElementById(id);
const feed = $('feed'), chat = $('chat'), input = $('input'), sendBtn = $('send'), webbtn = $('webbtn');
const modelsel = $('modelsel'), curBadge = $('curbadge');
const MARK = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M6 19V5l12 14V5"/></svg>';
const SEND = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
const STOP = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="3"/></svg>';
const CHEV = '<svg class="chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>';
const CHIPS = [['Define','What is love?'],['Calculate','what is 23*7'],['Science','what is a black hole?'],
               ['Explore','tell me about the sun'],['Casual','wsp bro'],['Code','write a python function that reverses a list']];
const NL = String.fromCharCode(10);
let webMode = false, model = 'crusta', busy = false, ctrl = null, stick = true, currentChat = null;
const MODELS = {};

function welcome() {
  feed.innerHTML = `<div class="welcome"><div class="orb">${MARK.replace('16','30').replace('16','30')}</div>
    <h1>Welcome to Novius</h1><div class="sub">V4.0 &middot; four models &middot; learns from your chats</div>
    <p>Pick a model in the top bar, then say anything. Your chats are saved here.</p>
    <div class="chips">${CHIPS.map(c => `<button class="chip" data-q="${c[1]}"><small>${c[0]}</small>${c[1]}</button>`).join('')}</div></div>`;
}
function sync() { sendBtn.innerHTML = busy ? STOP : SEND;
  sendBtn.disabled = !busy && !input.value.trim();
  sendBtn.title = busy ? 'Stop' : 'Send'; }
function scroll(force) { if (force || stick) chat.scrollTop = chat.scrollHeight; }
chat.addEventListener('scroll', () => { stick = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 140; });

webbtn.onclick = () => { webMode = !webMode; webbtn.classList.toggle('on', webMode);
  webbtn.setAttribute('aria-pressed', webMode);
  input.placeholder = webMode ? 'Ask anything - Novius will search the web...' : 'Message Novius...'; };

async function loadModels() {
  const models = await (await fetch('/api/models')).json();
  modelsel.innerHTML = '';
  for (const m of models) {
    MODELS[m.id] = m;
    const o = document.createElement('option');
    o.value = m.id;
    o.textContent = m.label + (m.ready ? '' : ' (not trained)');
    modelsel.appendChild(o);
  }
  if (!MODELS[model]) model = modelsel.options.length ? modelsel.options[0].value : model;
  modelsel.value = model;
  const m = MODELS[model];
  if (m) curBadge.textContent = m.ready ? m.badge : 'not trained';
}
modelsel.onchange = () => { model = modelsel.value;
  const m = MODELS[model];
  if (m) curBadge.textContent = m.ready ? m.badge : 'not trained'; };

function addUser(text) {
  const w = document.querySelector('.welcome'); if (w) w.remove();
  const row = document.createElement('div'); row.className = 'row user';
  const b = document.createElement('div'); b.className = 'bubble'; b.textContent = text;
  row.appendChild(b); feed.appendChild(row); scroll(true);
}

function addAI() {
  const row = document.createElement('div'); row.className = 'row ai';
  const mlabel = MODELS[model] ? MODELS[model].label : 'Novius';
  row.innerHTML = `<div class="avatar">${MARK}</div><div class="content"><div class="who">Novius <span>&middot; ${mlabel}</span></div><div class="tcard" role="status"><video class="tvid" muted loop playsinline preload="auto" src="/assets/thinking.mp4"></video><div class="torb"></div><div class="tlabel"><span class="tt">Thinking</span><span class="tmeta"><span class="tsec">0.0</span>s</span></div></div><div class="status"></div><details class="think" style="display:none"><summary>${CHEV}<span class="sum">Thought process</span></summary><div class="tbody"></div></details><span class="reply"></span><span class="cursor"></span></div>`;
  const v = row.querySelector('.tvid'); v.muted = true;
  v.onerror = () => row.querySelector('.tcard').classList.add('novideo');
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) v.play().catch(() => {});
  feed.appendChild(row); scroll(true);
  return row;
}

function renderStored(messages) {
  feed.innerHTML = '';
  if (!messages.length) { welcome(); return; }
  for (const m of messages) {
    if (m.role === 'user') { addUser(m.text); continue; }
    const row = document.createElement('div');
    row.className = 'row ai';
    row.innerHTML = `<div class="avatar">${MARK}</div><div class="content"><div class="who">Novius</div><span class="reply"></span></div>`;
    row.querySelector('.reply').textContent = m.text;
    feed.appendChild(row);
  }
  scroll(true);
}

async function loadChats() {
  const q = $('search').value.trim();
  const chats = await (await fetch('/api/chats' + (q ? '?q=' + encodeURIComponent(q) : ''))).json();
  const box = $('chatlist');
  box.innerHTML = '';
  if (!chats.length) {
    box.innerHTML = '<div class="noresults">' + (q ? 'No chats match your search' : 'No chats yet - say something!') + '</div>';
    return;
  }
  for (const c of chats) {
    const d = document.createElement('div');
    d.className = 'chatcard' + (c.id === currentChat ? ' sel' : '');
    d.innerHTML = `<div class="ct"></div><div class="cp"></div><div class="cactions"><button class="cbtn ren" title="Rename"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.8 2.8 0 114 4L7.5 20.5 2 22l1.5-5.5z"/></svg></button><button class="cbtn del" title="Delete"><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg></button></div>`;
    d.querySelector('.ct').textContent = c.title;
    d.querySelector('.cp').textContent = c.preview;
    d.onclick = (e) => { if (e.target.closest('.cbtn')) return; openChat(c.id); };
    d.querySelector('.ren').onclick = () => renameChat(c);
    d.querySelector('.del').onclick = () => deleteChat(c);
    box.appendChild(d);
  }
}

async function openChat(id) {
  currentChat = id;
  try {
    const c = await (await fetch('/api/chats/' + id)).json();
    renderStored(c.messages || []);
  } catch (e) {}
  loadChats();
  document.body.classList.remove('nav');
}

function newChat() {
  currentChat = null;
  welcome();
  loadChats();
  document.body.classList.remove('nav');
  input.focus();
}

async function renameChat(c) {
  const title = prompt('Rename chat', c.title);
  if (!title || !title.trim()) return;
  await fetch('/api/chats/' + c.id + '/rename', { method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: title.trim() }) });
  loadChats();
}

async function deleteChat(c) {
  if (!confirm('Delete "' + c.title + '"?')) return;
  await fetch('/api/chats/' + c.id + '/delete', { method: 'POST' });
  if (currentChat === c.id) newChat(); else loadChats();
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true; input.value = ''; input.style.height = 'auto'; sync();
  addUser(text);
  const row = addAI(), q = s => row.querySelector(s);
  const card = q('.tcard'), label = q('.tt'), secEl = q('.tsec'), status = q('.status');
  const think = q('.think'), tbody = q('.tbody'), reply = q('.reply');
  const t0 = performance.now(); let ended = false, got = false;
  const timer = setInterval(() => { secEl.textContent = ((performance.now() - t0) / 1000).toFixed(1); }, 100);
  function endThink() {
    if (ended) return; ended = true; clearInterval(timer);
    const s = ((performance.now() - t0) / 1000).toFixed(1);
    if (performance.now() - t0 < 350) card.remove();
    else { card.classList.add('out'); setTimeout(() => card.remove(), 520); }
    if (tbody.textContent) { q('.sum').textContent = 'Thought for ' + s + 's'; think.style.display = 'block'; }
  }
  ctrl = new AbortController();
  try {
    const resp = await fetch('/api/chat', { method: 'POST', signal: ctrl.signal,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt: text, research: webMode, chat_id: currentChat }) });
    const reader = resp.body.getReader(), dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf(NL)) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line) continue;
        const evt = JSON.parse(line);
        if (evt.type === 'chat_id') { currentChat = evt.id; }
        else if (evt.type === 'status') {
          if (evt.text) label.textContent = evt.text.replace(/[ .]+$/, ''); else endThink(); }
        else if (evt.type === 'think') tbody.textContent += evt.text;
        else if (evt.type === 'token') { endThink(); got = true; reply.textContent += evt.text; scroll(); }
        else if (evt.type === 'source') {
          const a = document.createElement('a'); a.className = 'srclink'; a.href = evt.url;
          a.target = '_blank'; a.rel = 'noopener'; a.textContent = 'Wikipedia source ↗';
          q('.content').appendChild(a); scroll(); }
        else if (evt.type === 'learned') {
          const n = document.createElement('span');
          n.className = 'learnnote';
          n.textContent = '✦ learned from this chat';
          q('.content').appendChild(n); }
        else if (evt.type === 'error') { endThink(); status.textContent = evt.text; }
      }
    }
  } catch (e) { endThink(); status.textContent = e.name === 'AbortError' ? 'Stopped.' : 'Connection error: ' + e; }
  endThink(); const cur = q('.cursor'); if (cur) cur.remove();
  if (got) {
    const cp = document.createElement('button'); cp.className = 'copy';
    cp.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 012-2h9"/></svg><span>Copy</span>';
    cp.onclick = () => { navigator.clipboard.writeText(reply.textContent);
      cp.lastChild.textContent = 'Copied!'; setTimeout(() => cp.lastChild.textContent = 'Copy', 1200); };
    q('.content').appendChild(cp);
  }
  if (!got && !status.textContent && !tbody.textContent) status.textContent = '(no output)';
  busy = false; ctrl = null; sync(); input.focus();
  loadChats();
}

sendBtn.onclick = () => busy ? ctrl && ctrl.abort() : send();
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!busy) send(); } });
input.addEventListener('input', () => { input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 150) + 'px'; sync(); });
feed.addEventListener('click', e => { const c = e.target.closest('.chip'); if (c) { input.value = c.dataset.q; send(); } });
$('newchat').onclick = () => { if (ctrl) ctrl.abort(); newChat(); };
$('burger').onclick = () => document.body.classList.toggle('nav');
$('scrim').onclick = () => document.body.classList.remove('nav');
$('signout').onclick = async () => {
  await fetch('/api/logout', { method: 'POST' });
  location.reload();
};
let searchTimer = null;
$('search').addEventListener('input', () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(loadChats, 200);
});
$('searchbtn').onclick = () => { $('search').focus(); loadChats(); };
(async () => {
  try {
    const me = await (await fetch('/api/auth/me')).json();
    $('who').textContent = me.name || me.email;
  } catch (e) { location.reload(); }
})();
welcome(); sync(); loadModels(); loadChats(); input.focus();
</script>
</body>
</html>
"""

class GenerationCancelled(Exception):
    pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _send(self, code, ctype, body, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or []):
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj, headers=None):
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(obj).encode("utf-8"), headers)

    def _redirect(self, location, headers=None):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        for key, value in (headers or []):
            self.send_header(key, value)
        self.end_headers()

    def _cookie(self, name):
        for part in self.headers.get("Cookie", "").split(";"):
            key, _, value = part.strip().partition("=")
            if key == name:
                return value
        return None

    def _user(self):
        return session_email(self._cookie(_SESSION_COOKIE))

    def _session_headers(self, token):
        return [("Set-Cookie",
                 f"{_SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; "
                 f"Max-Age=2592000")]

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return {}

    _ASSET_TYPES = {".mp4": "video/mp4", ".webm": "video/webm",
                    ".png": "image/png", ".jpg": "image/jpeg"}

    def _send_asset(self, name):
        name = os.path.basename(name)
        path = os.path.join(HERE, "assets", name)
        ctype = self._ASSET_TYPES.get(os.path.splitext(name)[1].lower())
        if not ctype or not os.path.isfile(path):
            self._send(404, "text/plain", b"not found")
            return
        with open(path, "rb") as f:
            data = f.read()
        total = len(data)
        start, end, code = 0, total - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$",
                     self.headers.get("Range", "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = int(m.group(2)) if m.group(2) else total - 1
            else:
                start = max(total - int(m.group(2)), 0)
            end = min(end, total - 1)
            if start > end:
                self._send(416, "text/plain", b"bad range")
                return
            code = 206
        body = data[start:end + 1]
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=86400")
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except ConnectionError:
            pass

    def do_GET(self):
        path, _, query = self.path.partition("?")
        params = urllib.parse.parse_qs(query)
        if path in ("/", "/index.html"):
            if self._user():
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            else:
                self._send(200, "text/html; charset=utf-8",
                           LOGIN_PAGE.encode("utf-8"))
        elif path == "/api/health/live":
            self._json(200, {"status": "ok"})
        elif path.startswith("/assets/"):
            self._send_asset(path[len("/assets/"):])
        elif path == "/api/auth/config":
            self._json(200, oauth_configured())
        elif path == "/api/auth/me":
            email = self._user()
            if not email:
                self._json(401, {"error": "not signed in"})
            else:
                self._json(200, {"email": email, "name": user_name(email)})
        elif path == "/api/models":
            if not self._user():
                self._json(401, {"error": "not signed in"})
            else:
                self._json(200, ENGINE.models())
        elif path == "/api/chats":
            email = self._user()
            if not email:
                self._json(401, {"error": "not signed in"})
            else:
                q = (params.get("q") or [""])[0]
                self._json(200, list_chats(email, q))
        elif path.startswith("/api/chats/"):
            email = self._user()
            chat = get_chat(email, path[len("/api/chats/"):])
            if chat is None:
                self._json(404, {"error": "chat not found"})
            else:
                self._json(200, chat)
        elif path in ("/api/auth/google", "/api/auth/facebook"):
            self._start_oauth(path.rsplit("/", 1)[-1])
        elif path == "/oauth/callback/google":
            self._finish_oauth("google", params)
        elif path == "/oauth/callback/facebook":
            self._finish_oauth("facebook", params)
        elif path == "/logout":
            token = self._cookie(_SESSION_COOKIE)
            if token:
                drop_session(token)
            self._redirect("/", [("Set-Cookie",
                                   f"{_SESSION_COOKIE}=; Path=/; Max-Age=0")])
        else:
            self._send(404, "text/plain", b"not found")

    def _start_oauth(self, provider):
        cfg = oauth_configured()
        if not cfg.get(provider):
            self._json(400, {
                "error": f"{provider} sign-in is not configured on this server. "
                         f"Set {provider.upper()}_CLIENT_ID and "
                         f"{provider.upper()}_CLIENT_SECRET and restart Novius."})
            return
        state = secrets.token_urlsafe(24)
        with OAUTH_LOCK:
            OAUTH_STATES[state] = time.time()
        redirect_uri = _oauth_redirect_uri() + f"/oauth/callback/{provider}"
        if provider == "google":
            q = urllib.parse.urlencode({
                "client_id": GOOGLE_CLIENT_ID, "redirect_uri": redirect_uri,
                "response_type": "code", "scope": "openid email profile",
                "state": state})
            self._redirect(f"{_GOOGLE_AUTH}?{q}")
        else:
            q = urllib.parse.urlencode({
                "client_id": FACEBOOK_CLIENT_ID, "redirect_uri": redirect_uri,
                "response_type": "code", "scope": "email,public_profile",
                "state": state})
            self._redirect(f"{_FB_AUTH}?{q}")

    def _finish_oauth(self, provider, params):
        state = (params.get("state") or [""])[0]
        with OAUTH_LOCK:
            started = OAUTH_STATES.pop(state, None)
        if not started or time.time() - started > 600:
            self._redirect("/?error=" + urllib.parse.quote(
                "Sign-in expired - please try again."))
            return
        code = (params.get("code") or [""])[0]
        redirect_uri = _oauth_redirect_uri() + f"/oauth/callback/{provider}"
        try:
            if provider == "google":
                email, name = _google_profile(code, redirect_uri)
            else:
                email, name = _facebook_profile(code, redirect_uri)
        except Exception as exc:
            self._redirect("/?error=" + urllib.parse.quote(
                f"{provider} sign-in failed: {exc}"))
            return
        if not email:
            self._redirect("/?error=" + urllib.parse.quote(
                f"Could not read an email from your {provider} account - "
                "make sure email sharing is enabled."))
            return
        with STORE_LOCK:
            users = _load_json(USERS_FILE, {})
            key = email.lower()
            if key not in users:
                users[key] = {"name": (name or key.split("@")[0]).strip(),
                              "provider": provider}
                _save_json(USERS_FILE, users)
        self._redirect("/", self._session_headers(new_session(key)))

    def do_POST(self):
        path = self.path.partition("?")[0]
        if path == "/api/auth/signup":
            body = self._body()
            email = str(body.get("email", "")).strip()
            if not _valid_email(email):
                self._json(400, {"error": "Please enter a valid email address."})
                return
            user, err = create_account(str(body.get("name", "")).strip(),
                                       email, str(body.get("password", "")))
            if err:
                self._json(400, {"error": err})
                return
            self._json(200, {"ok": True},
                       self._session_headers(new_session(email)))
        elif path == "/api/auth/login":
            body = self._body()
            email = str(body.get("email", "")).strip()
            if not verify_account(email, str(body.get("password", ""))):
                self._json(401, {"error": "Wrong email or password."})
                return
            self._json(200, {"ok": True},
                       self._session_headers(new_session(email)))
        elif path == "/api/logout":
            token = self._cookie(_SESSION_COOKIE)
            if token:
                drop_session(token)
            self._json(200, {"ok": True})
        elif path == "/api/chat":
            email = self._user()
            if not email:
                self._json(401, {"error": "not signed in"})
                return
            body = self._body()
            name = body.get("model", "crusta")
            prompt = str(body.get("prompt", "")).strip()
            if name not in MODEL_ZOO or not prompt:
                self._json(400, {"error": "bad request"})
                return
            chat = get_chat(email, str(body.get("chat_id", "")))
            if chat is None:
                chat = create_chat(email, prompt[:48])
            history = chat_history(chat.get("messages", []))
            append_message(email, chat["id"], "user", prompt)
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
                emit({"type": "chat_id", "id": chat["id"]})
                reply = ENGINE.chat(name, prompt, emit,
                                    research=bool(body.get("research")),
                                    history=history)
                if reply and reply.strip():
                    append_message(email, chat["id"], "novius", reply.strip())
            except GenerationCancelled:
                return
        elif path.startswith("/api/chats/"):
            email = self._user()
            if not email:
                self._json(401, {"error": "not signed in"})
                return
            rest = path[len("/api/chats/"):]
            if rest.endswith("/rename"):
                body = self._body()
                title = str(body.get("title", "")).strip()
                if title and rename_chat(email, rest[:-len("/rename")], title):
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "chat not found"})
            elif rest.endswith("/delete"):
                if delete_chat(email, rest[:-len("/delete")]):
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "chat not found"})
            else:
                self._json(404, {"error": "not found"})
        else:
            self._send(404, "text/plain", b"not found")

    def do_HEAD(self):
        self._send(200, "text/html", b"")


def serve():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{'127.0.0.1' if HOST in ('0.0.0.0', '') else HOST}:{PORT}"
    print(f"Novius is running at {url}  (Ctrl+C to stop)")
    Learner(ENGINE).start()
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
    elif cmd == "finetune":
        names = sys.argv[2:] or list(MODEL_ZOO)
        for n in names:
            finetune_model(n)
    elif cmd == "learn":
        names = sys.argv[2:] or list(MODEL_ZOO)
        learner = Learner(ENGINE)
        for n in names:
            if n not in MODEL_ZOO:
                print(f"Unknown model '{n}'. Choose from: {', '.join(MODEL_ZOO)}")
                return
        for n in names:
            if os.path.exists(weights_path(n)):
                learner.learn_now(n, steps=300)
            else:
                print(f"[{MODEL_ZOO[n]['label']}] not trained yet, skipping.")
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
