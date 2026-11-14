# Novius

Novius is an open-source, lightweight AI chat app written in pure Python + NumPy —
no ML frameworks. It ships with four GPT-style transformer models trained from
scratch on modern English (Wikipedia, dictionary definitions, simple stories and
internet chat).

Novius answers with a **hybrid brain**: the neural models generate replies
token-by-token, boosted by a built-in **knowledge base** of everyday knowledge,
code Q&A and small talk (see below), plus an exact **calculator** for arithmetic
(`what is 23*7`, `sqrt 144`, `15% of 80`, `2^10`) — something tiny networks can
never learn reliably.

There is also an optional **Internet research** mode: click the 🌐 **Web**
toggle in the chat box and Novius will search Wikipedia for factual questions,
then *talk* an answer from the real retrieved text (with a source link) instead
of just dumping the research at you. Casual messages stay conversational even
with Web mode on — no random definition dumps.

## Smart replies (V4.0)

- **It understands what you say** — gibberish like `asdfghjkl` or `jjjj kkkk`
  gets an honest "I couldn't understand that" instead of a made-up answer.
  Real slang (`wsp bro`, `hiii`, `ngl`) still works.
- **Stronger grammar** — a polish layer fixes contractions (`im` → I'm,
  `dont` → don't, `u` → you), capitalization, spacing and punctuation on every
  generated reply, and repeated-word stutters are removed.
- **Basic knowledge of many things** — space, science, the human body, math,
  computers, geography, everyday life and more are answered instantly and
  correctly from the built-in knowledge base; everything else goes to the
  neural net.
- **Honest fallback** — if the network's answer comes out garbled, Novius says
  it isn't sure and suggests Web mode instead of showing broken text.
- **Smarter web mode** — only factual questions ("what is…", "who was…",
  "tell me about…") trigger a live Wikipedia lookup, the search query is
  cleaned up first ("tell me about the sun" searches *sun*, not the whole
  sentence), and answers lead with a natural line like "Here's what I found on
  the web."

**Novius learns from you.** Every chat exchange is saved to `memory.jsonl`
and a background learner periodically finetunes the models on what you
talked about — the more you chat, the better it gets. You'll see a small
"✦ learned from this chat" note when an exchange is added to its memory.

## Accounts & your chats

Sign-in is required: the app opens on a **sign-in / sign-up** page with
**email + password**, or **Google** and **Facebook** sign-in. Accounts,
sessions and every named chat are stored locally in JSON files next to
`novius.py` (`users.json`, `sessions.json`, `chats.json`) — nothing leaves
your machine.

- **Sidebar** — your named chats in a scrollable list with a live **search**
  (searches titles *and* all messages), rename (✎) and delete (✕) on hover.
- **Top bar** — the model picker lives in the header now; pick Avion, Qlux,
  Crusta or Hyron for any chat.
- **Model memory** — multi-turn context is rebuilt from the saved chat, so
  Novius remembers the conversation even after a refresh.

Google / Facebook sign-in uses standard OAuth2. To enable them, set the
environment variables and restart:

```bash
export GOOGLE_CLIENT_ID=...      export GOOGLE_CLIENT_SECRET=...
export FACEBOOK_CLIENT_ID=...    export FACEBOOK_CLIENT_SECRET=...
# redirect URI to register: http://127.0.0.1:8763/oauth/callback/<provider>
```

Without credentials, email sign-up still works and the Google/Facebook
buttons show a hint instead of failing silently.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## The models (V3.0)

| Model | Params | Architecture | Thinking phase |
|---|---|---|---|
| **Avion V3.0** — the very fast thinker | 234k | 2 layers × 64, 4 heads | none (instant) |
| **Qlux V3.0** — quick thinker | 664k | 2 layers × 128, 4 heads | 24 thought tokens |
| **Crusta V3.0** — balanced thinker | 1.7M | 3 layers × 192, 6 heads | 48 thought tokens |
| **Hyron V3.0** — the most powerful thinker | 5.3M | 6 layers × 256, 8 heads | 96 thought tokens |

The "thinking" is real: heavier models first generate thought tokens at a higher
temperature (shown in a collapsible *Thought process* panel), then generate their
answer. In V4.0 every thinker thinks twice as long as before. Thoughts are
display-only — they are never fed back into the answer context, which keeps
answers clean.

## What it learned

Training data (auto-downloaded on first training, ~400 MB total):

- **WikiText-103** — real English Wikipedia articles (46M-token corpus)
- **WordNet** — ~80k word senses, converted into question → answer pairs
  (`what is love?` → `love is a strong positive emotion of regard and affection...`)
- **Dictionary** — ~40k extra word definitions
- **TinyStories** — clean, grammatical simple English
- **Chat corpus** — internet slang, broken grammar (`we was late lol`,
  `should of known better`) and everyday Q&A
- **CPython source code** — real Python modules, so the models can talk
  about and write code
- **Code Q&A** — ~50 question → answer pairs like `how do i reverse a list
  in python` → `you can use slicing: mylist[::-1]`

Training is a **two-pass** process:

1. **Main pass** — every model reads the full ~46M-token corpus (wiki + stories +
   dictionary Q&A + slang) so it learns English itself: grammar, words, facts.
2. **Finetune pass** — a focused pass on dialogue-formatted data
   (`user: ...` / `novius: ...`) built from the Q&A pairs and chat corpus, so the
   models *answer* questions in plain modern English instead of just continuing
   stories. Finetune data uses the exact same format the app uses at inference
   time.

## Setup

```bash
pip install numpy
```

Pre-trained weights are included in this repo (`weights_*.txt`). To train any
model from scratch yourself, delete its weights file and run the train command
below (~30 min for Avion up to a few hours for Hyron on a typical CPU). If you
trained before V4.0, delete `corpus.txt` once so the code data is added to the
main corpus.

## Usage

```bash
python novius.py                    # start the web UI at http://127.0.0.1:8763
python novius.py train              # (re)train all models
python novius.py train hyron        # (re)train one model
python novius.py finetune           # re-run the Q&A tuning pass
python novius.py gen hyron "hi"     # one-shot generation in the terminal
python novius.py learn              # finetune all models on your saved chats
python novius.py learn hyron        # finetune one model on your saved chats
```

Or use the engine from Python:

```python
from novius import ENGINE

events = []
ENGINE.chat("hyron", "What is love?", events.append)
for e in events:
    if e["type"] == "think":
        print("THINKING:", e["text"])
    elif e["type"] == "token":
        print(e["text"], end="")
```

## How it works

- **Tokenizer** — a byte-pair-encoding (BPE) tokenizer learned from the corpus
  (2048 tokens, ~2.6 chars/token).
- **Model** — decoder-only transformer: token + learned positional embeddings,
  pre-LayerNorm blocks with multi-head causal self-attention and GELU MLPs,
  softmax cross-entropy, Adam with warmup + cosine decay and gradient clipping.
  All forward/backward math is hand-written NumPy, verified against finite
  differences.
- **Training** — main pass over the full corpus (which now includes real
  Python source code), then a dialogue-format finetune pass (see above).
- **Calculator tool** — a small AST-whitelisted evaluator handles arithmetic
  prompts before generation; everything else goes to the neural models.
- **Internet research tool** — optional toggle; searches the Wikipedia API
  (stdlib `urllib`, no API key), then the model *reads* the retrieved text
  and talks an answer from it, with a source link. If the grounded answer
  comes out garbled (tiny models!), it falls back to a short neural
  lead-in plus the retrieved facts. Falls back to the neural models when
  offline or no results.
- **Online learning** — every exchange is saved to `memory.jsonl`; a
  background learner finetunes the models on the latest ~400 exchanges
  (mixed with built-in dialogue data to prevent drift) every 6 new chats,
  hot-swaps the serving weights, and saves the updated weights file. Ask a
  code question and the model also switches to a colder, more focused
  sampling profile.
- **UI** — dependency-free local web server (stdlib `http.server`) with a
  streaming ChatGPT-style interface: black background, animated blue/green
  auroras, glassmorphism panels, suggestion chips, copy buttons, thinking
  phases, and a mobile layout. Binds to `127.0.0.1` only.

## Notes

- These are still small models (0.2–5M params) running on a CPU: expect simple,
  sometimes wobbly English and best-effort definitions — not factual
  assistant-level answers. They chat naturally, understand casual/slang
  messages, and get common word definitions right most of the time.
- To make a model stronger, edit its entry in `MODEL_ZOO` inside `novius.py`
  (bigger `d_model` / `n_layers` / `train_steps`) and retrain.

## License

MIT
