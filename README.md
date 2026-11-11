# Novius

Novius is an open-source, lightweight AI chat app written in pure Python + NumPy —
no ML frameworks. It ships with four GPT-style transformer models trained from
scratch on modern English: Wikipedia, dictionary definitions, simple stories and
internet chat. Every reply is generated token-by-token; nothing is scripted or
predefined.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## The models

| Model | Params | Architecture | Thinking phase |
|---|---|---|---|
| **Avion V2.0** — the very fast thinker | 511k | 1 layer × 96, 4 heads | none (instant) |
| **Qlux V2.0** — quick thinker | 929k | 2 layers × 128, 4 heads | 12 thought tokens |
| **Crusta V2.0** — balanced thinker | 2.1M | 3 layers × 192, 6 heads | 24 thought tokens |
| **Hyron V2.0** — the most powerful thinker | 4.2M | 4 layers × 256, 8 heads | 48 thought tokens |

The "thinking" is real: heavier models first generate thought tokens at a higher
temperature (shown in a collapsible *Thought process* panel), then generate their
answer conditioned on both your message and their thoughts.

## What it learned

Training data (auto-downloaded on first training, ~400 MB total):

- **WikiText-103** — real English Wikipedia articles
- **WordNet** — ~80k word senses, converted into question → answer pairs
  (`what is love?` → `Love is any object of warm affection or devotion...`)
- **Dictionary** — ~40k extra word definitions
- **TinyStories** — clean, grammatical simple English
- **Chat corpus** — internet slang, broken grammar (`we was late lol`,
  `should of known better`) and everyday Q&A

After the main training pass, every model gets a final tuning pass on the
Q&A + chat data only, so it learned to *answer* questions in plain modern
English instead of just continuing stories.

## Setup

```bash
pip install numpy
```

Pre-trained weights are included in this repo (`weights_*.txt`). To train any
model from scratch yourself, delete its weights file and run the train command
below (~30 min for Avion up to ~2 hours for Hyron on a typical CPU).

## Usage

```bash
python novius.py                    # start the web UI at http://127.0.0.1:8763
python novius.py train              # (re)train all models
python novius.py train hyron        # (re)train one model
python novius.py finetune           # re-run the Q&A tuning pass
python novius.py gen hyron "hi"     # one-shot generation in the terminal
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
- **Training** — two passes: a main pass over the full ~95M-token corpus, then a
  focused finetune pass over question→answer and chat data so the models respond
  to questions instead of just continuing text.
- **UI** — dependency-free local web server (stdlib `http.server`) with a
  streaming ChatGPT-style interface (black background, blue/green auroras).
  Binds to `127.0.0.1` only.

## Notes

- These are still small models (0.5–4M params) running on a CPU: expect simple,
  sometimes wobbly English and basic knowledge, not factual assistant-level
  answers. But they answer in modern English, understand casual/slang messages,
  and get basic definitions right.
- To make a model stronger, edit its entry in `MODEL_ZOO` inside `novius.py`
  (bigger `d_model` / `n_layers` / `train_steps`) and retrain.

## License

MIT
