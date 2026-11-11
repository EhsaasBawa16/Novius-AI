# Novius

Novius is an open-source, lightweight AI chat app written in pure Python + NumPy —
no ML frameworks. It ships with four GPT-style transformer models trained from
scratch on English text (Tiny Shakespeare). Every reply is generated token-by-token;
nothing is scripted or predefined.

![Python](https://img.shields.io/badge/python-3.9%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## The models

| Model | Params | Architecture | Thinking phase |
|---|---|---|---|
| **Avion V1.0** — the very fast thinker | 130k | 1 layer × 64, 4 heads | none (instant) |
| **Qlux V1.0** — quick thinker | 328k | 2 layers × 96, 4 heads | 12 thought tokens |
| **Crusta V1.0** — balanced thinker | 535k | 2 layers × 128, 4 heads | 24 thought tokens |
| **Hyron V1.0** — the most powerful thinker | 1.54M | 3 layers × 192, 6 heads | 48 thought tokens |

The "thinking" is real: heavier models first generate thought tokens at a higher
temperature (shown in a collapsible *Thought process* panel), then generate their
answer conditioned on both your message and their thoughts.

## Setup

```bash
pip install numpy
```

Pre-trained weights are included in this repo (`weights_*.txt`). To train any
model from scratch yourself, delete its weights file and run the train command
below (~4 min for Avion up to ~40 min for Hyron on a typical CPU).

## Usage

```bash
python novius.py                  # start the web UI at http://127.0.0.1:8763
python novius.py train            # (re)train all models
python novius.py train hyron      # (re)train one model
python novius.py gen hyron "hi"   # one-shot generation in the terminal
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
  (512 tokens, ~2.35 chars/token).
- **Model** — decoder-only transformer: token + learned positional embeddings,
  pre-LayerNorm blocks with multi-head causal self-attention and GELU MLPs,
  softmax cross-entropy, Adam with warmup + cosine decay and gradient clipping.
  All forward/backward math is hand-written NumPy, verified against finite
  differences.
- **UI** — dependency-free local web server (stdlib `http.server`) with a
  streaming ChatGPT-style interface. Binds to `127.0.0.1` only.

## Notes

- These are small research models trained on Shakespeare: expect poetic,
  Shakespeare-flavored English rather than factual assistant answers.
- To make a model stronger, edit its entry in `MODEL_ZOO` inside `novius.py`
  (bigger `d_model` / `n_layers` / `train_steps`) and retrain.
- `corpus.txt` is downloaded automatically on first training.

## License

MIT
