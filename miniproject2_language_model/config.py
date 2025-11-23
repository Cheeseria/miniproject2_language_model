from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # ────────────────────── Model Architecture ──────────────────────
    block_size: int = 128
    vocab_size: Optional[int] = None
    embed_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.1

    # ────────────────────── Training ──────────────────────
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_iters: int = 8000
    eval_interval: int = 500

    # ────────────────────── Generation / Inference ──────────────────────
    gen_max_new_tokens: int = 500
    gen_temperature: float = 0.8
    gen_top_k: int = 40