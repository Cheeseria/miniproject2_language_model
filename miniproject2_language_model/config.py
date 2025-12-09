from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 128
    vocab_size: int = None        # will be filled after dataset
    embed_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_iters: int = 5000
    eval_interval: int = 500
    grad_clip: float = 1.0

    # Generation
    gen_max_new_tokens: int = 300
    gen_temperature: float = 0.8
    gen_top_k: int = 40