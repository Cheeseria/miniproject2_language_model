import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, config, data: str):
        self.block_size = config.block_size

        chars = sorted(list(set(data)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        self.data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)
        print(f"[Dataset] Vocab size: {self.vocab_size} | Examples: {len(self)}")

    def get_vocab_size(self):
        return self.vocab_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

    def decode(self, tokens):
        return "".join(self.itos[i.item()] for i in tokens)