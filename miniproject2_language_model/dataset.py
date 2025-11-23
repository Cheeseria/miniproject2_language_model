import torch
from torch.utils.data import Dataset
import requests

#download
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(f"Dataset length: {len(text):,} characters")
print("First 500 characters:")
print(repr(text[:500]))

class CharDataset(Dataset):
    """
    Emits batches of characters.

    Adapted from "https://github.com/karpathy/minGPT".
    """

    def __init__(self, config, data):
        #config should have block size
        self.block_size = config.block_size


        chars = sorted(list(set(data))) # get characters from the input data
        self.vocab_size = len(chars)
        print("Found " + str(self.vocab_size) + " unique chars")
        
        self.stoi = { ch:i for i,ch in enumerate(chars) } # map characters to integer indices
        self.itos = { i:ch for i,ch in enumerate(chars) }

        #convert into tensor
        self.data = torch.tensor([self.stoi[c] for c in data], dtype=torch.long)
        print("Data tensor has shape: " + str(self.data.shape))
        print("First 50 tokens: " + str(self.data[:50].tolist()))

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.block_size
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]

        # encode every character to an integer
        # return the chunk and the shifted version as tensors
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
    




def decode(self, tokens):
    return "".join(self.itos[i.item()] for i in tokens)