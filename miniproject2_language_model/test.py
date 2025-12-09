from config import Config
from model import GPT
from dataset import CharDataset
import requests, torch

config = Config()
text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
dataset = CharDataset(config, text)
config.vocab_size = dataset.get_vocab_size()

model = GPT(config)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

prompt = torch.tensor([[dataset.stoi[c] for c in "O God, O God! "]], dtype=torch.long)
output = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
print(dataset.decode(output[0]))