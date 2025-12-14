import torch
from torch.utils.data import DataLoader
import requests
import time

if __name__ == '__main__':

    from config import Config
    from dataset import CharDataset
    from model import GPT

    # Config
    config = Config()
    config.batch_size = 32
    config.max_iters = 10000 # 1k is enough for actual works but we want cinema, we want a masterpiece
    config.eval_interval = 500

    # Data
    print("Loading TinyShakespeare...")
    text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text

    dataset = CharDataset(config, text)
    config.vocab_size = dataset.get_vocab_size()

    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True
    )

    data_iter = iter(train_loader)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Training started...\n")

    model.train()
    for step in range(config.max_iters + 1):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)

        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % config.eval_interval == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.4f}")

            model.eval()
            with torch.no_grad():
                prompt = torch.tensor([[dataset.stoi.get(c, 0) for c in "O God, O God! "]], device=device)
                gen = model.generate(prompt, max_new_tokens=300, temperature=0.8, top_k=40)
                print("\n" + dataset.decode(gen[0].cpu()))
                print("\n" + "—" * 80)
            model.train()

    print("Training finished!")