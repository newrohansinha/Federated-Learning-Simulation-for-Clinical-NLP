import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
from fl_health.dataset import load_imdb_tensors, global_loaders
from fl_health.model import TextClassifier
from fl_health.utils import set_seed, accuracy, append_result, new_run_id
from opacus import PrivacyEngine
import os
import time

def train_epoch(model, loader, opt, device):
    model.train()
    for xb,yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--vocab_size", type=int, default=20000)
    p.add_argument("--seq_len", type=int, default=200)
    p.add_argument("--dp", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=4.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results_path", type=str, default="results/metrics.csv")
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, te, vocab = load_imdb_tensors(args.vocab_size, args.seq_len)
    tr_loader, te_loader = global_loaders(tr, te, args.batch_size)
    model = TextClassifier(vocab).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    privacy_engine = None
    if args.dp==1:
        privacy_engine = PrivacyEngine()
        sample_rate = min(1.0, args.batch_size/max(1,len(tr)))
        model, opt, tr_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=opt,
            data_loader=tr_loader,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            epochs=args.epochs,
            max_grad_norm=1.0,
        )
    for _ in range(args.epochs):
        train_epoch(model, tr_loader, opt, device)
    acc = accuracy(model, te_loader, device)
    eps_val = None
    if privacy_engine is not None:
        try:
            eps_val = privacy_engine.get_epsilon(args.delta)
        except Exception:
            eps_val = args.epsilon
    rid = new_run_id()
    row = {
        "run_id": rid,
        "mode": "centralized",
        "round": -1,
        "accuracy": float(acc),
        "dp": int(args.dp),
        "epsilon": float(eps_val) if eps_val is not None else None,
        "attack": 0,
        "defense": "none",
        "max_update_norm": None,
        "num_clients": None,
        "num_rounds": None,
        "local_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "timestamp": int(time.time())
    }
    append_result(args.results_path, row)

if __name__ == "__main__":
    main()
