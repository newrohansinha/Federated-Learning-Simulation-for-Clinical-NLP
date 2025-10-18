import argparse
import time
import torch
import flwr as fl
from flwr.simulation import start_simulation
from flwr.common import ndarrays_to_parameters
from torch.utils.data import DataLoader
from fl_health.dataset import load_imdb_tensors, split_clients, flip_labels
from fl_health.model import TextClassifier
from fl_health.client import FLHealthClient
from fl_health.strategy import ClippedFedAvg
from fl_health.utils import set_seed, append_result, new_run_id, accuracy
import numpy as np
import pandas as pd
import os

def client_fn_builder(clients_data, device, dp, epsilon, delta, max_grad_norm, batch_size, epochs, lr):
    def client_fn(cid):
        idx = int(cid)
        tr, te, mal = clients_data[idx]
        model = TextClassifier(vocab_size).to(device)
        if mal:
            tr_f = flip_labels(tr)
        else:
            tr_f = tr
        return FLHealthClient(model, tr_f, te, device, dp==1, epsilon, delta, max_grad_norm, batch_size, epochs, lr)
    return client_fn

def evaluate_fn_builder(global_test_loader, device, results_path, run_id, cfg):
    def evaluate(server_round, parameters, config):
        model = TextClassifier(vocab_size).to(device)
        params_dict = zip(model.state_dict().keys(), [torch.tensor(p) for p in parameters])
        state_dict = {k: v for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        acc = accuracy(model, global_test_loader, device)
        row = {
            "run_id": run_id,
            "mode": "federated",
            "round": int(server_round),
            "accuracy": float(acc),
            "dp": int(cfg["dp"]),
            "epsilon": float(cfg["epsilon"]) if cfg["dp"]==1 else None,
            "attack": int(cfg["attack"]),
            "defense": cfg["defense"],
            "max_update_norm": float(cfg["max_update_norm"]) if cfg["defense"]=="clip" else None,
            "num_clients": int(cfg["num_clients"]),
            "num_rounds": int(cfg["num_rounds"]),
            "local_epochs": int(cfg["local_epochs"]),
            "batch_size": int(cfg["batch_size"]),
            "lr": float(cfg["lr"]),
            "timestamp": int(time.time())
        }
        append_result(results_path, row)
        return 0.0, {"accuracy": float(acc)}
    return evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_rounds", type=int, default=5)
parser.add_argument("--local_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--vocab_size", type=int, default=20000)
parser.add_argument("--seq_len", type=int, default=200)
parser.add_argument("--attack", type=int, default=0)
parser.add_argument("--frac_malicious", type=float, default=0.2)
parser.add_argument("--dp", type=int, default=0)
parser.add_argument("--epsilon", type=float, default=4.0)
parser.add_argument("--delta", type=float, default=1e-5)
parser.add_argument("--defense", type=str, default="none", choices=["none","clip"])
parser.add_argument("--max_update_norm", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--results_path", type=str, default="results/metrics.csv")
args = parser.parse_args()

set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ds, test_ds, vocab = load_imdb_tensors(args.vocab_size, args.seq_len)
global_test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, drop_last=False)
clients = split_clients(train_ds, test_ds, args.num_clients, args.frac_malicious if args.attack==1 else 0.0, seed=args.seed)
vocab_size = vocab
run_id = new_run_id()
cfg = {
    "dp": args.dp,
    "epsilon": args.epsilon,
    "attack": args.attack,
    "defense": args.defense,
    "max_update_norm": args.max_update_norm,
    "num_clients": args.num_clients,
    "num_rounds": args.num_rounds,
    "local_epochs": args.local_epochs,
    "batch_size": args.batch_size,
    "lr": args.lr
}
client_fn = client_fn_builder(clients, device, args.dp, args.epsilon, args.delta, args.max_update_norm, args.batch_size, args.local_epochs, args.lr)
model_init = TextClassifier(vocab_size)
init_params = ndarrays_to_parameters([val.cpu().detach().numpy() for _, val in model_init.state_dict().items()])
evaluate_fn = evaluate_fn_builder(global_test_loader, device, args.results_path, run_id, cfg)
strategy = ClippedFedAvg(defense=args.defense, max_update_norm=args.max_update_norm, init_parameters=init_params, evaluate_fn=evaluate_fn)
start_simulation(client_fn=client_fn, num_clients=args.num_clients, config=fl.simulation.SimulationConfig(num_rounds=args.num_rounds), strategy=strategy, client_resources={"num_cpus": 0})
