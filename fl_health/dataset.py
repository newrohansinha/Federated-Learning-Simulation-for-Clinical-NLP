import torch
from datasets import load_dataset
import re
from collections import Counter
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset

def _clean(t):
    t = t.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_vocab(texts, vocab_size):
    c = Counter()
    for t in texts:
        c.update(_clean(t).split())
    most = [w for w,_ in c.most_common(vocab_size-2)]
    stoi = {w:i+2 for i,w in enumerate(most)}
    stoi["<pad>"] = 0
    stoi["<unk>"] = 1
    return stoi

def encode(text, stoi, seq_len):
    tokens = _clean(text).split()
    ids = [stoi.get(tok,1) for tok in tokens[:seq_len]]
    if len(ids) < seq_len:
        ids += [0]*(seq_len-len(ids))
    return ids

def load_imdb_tensors(vocab_size=20000, seq_len=200):
    ds = load_dataset("imdb")
    tr_texts = [x["text"] for x in ds["train"]]
    tr_labels = [x["label"] for x in ds["train"]]
    te_texts = [x["text"] for x in ds["test"]]
    te_labels = [x["label"] for x in ds["test"]]
    stoi = build_vocab(tr_texts, vocab_size)
    Xtr = torch.tensor(np.vstack([encode(t, stoi, seq_len) for t in tr_texts]), dtype=torch.long)
    ytr = torch.tensor(np.array(tr_labels), dtype=torch.long)
    Xte = torch.tensor(np.vstack([encode(t, stoi, seq_len) for t in te_texts]), dtype=torch.long)
    yte = torch.tensor(np.array(te_labels), dtype=torch.long)
    return (TensorDataset(Xtr,ytr), TensorDataset(Xte,yte), len(stoi))

def split_clients(train_ds, test_ds, num_clients, frac_malicious=0.0, seed=42):
    g = torch.Generator().manual_seed(seed)
    tr_idx = torch.randperm(len(train_ds), generator=g).tolist()
    te_idx = torch.randperm(len(test_ds), generator=g).tolist()
    tr_splits = np.array_split(tr_idx, num_clients)
    te_splits = np.array_split(te_idx, num_clients)
    num_mal = int(num_clients*frac_malicious)
    mal_set = set(list(range(num_clients))[:num_mal])
    clients = []
    for cid in range(num_clients):
        tr_sub = Subset(train_ds, tr_splits[cid].tolist())
        te_sub = Subset(test_ds, te_splits[cid].tolist())
        clients.append((tr_sub, te_sub, cid in mal_set))
    return clients

def flip_labels(dataset):
    X = dataset.dataset.tensors[0][dataset.indices]
    y = dataset.dataset.tensors[1][dataset.indices]
    y = 1-y
    return TensorDataset(X, y)

def global_loaders(train_ds, test_ds, batch_size):
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    te = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return tr, te
