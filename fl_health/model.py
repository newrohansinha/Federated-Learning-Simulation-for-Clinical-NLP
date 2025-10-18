import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        emb = self.emb(x)
        mask = (x!=0).unsqueeze(-1).float()
        s = (emb*mask).sum(dim=1)
        d = mask.sum(dim=1).clamp(min=1.0)
        avg = s/d
        h = self.fc1(avg)
        h = self.act(h)
        h = self.drop(h)
        out = self.fc2(h)
        return out
