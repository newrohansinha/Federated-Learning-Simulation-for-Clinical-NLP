import torch
import numpy as np
import pandas as pd
import os
import time
import uuid

def set_seed(s):
    import random
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def to_device(batch, device):
    x,y = batch
    return x.to(device), y.to(device)

def accuracy(model, loader, device):
    model.eval()
    c = 0
    t = 0
    import torch.nn.functional as F
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb = to_device((xb,yb), device)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            c += (preds==yb).sum().item()
            t += yb.size(0)
    return c/ max(1,t)

def append_result(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)

def new_run_id():
    return str(int(time.time())) + "-" + uuid.uuid4().hex[:8]
