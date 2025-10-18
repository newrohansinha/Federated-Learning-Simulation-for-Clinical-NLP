import flwr as fl
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from opacus import PrivacyEngine
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
from .utils import to_device, accuracy

class FLHealthClient(fl.client.NumPyClient):
    def __init__(self, model, train_dataset, test_dataset, device, dp, epsilon, delta, max_grad_norm, batch_size, epochs, lr):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.dp = dp
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.last_epsilon = None

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        privacy_engine = None
        if self.dp:
            privacy_engine = PrivacyEngine()
            sample_rate = min(1.0, self.batch_size / max(1,len(self.train_dataset)))
            self.model, opt, loader = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=opt,
                data_loader=loader,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                epochs=self.epochs,
                max_grad_norm=self.max_grad_norm,
            )
        for _ in range(self.epochs):
            for xb,yb in loader:
                xb,yb = to_device((xb,yb), self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
        if privacy_engine is not None:
            try:
                self.last_epsilon = privacy_engine.get_epsilon(self.delta)
            except Exception:
                self.last_epsilon = self.epsilon
        else:
            self.last_epsilon = None
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        num_examples = len(self.train_dataset)
        metrics = {}
        if self.last_epsilon is not None:
            metrics["epsilon"] = float(self.last_epsilon)
        return params, num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device)
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.model.eval()
        import torch.nn.functional as F
        loss_sum = 0.0
        n = 0
        with torch.no_grad():
            for xb,yb in loader:
                xb,yb = to_device((xb,yb), self.device)
                logits = self.model(xb)
                loss = F.cross_entropy(logits, yb, reduction="sum")
                loss_sum += loss.item()
                n += yb.size(0)
        acc = accuracy(self.model, loader, self.device)
        return float(loss_sum / max(1,n)), n, {"accuracy": float(acc)}
