import flwr as fl
import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import torch

class ClippedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, defense, max_update_norm, init_parameters, evaluate_fn=None, fit_metrics_aggregation_fn=None, eval_metrics_aggregation_fn=None):
        super().__init__(evaluate_fn=evaluate_fn, fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, eval_metrics_aggregation_fn=eval_metrics_aggregation_fn, initial_parameters=init_parameters)
        self.defense = defense
        self.max_update_norm = max_update_norm
        self.current_weights = [arr.copy() for arr in parameters_to_ndarrays(init_parameters)]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        weights_before = [w.copy() for w in self.current_weights]
        deltas = []
        sizes = []
        metrics = []
        for client, fitres in results:
            client_w = parameters_to_ndarrays(fitres.parameters)
            delta = [cw - gw for cw,gw in zip(client_w, weights_before)]
            flat = np.concatenate([d.reshape(-1) for d in delta])
            if self.defense == "clip":
                norm = np.linalg.norm(flat) + 1e-12
                if norm > self.max_update_norm:
                    flat = flat * (self.max_update_norm / norm)
            start = 0
            clipped = []
            for d in delta:
                sz = d.size
                seg = flat[start:start+sz].reshape(d.shape)
                clipped.append(seg)
                start += sz
            deltas.append(clipped)
            sizes.append(fitres.num_examples)
            metrics.append(fitres.metrics if fitres.metrics is not None else {})
        total = np.sum(sizes)
        agg = []
        for i in range(len(weights_before)):
            s = sum(d[i]* (n/total) for d,n in zip(deltas, sizes))
            agg.append(weights_before[i] + s)
        self.current_weights = [a.copy() for a in agg]
        params = ndarrays_to_parameters(agg)
        agg_metrics = {}
        if len(metrics)>0 and "epsilon" in metrics[0]:
            vals = [m.get("epsilon",0.0) for m in metrics]
            agg_metrics["epsilon_avg"] = float(np.mean(vals))
        return params, agg_metrics
