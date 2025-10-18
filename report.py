import argparse
import pandas as pd
import numpy as np
import os
import time

p = argparse.ArgumentParser()
p.add_argument("--results_path", type=str, default="results/metrics.csv")
p.add_argument("--out_path", type=str, default="results/report.md")
args = p.parse_args()

if not os.path.exists(args.results_path):
    raise SystemExit(1)
df = pd.read_csv(args.results_path)
cen = df[df["mode"]=="centralized"]
fed = df[df["mode"]=="federated"]
last = fed.sort_values(["run_id","round"]).groupby("run_id").tail(1)
lines = []
lines.append("# FL-Health Report")
if len(cen)>0:
    cacc = float(cen.tail(1)["accuracy"].iloc[0])
    lines.append(f"Centralized accuracy: {cacc:.4f}")
for _,r in last.iterrows():
    rid = r["run_id"]
    acc = r["accuracy"]
    dp = r["dp"]
    eps = r["epsilon"]
    atk = r["attack"]
    dfn = r["defense"]
    mon = r["max_update_norm"]
    lines.append(f"Run {rid}: accuracy {acc:.4f}, dp {dp}, epsilon {eps}, attack {atk}, defense {dfn}, max_update_norm {mon}")
    if len(cen)>0:
        drop = (cacc - acc)/max(1e-9,cacc)*100.0
        lines.append(f"Drop vs centralized: {drop:.2f}%")
groups = last.copy()
groups["eps_val"] = groups["epsilon"].fillna(0.0)
trade = groups.groupby("eps_val")["accuracy"].mean().reset_index().sort_values("eps_val")
lines.append("## Accuracy-Privacy Trade-off")
for _,row in trade.iterrows():
    lines.append(f"Epsilon {row['eps_val']}: mean accuracy {row['accuracy']:.4f}")
atk_cmp = last.groupby("attack")["accuracy"].mean()
if 0 in atk_cmp.index and 1 in atk_cmp.index:
    drop_atk = (atk_cmp[0]-atk_cmp[1])/max(1e-9,atk_cmp[0])*100.0
    lines.append(f"Attack impact: accuracy drop {drop_atk:.2f}%")
with open(args.out_path, "w") as f:
    f.write("\n".join(lines))
