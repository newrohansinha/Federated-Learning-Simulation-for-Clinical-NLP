import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="FL-Health Dashboard", layout="wide")
st.title("FL-Health Dashboard")

path = st.text_input("Metrics CSV path", "results/metrics.csv")
if not os.path.exists(path):
    st.stop()
df = pd.read_csv(path)
fed = df[df["mode"]=="federated"]
cen = df[df["mode"]=="centralized"]

st.subheader("Global Accuracy Over Rounds")
runs = sorted(fed["run_id"].unique().tolist())
sel = st.multiselect("Select run_id", runs, runs[-1:] if runs else [])
if sel:
    plot_df = fed[fed["run_id"].isin(sel)][["run_id","round","accuracy"]]
    plot_df = plot_df.pivot_table(index="round", columns="run_id", values="accuracy", aggfunc="mean")
    st.line_chart(plot_df)

st.subheader("Attack vs No Attack")
if len(fed)>0:
    base = fed.copy()
    opts = {}
    cols = ["dp","epsilon","defense","max_update_norm","num_clients","num_rounds","local_epochs","batch_size","lr"]
    for c in cols:
        vals = sorted([v for v in base[c].dropna().unique().tolist()])
        if len(vals)>0:
            opts[c] = st.selectbox(c, vals, index=0)
    if opts:
        mask = np.ones(len(base)).astype(bool)
        for c in cols:
            mask &= base[c]==opts[c]
        sub = base[mask]
        if len(sub)>0:
            grp = sub.groupby(["attack","round"])["accuracy"].mean().reset_index()
            p0 = grp[grp["attack"]==0].set_index("round")["accuracy"]
            p1 = grp[grp["attack"]==1].set_index("round")["accuracy"]
            comp = pd.concat([p0.rename("no_attack"), p1.rename("attack")], axis=1)
            st.line_chart(comp)

st.subheader("Accuracy vs Epsilon")
g = fed.sort_values(["run_id","round"]).groupby("run_id").tail(1)
g = g[g["dp"]==1]
if len(g)>0:
    t = g[["epsilon","accuracy","attack","defense"]].dropna()
    st.scatter_chart(t, x="epsilon", y="accuracy", color="attack", size=None)
