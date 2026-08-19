from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np

def corr_cluster_order(returns, method='average', eps=1e-6, min_periods=10):
    returns = returns.detach().cpu().numpy()
    returns = returns[...,0]
    df = pd.DataFrame(returns)
    df = df.ffill(axis=1).bfill(axis=1).fillna(0.0)

    var = df.var(axis=1)
    keep = var > 0
    # if (~keep).any():
    #     print(f"[info] drop {(~keep).sum()} zero‑variance stocks")

    df_kept = df.loc[keep]                              # (B', T)
    # Noise shape must match df_kept (after filtering).
    df_kept += np.random.normal(0, eps, df_kept.shape)

    corr = df_kept.T.corr(min_periods=min_periods).to_numpy()
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    dist = np.sqrt(2 * (1 - corr)).astype(np.float32)
    Z = linkage(squareform(dist, checks=False), method=method)
    order_kept = leaves_list(Z)

    full_order = np.concatenate([np.where(keep)[0][order_kept],
                                 np.where(~keep)[0]])
    return full_order