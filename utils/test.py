import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import spearmanr

def RankIC(df, column1='LABEL0', column2='Pred'):
    ric_values_multiindex = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0_rank'] = daily_data[column1].rank()
        daily_data['pred_rank'] = daily_data[column2].rank()
        ric, _ = spearmanr(daily_data['LABEL0_rank'], daily_data['pred_rank'])
        ric_values_multiindex.append(ric)

    if not ric_values_multiindex:
        return np.nan, np.nan

    ric = np.nanmean(ric_values_multiindex)
    std = np.nanstd(ric_values_multiindex)
    ir = ric / std if std != 0 else np.nan
    return pd.DataFrame({'RankIC': [ric], 'RankIC_IR': [ir]})

def calc_ic(pred, label):
    df = pd.DataFrame({'pred': pred, 'label': label})
    df.dropna(inplace=True)
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def Cal_IC_IR(df, column1='LABEL0', column2='Pred'):
    ic = []
    ric = []

    for date in df.index.get_level_values(0).unique():
        daily_data = df.loc[date].copy()
        daily_data['LABEL0'] = daily_data[column1]
        daily_data['pred'] = daily_data[column2]
        ic_, ric_ = calc_ic(daily_data['pred'], daily_data['LABEL0'])
        ic.append(ic_)
        ric.append(ric_)

    metrics = {
        'IC': np.nanmean(ic),
        'ICIR': np.nanmean(ic) / np.nanstd(ic),
        'RankIC': np.nanmean(ric),
        'RankICIR': np.nanmean(ric) / np.nanstd(ric)
    }

    return metrics

@torch.no_grad()
def run_inference(model, data_loader, config, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_vq = config['vqvae']
    config_pred = config['predictor']

    n_features = config_vq['num_features']
    n_prior_factors = config_vq['num_prior_factors']
    target_index = config_pred['target_day'] - 1 # ex. 5 -> 4 (start from 0)

    model.eval()
    model.to(device)
    preds = []
    reals = []

    # Per-batch IC (mirrors validation_step).
    batch_ics = []
    batch_rics = []
    test_index = data_loader.dataset.get_index()
    test_index_sorted = test_index.sortlevel(0)[0]
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Running Inference")):
        batch = batch.squeeze(0)
        batch = batch.float()
        batch = batch.to(device)

        feature = batch[:, :, 0:n_features] # (300, 20, 158)
        prior_factor = batch[:, -1, n_features : n_features+n_prior_factors] # (300, 13)
        future_returns = batch[:, -1, n_features+n_prior_factors: ] # (300, 10)
        label = future_returns[:, target_index] # (300, 1)

        # wo_prior ablation drops prior_factor.
        if hasattr(model, 'num_prior_factors') and hasattr(model, 'return_predictor') and not model.return_predictor.use_prior:
            y_pred, aux_loss = model(feature)
        else:
            y_pred, beta_p, beta_l, z_q, _ = model(feature, prior_factor)

        daily_ic, daily_ric = calc_ic(y_pred.cpu().detach().numpy(), label.cpu().detach().numpy())
        batch_ics.append(daily_ic)
        batch_rics.append(daily_ric)

        preds.append(y_pred.cpu().detach().numpy())
        reals.append(label.cpu().detach().numpy())

    batch_avg_ic = np.mean(batch_ics)
    batch_avg_ric = np.mean(batch_rics)
    print(f"Per-batch mean IC: {batch_avg_ic:.4f}")
    print(f"Per-batch mean RIC: {batch_avg_ric:.4f}")


    preds_s = pd.Series(np.concatenate(preds, axis=0).squeeze(), index=test_index_sorted)
    reals_s = pd.Series(np.concatenate(reals, axis=0).squeeze(), index=test_index_sorted)
    df = pd.DataFrame({'score': preds_s, 'label': reals_s})

    rankic = RankIC(df.dropna(), column1='label', column2='score')
    print(f"Per-date RankIC\n{rankic}")
    icir = Cal_IC_IR(df, column1='label', column2='score')
    print(f"Per-date metrics\n{icir}")

    return df, rankic, icir
