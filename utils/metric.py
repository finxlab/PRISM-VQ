import wandb
import pandas as pd
import numpy as np
import hashlib

def log_metrics_as_bar_chart(metrics_dict, model_name=None):
    """
    Log a bar chart of metrics to wandb.

    metrics_dict: e.g. {'IC': 0.12, 'IC_IR': 0.08, 'RankIC': 0.10, 'RankIC_IR': 0.07}
    model_name: optional identifier used to disambiguate the chart and log key.
    """

    data = []
    for key, value in metrics_dict.items():
        data.append([key, float(value)])

    table = wandb.Table(data=data, columns=["Metric", "Value"])

    if model_name:
        chart_title = f"Metrics Bar Chart - {model_name}"
        # W&B builds artifact names like run-<id>-<key>; shorten the key with a hash
        # to stay under the 128-char limit.
        short_token = hashlib.md5(model_name.encode("utf-8")).hexdigest()[:8]
        log_key = f"mbar_{short_token}"
    else:
        chart_title = "Metrics Bar Chart"
        log_key = "mbar"

    bar_chart = wandb.plot.bar(
        table,
        "Metric",
        "Value",
        title=chart_title
    )

    wandb.log({log_key: bar_chart})


def calculate_table_metrics(series, period, name, target_return=0):

    if period is not None:
        if type(period) == int:
            series = series[series.index.year == int(period)].copy()
            # series['return'] = series['return'] / series['return'].iloc[0]  
        elif type(period) == list:
            series = series.loc[period[0]:period[1]].copy()
    try:  
        daily_log_returns = series['return']
        cum_return = series['return'].cumsum()
    except:
        daily_log_returns = series
        cum_return = series.cumsum()
    normal_cum_return = np.exp(cum_return)

    # MDD over cumulative simple returns.
    max_cumulative_returns = normal_cum_return.cummax()
    drawdown = (normal_cum_return - max_cumulative_returns) / (max_cumulative_returns + 1e-9)
    mdd = drawdown.min()

    annual_return = daily_log_returns.mean() * 252
    annual_std = daily_log_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_std

    # Sortino Ratio
    # Calculate downside deviation
    downside_returns = daily_log_returns[daily_log_returns < target_return]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annual_return - target_return) / downside_std if downside_std != 0 else np.nan
    
    # Calmar Ratio
    calmar_ratio = annual_return / abs(mdd) if mdd != 0 else np.nan

    # Turnover
    turnover = series['turnover'].mean()
    turnover = round(turnover, 4)
    
    result = {
        'Annualized Return': round(annual_return, 4),
        'Annual Std': round(annual_std, 4),
        'MDD': round(mdd, 4),
        'Sharpe Ratio': round(sharpe_ratio, 4),
        'Sortino Ratio': round(sortino_ratio, 4),
        'Calmar Ratio': round(calmar_ratio, 4),
        'Cumulative Returns': round(cum_return.iloc[-1], 4),
        'Turnover': turnover
    }

    return pd.DataFrame.from_dict(result, orient='index', columns=[f'{name}'])