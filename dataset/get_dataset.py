import pickle
import torch
import pandas as pd
import numpy as np
import pprint as pp
import os
import yaml
from argparse import ArgumentParser
from qlib.data.dataset.handler import DataHandlerLP
from qlib.tests.data import GetData
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.workflow import R
from qlib.utils import init_instance_by_config
from qlib.constant import REG_CN, REG_US
import qlib
from qlib.data import D
import sys
from pathlib import Path
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.processor import RobustZScoreNorm, Fillna, DropnaLabel, CSRankNorm
from qlib.data.dataset import TSDatasetH, DataHandlerLP
from qlib.data.dataset.processor import Processor
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from qlib.data.dataset.utils import get_level_index

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))


def get_root_dir():
    return Path(__file__).parent.parent


def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


class GlobalFactorMerger(Processor):
    """
    Append `factor_mat` (rows: datetime, columns: factor) to a
    (datetime, instrument) MultiIndex DataFrame by broadcasting along instruments.
    """

    def __init__(self, factor_mat: pd.DataFrame, fields_group: str = "prior"):
        self.factor_mat = factor_mat
        self.fg = fields_group

    def __call__(self, df: pd.DataFrame):
        # 1. Broadcast factor rows across instruments per datetime.
        dt_index = df.index.get_level_values("datetime")
        fac_block = self.factor_mat.loc[dt_index].values
        repeat = int(len(df) / len(dt_index))
        fac_block = np.repeat(fac_block, repeat, axis=0)

        # 2. Wrap with the target MultiIndex columns.
        fac_df = pd.DataFrame(
            fac_block,
            index=df.index,
            columns=pd.MultiIndex.from_tuples(
                [(self.fg, c) for c in self.factor_mat.columns]
            ),
        )
        return pd.concat([df, fac_df], axis=1)


class Alpha158WithJKP(Alpha158):
    def __init__(self, jkp_factor_mat: pd.DataFrame, **kwargs):
        gfm = GlobalFactorMerger(jkp_factor_mat, fields_group="prior")

        kwargs["infer_processors"] = [gfm] + kwargs.get("infer_processors", [])

        super().__init__(**kwargs)


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--data_handler_config', type=str, help="Path to the data handler config file.",
                        default=get_root_dir().joinpath('configs', '2024_csi300.yaml'))
    parser.add_argument('--universe', type=str, help="Market universe (csi300 or sp500 or csi500)",
                        default="csi300")
    return parser.parse_args()


def build_factor_matrix(raw_df: pd.DataFrame, query: str, qlib_calendar: pd.Index, window: int) -> pd.DataFrame:
    """Pivot raw factor returns and compute a rolling cumulative return without current-day data."""
    factor_returns = (
        raw_df
        .query(query)
        .pivot(index="date", columns="name", values="ret")
        .sort_index()
    )

    factor_returns.columns = [f"JKP_{c}" for c in factor_returns.columns]
    factor_returns = factor_returns.reindex(qlib_calendar)

    shifted = factor_returns.shift(1)
    rolling_log = np.log1p(shifted).rolling(window=window, min_periods=window).sum()
    cumulative_returns = np.expm1(rolling_log)

    cumulative_returns.columns = [f"{col}_RET{window}D" for col in cumulative_returns.columns]
    return cumulative_returns


if __name__ == "__main__":
    args = load_args()
    print(args)
    window = 20

    if args.universe == "csi300":
        print("********** China Market **********")
        provider_uri = "qlib_data/cn_data"  # target_dir

        qlib.init(provider_uri=provider_uri, region=REG_CN)
        name = '[chn]_[all_themes]_[daily]_[vw_cap].csv'
        market = "csi300"
        benchmark = "SH000300"
        region = 'CN'
        query = "location=='chn' and weighting=='vw_cap' and freq=='daily'"
        with open(f"dataset/2024_csi300.yaml", 'r') as f:
            config = yaml.safe_load(f)

    elif args.universe == "csi500":
        print("********** China Market 500 **********")
        provider_uri = "qlib_data/cn_data"  # target_dir

        qlib.init(provider_uri=provider_uri, region=REG_CN)
        name = '[chn]_[all_themes]_[daily]_[vw_cap].csv'
        market = "csi500"
        benchmark = "SH000500"
        region = 'CN'
        query = "location=='chn' and weighting=='vw_cap' and freq=='daily'"
        with open(f"dataset/2024_csi500.yaml", 'r') as f:
            config = yaml.safe_load(f)

    elif args.universe == "sp500":
        print("********** US Market **********")
        provider_uri = "qlib_data/us_data"  # target_dir
        name = '[usa]_[all_themes]_[daily]_[vw_cap].csv'
        market = "sp500"
        benchmark = "^gspc"
        region = 'US'
        print(f"provider_uri: {provider_uri}, region: {REG_US}, name: {name}")
        qlib.init(provider_uri=provider_uri, region=REG_US)
        with open(f"dataset/2024_sp500.yaml", 'r') as f:
            config = yaml.safe_load(f)
        query = "location=='usa' and weighting=='vw_cap' and freq=='daily'"

    else:
        raise ValueError(f"Invalid universe: {args.universe}")

    seq_len = config['task']['dataset']['kwargs']['step_len']
    raw_df = pd.read_csv(f'dataset/data/{name}')
    raw_df["date"] = pd.to_datetime(raw_df["date"])

    qlib_calendar = D.calendar(start_time=config['data_handler_config']['start_time'],
                               end_time=config['data_handler_config']['end_time'])

    factor_mat = build_factor_matrix(raw_df, query, qlib_calendar, window)

    horizons = list(range(0, 10))
    label_expr = [f"Ref($close, -{h + 1}) / Ref($close, -1) - 1" for h in horizons]
    label_names = [f"RET_{h + 1}D" for h in horizons]

    config['data_handler_config']["label"] = (label_expr, label_names)
    handler = Alpha158WithJKP(factor_mat, **config['data_handler_config'])

    dataframe = handler.fetch(col_set="__all", data_key=DataHandlerLP.DK_L)
    df_I = handler.fetch(col_set="__all", data_key=DataHandlerLP.DK_I)
    print("=== dataframe index info ===")
    print(f"index names: {dataframe.index.names}")
    print(f"index sample: {dataframe.index[:5]}")
    print(f"shape: {dataframe.shape}")
    print()

    dataframe.to_pickle(f"./dataset/data/{region}/{args.universe}_{seq_len}_dataframe_learn.pkl")
    df_I.to_pickle(f"./dataset/data/{region}/{args.universe}_{seq_len}_dataframe_infer.pkl")

    dataset = TSDatasetH(
        handler=handler,
        segments=config['task']['dataset']['kwargs']['segments'],
        step_len=config['task']['dataset']['kwargs']['step_len'],
        fillna_type='ffill+bfill'
    )

    if not os.path.exists("./dataset/data/CN"):
        os.makedirs("./dataset/data/CN")

    if not os.path.exists("./dataset/data/US"):
        os.makedirs("./dataset/data/US")

    print("Preparing datasets...")  # DataLoader yields (feature, prior, label/future_returns) in order
    dl_train = dataset.prepare(
        "train", col_set=["feature", "prior", "label"], data_key=DataHandlerLP.DK_L)
    dl_valid = dataset.prepare(
        "valid", col_set=["feature", "prior", "label"], data_key=DataHandlerLP.DK_L)
    dl_test = dataset.prepare(
        "test", col_set=["feature", "prior", "label"], data_key=DataHandlerLP.DK_I)  # DK_I

    print(f"dl_train type: {type(dl_train)}")

    if hasattr(dl_train, 'get_index'):
        train_index = dl_train.get_index()
        print(f"dl_train index type: {type(train_index)}")
        print(f"dl_train index names: {train_index.names}")
        print(f"dl_train index sample (first 5): {train_index[:5]}")
        print(f"dl_train index length: {len(train_index)}")
    else:
        print("dl_train has no get_index() method.")

    if hasattr(dl_valid, 'get_index'):
        valid_index = dl_valid.get_index()
        print(f"dl_valid index type: {type(valid_index)}")
        print(f"dl_valid index names: {valid_index.names}")
        print(f"dl_valid index sample (first 5): {valid_index[:5]}")
        print(f"dl_valid index length: {len(valid_index)}")

    if hasattr(dl_test, 'get_index'):
        test_index = dl_test.get_index()
        print(f"dl_test index type: {type(test_index)}")
        print(f"dl_test index names: {test_index.names}")
        print(f"dl_test index sample (first 5): {test_index[:5]}")
        print(f"dl_test index length: {len(test_index)}")

    dl_train.config(fillna_type='ffill+bfill')
    dl_valid.config(fillna_type='ffill+bfill')
    dl_test.config(fillna_type='ffill+bfill')

    with open(f"./dataset/data/{region}/{args.universe}_{seq_len}_h{len(horizons)}_dl2_train.pkl", "wb") as f:
        pickle.dump(dl_train, f)
    with open(f"./dataset/data/{region}/{args.universe}_{seq_len}_h{len(horizons)}_dl2_valid.pkl", "wb") as f:
        pickle.dump(dl_valid, f)
    with open(f"./dataset/data/{region}/{args.universe}_{seq_len}_h{len(horizons)}_dl2_test.pkl", "wb") as f:
        pickle.dump(dl_test, f)
    with open(f"./dataset/data/{region}/{args.universe}_{seq_len}_h{len(horizons)}_dl2_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
