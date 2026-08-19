import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np 
import logging
import pandas as pd
import copy
from torch.utils.data.dataloader import default_collate

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

        self.index_df = self.data_source.get_index()
        datetime_level = self.index_df.names.index('datetime')
        daily_groups = pd.Series(self.index_df.values).groupby(self.index_df.get_level_values(datetime_level))

        self.daily_count = daily_groups.size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0
        self.dates = daily_groups.groups.keys()

    def __iter__(self):
        date_indices = np.arange(len(self.dates))
        if self.shuffle:
            np.random.shuffle(date_indices)

        # Yield positional indices into the full dataset for each date.
        datetime_level = self.index_df.names.index('datetime')
        all_datetimes = self.index_df.get_level_values(datetime_level)

        for i in date_indices:
            target_date = list(self.dates)[i]
            indices_for_date = np.where(all_datetimes == target_date)[0]
            if len(indices_for_date) != self.daily_count[i]:
                print(f"Warning: Index count mismatch for date {target_date}. Expected {self.daily_count[i]}, Found {len(indices_for_date)}")
            yield indices_for_date

    def __len__(self):
        return len(self.daily_count) # len(self.data_source)
 

def init_data_loader(handler, shuffle, num_workers=0, index=False):
    sampler = DailyBatchSamplerRandom(handler, shuffle)
    num_batches_per_epoch = len(sampler)

    def float_collate_fn(batch):
        batch = default_collate(batch)
        if isinstance(batch, torch.Tensor):
            return batch.float()
        return batch

    data_loader = DataLoader(handler,
                             batch_sampler=sampler,
                             pin_memory=True,
                             num_workers=num_workers,
                             drop_last=False,
                             collate_fn=float_collate_fn)

    if index == True:
        return data_loader, handler, num_batches_per_epoch
    else:
        return data_loader, num_batches_per_epoch

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

def zscore(x):
    return (x - x.mean()).div(x.std())

def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)  
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]

def drop_na(x):
    N = x.shape[0]
    mask = ~x.isnan()
    return mask, x[mask]

def drop_duplicates(x, tolerance=1e-10):
    """
    Return a per-stock mask marking stocks whose feature rows are all distinct.

    The check is vectorized: each row is collapsed to a weighted-sum hash after
    rounding to `tolerance`, and a stock is flagged duplicate if any two
    timesteps share the same hash.

    Args:
        x: Tensor of shape (N, T, F).
        tolerance: float, rounding unit used to absorb floating-point error.

    Returns:
        Bool Tensor of shape (N,); True if the stock has no duplicate rows.
    """
    N, T, F = x.shape
    device = x.device

    weights = torch.arange(1, F+1, device=device, dtype=x.dtype)

    x_rounded = torch.round(x / tolerance)
    row_hash = torch.sum(x_rounded * weights, dim=-1)  # (N, T)

    sorted_hash, _ = torch.sort(row_hash, dim=1)
    dup_flags = (sorted_hash[:, 1:] == sorted_hash[:, :-1])
    mask = ~torch.any(dup_flags, dim=1)

    return mask