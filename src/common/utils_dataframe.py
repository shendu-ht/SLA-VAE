# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : utils_dataframe.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description:
"""

import numpy as np
import pandas as pd

from src.common.exception import CommonException


def drop_duplicate(df: pd.DataFrame, by: str = 'timestamp', max_ratio=0.2, ascending=True):
    """Drop the duplicate lines in dataframe"""

    if by not in df.columns:
        err = f'Drop duplicates in dataframe，df.columns={df.columns} where by={by} is missing.'
        raise CommonException(err)

    duplicate_count = df[by].value_counts()
    is_duplicated = np.where(duplicate_count.values > 1, 1, 0)
    d_ratio = np.sum(is_duplicated) / is_duplicated.shape[0]

    if d_ratio == 0:
        return df.sort_values(by=by, ascending=ascending)
    elif 0 < d_ratio <= max_ratio:
        return df.drop_duplicates(subset=by).reset_index(drop=True).sort_values(by=by, ascending=ascending)

    d_data = df[df.duplicated(subset=by, keep=False)]
    err = f'The duplication ratio of {by} is {d_ratio}，\n{d_data[-10:]}'
    raise CommonException(err)
