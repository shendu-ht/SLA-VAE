# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : utils_timestamp.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description:
"""
import datetime

import numpy as np
import pandas as pd

from src.common.exception import CommonException


def timestamp_to_freq(timestamp: np.ndarray, form: str = 's'):
    """
    Extract the frequency based on timestamps.

    :param timestamp:
    :param form:
    :return: N [in minutes]
    """

    if form == 'ms':
        timestamp = np.sort(timestamp) / 1000
    elif form == 's':
        timestamp = np.sort(timestamp)
    else:
        err = f'The form={form} is not expected，only support "ms" and "s"'
        raise CommonException(err)

    diff, count = np.unique(np.diff(timestamp), return_counts=True)
    max_ratio = np.max(count) / np.sum(count)
    if max_ratio > 0.9:
        return diff[count.tolist().index(np.max(count))] / 60
    else:
        return np.gcd.reduce(np.unique(np.diff(timestamp)).astype(int)) / 60


def restore_timestamp(df: pd.DataFrame, by: str = 'timestamp', fill_value: np.float = np.nan):
    """
    The timestamp may be missing in practice. We restore it by filling with nan.

    :param df:
    :param by:
    :param fill_value:
    :return:
    """

    if df.shape[0] <= 1:
        return df, None

    df = df.sort_values(by=by)
    timestamp = df[by].values
    freq = timestamp_to_freq(timestamp)

    st, et = timestamp.min(), timestamp.max()
    date_index = pd.date_range(datetime.datetime.fromtimestamp(st),
                               datetime.datetime.fromtimestamp(et), freq='{}T'.format(freq))

    df['date_time'] = [datetime.datetime.fromtimestamp(t) for t in timestamp]
    df = df.set_index('date_time').reindex(date_index, fill_value=fill_value)
    df['timestamp'] = [idx.timestamp() for idx in df.index]
    return df, int(freq)


def time_series_process(df: pd.DataFrame, by: str = 'timestamp', **kwargs):
    """
    Process the input time series.
            1. Repair the time series by timestamp.
            2. Extract the target data that are expected to be detected.

    :param df:
    :param by:
    :param kwargs:
    :return:
    """

    df, freq = restore_timestamp(df, by)
    timestamp = df[by].values

    if '__start_time__' not in kwargs or kwargs['__start_time__'] not in timestamp:
        err = f'Time series process, kwargs={kwargs} is unexpected.'
        raise CommonException(err)

    target_index = kwargs.get('__target_index__', None)
    if target_index is None:
        timestamp_index = [datetime.datetime.timestamp(t) for t in df.index]
        start_index = timestamp_index.index(kwargs['__start_time__'])
        target_index = np.arange(start_index, df.shape[0])
        target_index = [idx for idx in target_index if timestamp_index[idx] in timestamp]

    return df, target_index, freq
