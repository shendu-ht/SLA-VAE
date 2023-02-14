# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : utils_window.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""

import numpy as np

from src.common.exception import CommonException


def p_window_ext(x: np.ndarray, pre_window: int, post_window: int, day_window: int, idx: int):
    """
    Periodic window extraction for time series data.

    :param x:
    :param pre_window: Pre-window size
    :param post_window: Post-window size
    :param day_window:
    :param idx:
    :return:
    """

    assert pre_window > 0
    assert post_window > 0

    dim = len(x.shape)

    # one-dimensional data. x.shape=[N, ]
    if dim == 1:
        assert 0 <= idx < x.shape[0]

        # Cold start. Deal with the case that input length did not meet expectations.
        if idx < pre_window:
            x_window = x[: idx + 1]
        else:
            cur_day = x[idx - pre_window: idx + 1]
            if idx < day_window:
                pre_day_post = x[: idx - day_window + post_window]
                x_window = np.concatenate((pre_day_post, cur_day), axis=0)
            else:
                pre_day_post = x[idx - day_window: idx - day_window + post_window]
                pre_day_pre = x[: idx - day_window + 1] if idx < day_window + pre_window else \
                    x[idx - day_window - pre_window: idx - day_window + 1]
                x_window = np.concatenate((pre_day_pre, pre_day_post, cur_day), axis=0)
    # two-dimensional data. x.shape=[Batch, N]
    elif dim == 2:
        assert 0 <= idx < x.shape[1]

        if idx < pre_window:
            x_window = x[:, : idx + 1]
        else:
            cur_day = x[:, idx - pre_window: idx + 1]
            if idx < day_window:
                pre_day_post = x[:, : idx - day_window + post_window]
                x_window = np.concatenate((pre_day_post, cur_day), axis=1)
            else:
                pre_day_post = x[:, idx - day_window: idx - day_window + post_window]
                pre_day_pre = x[:, : idx - day_window + 1] if idx < day_window + pre_window else \
                    x[:, idx - day_window - pre_window: idx - day_window + 1]
                x_window = np.concatenate((pre_day_pre, pre_day_post, cur_day), axis=1)
    else:
        err = f'Periodic window extraction, dim>=3 is not supported.'
        raise CommonException(err)
    return x_window


def f_window_ext(x: np.ndarray, window: int, idx: int):
    """
    Fix window extraction for time series data.

    :param x:
    :param window:
    :param idx:
    :return:
    """

    assert window > 0

    dim = len(x.shape)
    if dim == 1:
        if idx < 0:
            return np.array([])
        elif idx < window:
            return x[: idx + 1]
        return x[idx - window: idx + 1]
    elif dim == 2:
        if idx < 0:
            return np.array([]).reshape(x.shape[0], 0)  # shape=[Batch, 0]
        if idx < window:
            return x[:, :idx + 1]
        return x[:, idx - window: idx + 1]
    else:
        err = f'Fix window extraction, dim>=3 is not supported.'
        raise CommonException(err)


def e_window_ext(x: np.ndarray, sample_size: int, type_: str = 'up'):
    """
    Extract up/fall extreme values in time series data

    :param x:
    :param sample_size:
    :param type_: anomaly type, 'up', 'fall'
    :return:
    """

    if type_ not in ('up', 'fall'):
        err = f'Extreme window extraction, input type_ only support "up" and "fall".'
        raise CommonException(err)

    dim = len(x.shape)
    if dim == 1:
        sort_idxes = np.argsort(x)
        x_window = x[sort_idxes[:sample_size]] if type_ == 'fall' else x[sort_idxes[-sample_size:]]
    elif dim == 2:
        sort_idxes = np.argsort(x, axis=1)
        x_window = np.zeros(shape=(x.shape[0], sample_size))
        for i in range(x.shape[0]):
            x_window[i, :] = x[i, sort_idxes[i, :sample_size]] if type_ == 'fall' else x[
                i, sort_idxes[i, -sample_size:]]
    else:
        err = f'Extreme window extraction, dim>=3 is not supported.'
        raise CommonException(err)
    return x_window
