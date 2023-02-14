# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : utils_smooth.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""

import numpy as np

from src.common.constant import SmoothParams
from src.common.exception import CommonException
from src.common.utils_cleaning import ext_search, diff_series


def ma_smooth(x: np.ndarray, **kwargs):
    """
    Smoothing based on moving average.

    :param x:
    :param kwargs: params
    :return:
    """

    # ma hyperparameters
    window = kwargs.get('ma_smooth_window_size', SmoothParams.WINDOW_SIZE)
    method = kwargs.get('ma_method', SmoothParams.MA_METHOD)
    decay = kwargs.get('decay', SmoothParams.DECAY)

    dim = len(x.shape)

    # x.shape=[N,]
    if dim == 1:
        x_window = np.zeros(shape=(x.shape[0], window))
        if method == 'avg':
            weight = np.concatenate((np.arange(1, window + 1), np.ones(shape=(x.shape[0] - window,)) * window))
            for i in range(window):
                x_window[i:, i] = x[:x.shape[0] - i]
        elif method == 'exp':
            weight = np.concatenate((
                (1 - np.power(decay, np.arange(1, window + 1))) / (1 - decay),
                np.ones(shape=(x.shape[0] - window,)) * (1 - np.power(decay, window)) / (1 - decay)
            ))
            for i in range(window):
                x_window[i:, window - i - 1] = x[:x.shape[0] - i] * np.power(decay, i)
        else:
            err = 'Smoothing based on moving average, dim>=3 is not supported.'
            raise CommonException(err)

        return np.sum(x_window, axis=1) / weight

    # x.shape=[batch, N]
    elif dim == 2:
        x_window = np.zeros(shape=(x.shape[0], x.shape[1], window))
        if method == 'avg':
            weight = np.concatenate((np.arange(1, window + 1), np.ones(shape=(x.shape[1] - window,)) * window))
            for i in range(window):
                x_window[:, i:, i] = x[:, :(x.shape[1] - i)]
        elif method == 'exp':
            weight = np.concatenate((
                (1 - np.power(decay, np.arange(1, window + 1))) / (1 - decay),
                np.ones(shape=(x.shape[1] - window,)) * (1 - np.power(decay, window)) / (1 - decay)
            ))
            for i in range(window):
                x_window[:, i:, window - i - 1] = x[:, :(x.shape[1] - i)] * np.power(decay, i)
        else:
            err = 'Smoothing based on moving average, dim>=3 is not supported.'
            raise CommonException(err)

        return np.sum(x_window, axis=2) / weight
    else:
        err = f'Smoothing based on moving average, only "avg" and "exp"，input method is {method}.'
        raise CommonException(err)


def _indices_smooth(x: np.ndarray, indices: np.ndarray, **kwargs):
    """
    Given the indices of one time series data, smooth the corresponding value in target indices.

    :param x:
    :param indices: indices.shape == x.shape
    :param kwargs:
    :return:
    """

    start_indices = kwargs.get('start_index', SmoothParams.START_INDEX)
    window = kwargs.get('index_smooth_window_size', SmoothParams.WINDOW_SIZE)
    dim = len(x.shape)

    def _series_smooth(x_i: np.ndarray, index_i: np.ndarray):
        # time series array and index array should be the same shape
        if x_i.shape != index_i.shape:
            raise CommonException(f'Time series array and index array have different shapes, '
                                  f'x.shape={x_i.shape}, index.shape={index_i.shape}.')

        # If all values in x is zero, directly return the original time series.
        if np.max(x_i) == 0:
            return x_i

        normal_indices = np.where(index_i == 0)[0]
        # If all values in x are anomalous, return zeros
        if not normal_indices.shape[0]:
            return np.zeros_like(x_i)

        start_idx, end_idx = normal_indices[0], normal_indices[-1]
        smooth_x_i = np.copy(x_i)
        smooth_x_i[np.where(index_i[start_idx:end_idx] == 1)] = np.nanmean(x_i)
        weight = np.concatenate((np.arange(1, window + 1), np.ones(shape=(smooth_x_i.shape[0] - window,)) * window))
        x_window = np.zeros(shape=(smooth_x_i.shape[0], window))
        for k in range(window):
            x_window[k:, k] = smooth_x_i[:smooth_x_i.shape[0] - k]
        ma_x_i = np.sum(x_window, axis=1) / weight

        # The start index and end index will not be smoothed for the lack of reference data
        roll_window = window // 2
        smooth_x_i[roll_window:smooth_x_i.shape[0] - window + roll_window] = ma_x_i[window:]

        x_i[start_idx: end_idx][np.where(index_i[start_idx:end_idx] == 1)] = smooth_x_i[
            np.where(index_i[start_idx:end_idx] == 1)]
        return x_i

    # x.shape=[N,]
    if dim == 1:
        indices[:start_indices] = 0
        return _series_smooth(x, indices)
    # x.shape=[Batch, N]
    elif dim == 2:
        smooth_x = np.zeros_like(x)
        for i in range(indices.shape[0]):
            indices[i, :][:start_indices] = 0
            smooth_x[i, :] = _series_smooth(x[i, :], indices[i, :])
        return smooth_x
    else:
        err = 'Index smoothing, dim>=3 is not supported.'
        raise CommonException(err)


def ext_smooth(x: np.ndarray, **kwargs):
    """
    Smooth processing for extreme values in time series data

    :param x:
    :param kwargs: params
    :return:
    """

    # If all values in x is nan, return zeros
    if np.min(np.isnan(x)) == 1:
        return np.zeros_like(x)

    anomaly_indices = ext_search(x)
    is_diff = kwargs.get('extremum_diff_smooth', SmoothParams.DIFF_SMOOTH)
    if is_diff:
        diff_x = diff_series(x)
        anomaly_indices = anomaly_indices | ext_search(diff_x)

    # If no extremum, return original time series
    if not np.max(anomaly_indices):
        return x

    return _indices_smooth(x, anomaly_indices, **kwargs)
