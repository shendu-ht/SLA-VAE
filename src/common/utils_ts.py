# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : utils_ts.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""

import numpy as np

from src.common.constant import SRParams, CSDParams
from src.common.exception import CommonException
from src.common.utils_smooth import ma_smooth, ext_smooth


def constant_detect(x: np.ndarray, **kwargs):
    """
    Detect whether the time series x is constant. Fourier transform can hardly deal with constant data.

    :param x:
    :param kwargs:
    :return:
    """

    # The hyperparameters of constant detect
    is_strict = kwargs.get('is_strict', CSDParams.IS_STRICT)
    is_diff = kwargs.get('is_diff', CSDParams.IS_DIFF)
    th = kwargs.get('constant_ratio_threshold', CSDParams.CONSTANT_RATIO_THRESHOLD)

    dim = len(x.shape)

    # x.shape=[N,]
    if dim == 1:
        def const_dim_1(x_i):
            """Detect one-dimensional data"""

            _, count_i = np.unique(x_i, return_counts=True)
            return np.max(count_i) > x_i.shape[0] * th

        if is_strict:
            return np.max(x) == np.min(x)
        if is_diff:
            return const_dim_1(x) | const_dim_1(np.diff(x))
        return const_dim_1(x)

    # x.shape=[Batch, N]
    elif dim == 2:
        def const_dim_2(x_i):
            """Detect two-dimensional data"""

            max_count_i = np.zeros(shape=(x_i.shape[0],))
            for i in range(x_i.shape[0]):
                _, count_i = np.unique(x_i[i, :], return_counts=True)
                max_count_i[i] = np.max(count_i)
            return max_count_i > x_i.shape[1] * th

        if is_strict:
            return np.max(x, axis=1) == np.min(x, axis=1)  # [Batch]
        if is_diff:
            return const_dim_2(x) | const_dim_2(np.diff(x, axis=1))
        return const_dim_2(x)
    else:
        err = 'Constant detection, dim>=3 is not supported.'
        raise CommonException(err)


def sr_algo(x: np.ndarray, window: int, **kwargs):
    """
    Spectral residual algorithm to extract the spectral residual of time series.

    :param x:
    :param window:
    :param kwargs:
    :return:
    """

    extent_size = kwargs.get('extent_size', SRParams.EXTENT_SIZE)

    dim = len(x.shape)
    is_constant = constant_detect(x, **kwargs)
    smooth_x = ma_smooth(ext_smooth(x, **kwargs), **kwargs)

    # one-dimensional data. x.shape=[N,]
    if dim == 1:
        if is_constant:
            return np.zeros_like(x)
        before_x = (smooth_x[0] - smooth_x[window]) + smooth_x[window - extent_size: window] \
            if x.shape[0] > window else [smooth_x[0]] * extent_size
        extent_x = (smooth_x[-1] - smooth_x[-window - 1]) + smooth_x[-window:-window + extent_size] \
            if x.shape[0] > window else [smooth_x[-1]] * extent_size
        x_new = np.concatenate((before_x, x, extent_x), axis=0)

    # two-dimensional data. x.shape=[Batch, N]
    elif dim == 2:
        if x.shape[1] > window:
            before_x = np.repeat(
                (smooth_x[:, 0] - smooth_x[:, window]), extent_size
            ).reshape(x.shape[0], extent_size) + smooth_x[:, window - extent_size: window]
            extent_x = np.repeat(
                (smooth_x[:, -1] - smooth_x[:, -window - 1]), extent_size
            ).reshape(x.shape[0], extent_size) + smooth_x[:, -window:-window + extent_size]
        else:
            before_x = np.repeat(smooth_x[:, 0], extent_size).reshape(x.shape[0], extent_size)
            extent_x = np.repeat(smooth_x[:, -1], extent_size).reshape(x.shape[0], extent_size)
        x_new = np.concatenate((before_x, x, extent_x), axis=1)
    else:
        err = 'Spectral residual algorithm, dim>=3 is not supported.'
        raise CommonException(err)

    # Extract spectral residual
    freq = np.fft.fft(x_new)
    mag = np.sqrt(freq.real ** 2 + freq.imag ** 2)
    mag_log = np.log(mag)
    mag_smooth = ma_smooth(mag_log)
    sr = np.exp(mag_log - mag_smooth)
    freq.real = freq.real * sr / mag
    freq.imag = freq.imag * sr / mag

    saliency_map = np.fft.ifft(freq)
    if dim == 1:
        saliency_map = saliency_map.real[extent_size:extent_size + x.shape[0]]
    elif dim == 2:
        saliency_map = saliency_map.real[:, extent_size:extent_size + x.shape[1]]
        # process constant time series
        saliency_map[is_constant, :] = 0

    # The output may be uncontrollable in extreme case. Limit the output range.
    return np.clip(saliency_map, -1.0, 1.0)
