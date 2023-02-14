# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : utils_cleaning.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""

import numpy as np

from src.common.constant import CleanParams
from src.common.exception import CommonException


def diff_series(x: np.ndarray):
    """
    Differential processing of time series data

    :param x:
    :return: return shape=x.shape
    """

    diff_x = np.zeros_like(x)
    dim = len(x.shape)

    # x.shape=[N,]
    if dim == 1:
        diff_x[1:] = np.diff(x)
    # x.shape=[Batch, N]
    elif dim == 2:
        diff_x[:, 1:] = np.diff(x)
    else:
        err = 'Differential processing of time series data, dim>=3 is not supported.'
        raise CommonException(err)
    return diff_x


def ext_search(x: np.ndarray, **kwargs):
    """
    Searching extreme value in time series data

    :param x: Multiple dimension time series data：1d->[N], 2d->[Batch, N]
    :param kwargs:
    :return: return shape=x.shape
    """

    method = kwargs.get('es_method', CleanParams.ES_METHOD)
    anomaly_type = kwargs.get('anomaly_type', CleanParams.ANOMALY_TYPE)

    dim = len(x.shape)

    if method == 'NSigma':
        n_sigma = kwargs.get('n_sigma', CleanParams.N_SIGMA)
        if dim == 1:
            upper, lower = np.nanmean(x) + n_sigma * np.nanstd(x), np.nanmean(x) - n_sigma * np.nanstd(x)
        elif dim == 2:
            upper = np.repeat(np.nanmean(x, axis=1) + n_sigma * np.nanstd(x, axis=1), x.shape[1]).reshape(x.shape)
            lower = np.repeat(np.nanmean(x, axis=1) - n_sigma * np.nanstd(x, axis=1), x.shape[1]).reshape(x.shape)
        else:
            err = 'Searching extreme value in time series data, dim>=3 is not supported.'
            raise CommonException(err)
    elif method == 'BoxPlot':
        iqr = kwargs.get('boxplot_iqr', CleanParams.BOXPLOT_IQR)
        if dim == 1:
            q_75, q_25 = np.nanpercentile(x, [75, 25])
            upper, lower = q_75 + iqr * (q_75 - q_25), q_75 - iqr * (q_75 - q_25)
        elif dim == 2:
            q_75, q_25 = np.nanpercentile(x, [75, 25], axis=1)
            upper = np.repeat(q_75 + iqr * (q_75 - q_25), x.shape[1]).reshape(x.shape)
            lower = np.repeat(q_75 - iqr * (q_75 - q_25), x.shape[1]).reshape(x.shape)
        else:
            err = 'Searching extreme value in time series data, dim>=3 is not supported.'
            raise CommonException(err)
    else:
        err = 'Searching extreme value in time series data，only support "NSigma" and "BoxPlot".'
        raise CommonException(err)

    # Select anomalies based on anomaly detection
    # up anomaly
    if anomaly_type == 'up':
        return (x > upper).astype(int)
    # fall anomaly
    elif anomaly_type == 'fall':
        return (x < lower).astype(int)
    # both up and fall anomaly
    elif anomaly_type == 'bi':
        return ((x > upper) | (x < lower)).astype(int)
    err = 'Anomaly direction parameter error，please select one from "up", "fall", and "bi".'
    raise CommonException(err)
