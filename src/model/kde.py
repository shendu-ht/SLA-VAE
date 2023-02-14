# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : kde.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""

import numpy as np

from src.common.constant import KDEParams
from src.common.exception import ModelException


def _gaussian(v: np.ndarray):
    """
    Gaussian Kernel

    :param v: v ~ N(0,1) 标准正态分布
    :return: cumulative probability
    """

    return np.exp(-0.5 * np.square(v)) / np.sqrt(2 * np.pi)


def _uniform(v: np.ndarray):
    """
    Uniform Kernel

    :param v: v ~ [-1, 1]
    :return:
    """

    return (np.abs(v) < 0.5).astype(np.float)


class KDE:
    """
    Kernel Density Estimate
    """
    __slots__ = [
        'dims',  # The dim number k of input matrix.
        'bins',  # The number of bins.
        'h',  # The bandwidth of kde.
        'kernel',  # The kernel function.
        'floor',  # The floor value, shape=[N_1, N_2, ..., N_k].
        'ceil',  # The ceil value, shape=[N_1, N_2, ..., N_k].
        'step',  # The step value, shape=[N_1, N_2, ..., N_k].
        'pdf',  # The estimated probability density function.
        'cdf',  # The estimated cumulative density function.
        'kwargs'  # Hyperparameters
    ]

    KERNEL_DICT = {
        'Gaussian': _gaussian,
        'Uniform': _uniform,
    }

    def __init__(self, bins: int = 100, h: float = None, kernel: str = 'Gaussian', **kwargs):
        self.bins = bins
        self.h = h
        self.kernel = self.KERNEL_DICT[kernel]
        self.kwargs = kwargs

    def reset_bins(self, bins):
        self.bins = bins

    def reset_h(self, h):
        self.h = h

    def reset_kernel(self, kernel):
        self.kernel = self.KERNEL_DICT[kernel]

    def reset_kwargs(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, x: np.ndarray):
        """
        Kernel Density Estimation

        :param x: x.shape=[N_1, N_2, ..., N_k, M], where M is the sample size.
        :return:
        """

        epsilon = self.kwargs.get('kde_epsilon', KDEParams.EPSILON)
        n_sigma = self.kwargs.get('kde_n_sigma', KDEParams.N_SIGMA)

        self.dims = x.shape
        x_mean, x_std = np.nanmean(x, axis=-1), np.nanstd(x, axis=-1)
        x_min, x_max = np.nanmin(x, axis=-1), np.nanmax(x, axis=-1)
        x_floor, x_ceil = x_mean - n_sigma * x_std, x_mean + n_sigma * x_std

        # 利用矩阵运算取x_min和x_floor的最小值，并外延一个标准差
        self.floor = (x_min + x_floor) / 2 - np.abs(x_min - x_floor) / 2 - x_std
        self.ceil = (x_max + x_ceil) / 2 + np.abs(x_max - x_ceil) / 2 + x_std
        self.step = (self.ceil - self.floor) / self.bins

        if self.h is None:
            self.h = np.power((4 / 3 / x.shape[-1]), (1 / 5))

        x_new = np.repeat(x, self.bins).reshape(list(x.shape) + [self.bins])
        x_bins = np.repeat(self.floor, x.shape[-1] * self.bins).reshape(list(x.shape) + [self.bins]) + np.repeat(
            self.step, x.shape[-1] * self.bins).reshape(list(x.shape) + [self.bins]) * np.arange(1, self.bins + 1)
        # epsilon确保带宽严格大于0
        x_h = np.repeat(self.h * x_std + epsilon, x.shape[-1] * self.bins).reshape(list(x.shape) + [self.bins])
        x_pdf = np.mean(self.kernel((x_new - x_bins) / x_h), axis=-2)
        p_sum = np.repeat(np.sum(x_pdf, axis=-1), self.bins).reshape(list(x.shape[:-1]) + [self.bins])
        self.pdf = x_pdf / p_sum
        self.cdf = np.cumsum(self.pdf, axis=-1)

    def _idx_prob(self, idx: np.ndarray):
        """获取下标idx对应的概率数据，v.shape[:-1]==idx.shape[:-1]"""

        if self.cdf.shape[:-1] != idx.shape[:-1]:
            err = f'下标idx格式不符合预期，cdf.shape={self.cdf.shape}, idx.shape={idx.shape}'
            raise ModelException(err)

        prob = np.zeros(shape=idx.shape)

        dim = len(idx.shape)
        if dim == 1:
            prob = self.cdf[idx]
        elif dim == 2:
            for i in range(idx.shape[0]):
                prob[i, :] = self.cdf[i, :][idx[i, :]]
        elif dim == 3:
            for i in range(idx.shape[0]):
                for j in range(idx.shape[1]):
                    prob[i, j, :] = self.cdf[i, j, :][idx[i, j, :]]
        elif dim == 4:
            for i in range(idx.shape[0]):
                for j in range(idx.shape[1]):
                    for k in range(idx.shape[2]):
                        prob[i, j, k, :] = self.cdf[i, j, k, :][idx[i, j, k, :]]
        else:
            err = f'KDE._idx_prob暂未支持更高维度的概率转换，当前输入维度idx.shape={idx.shape}'
            raise ModelException(err)
        return prob

    def _prob2feature(self, prob, type_='bi'):
        """将概率转换为特征值"""

        sig = self.kwargs.get('kde_significance', KDEParams.SIGNIFICANCE)
        sig_base = sig ** 2

        if type_ == 'bi':
            prob_v = 1 - prob
            prob = (prob + prob_v) - np.abs(prob - prob_v)
        elif type_ == 'left':
            pass
        elif type_ == 'right':
            prob = 1 - prob
        else:
            err = f'KDE.prob2feature 概率假设检验参数type_不符合预期，输入type_={type_}'
            raise ModelException(err)
        return np.log10(np.clip(prob, sig_base, 1)) / np.log10(sig_base)

    def predict(self, x: np.ndarray, type_='bi'):
        """
        KDE Probability Predict

        :param x:       x.shape=[N_1, N_2, ..., N_k, N] N是样本点个数
        :param type_:   假设检验方向：单向检验【left or right】 双向检验【bi】
        :return:        shape=[N_1, N_2, ..., N_k, N]
        """

        x_dim = len(x.shape)
        if x_dim == len(self.dims):
            x_floor = np.repeat(self.floor, x.shape[-1]).reshape(x.shape)
            idx = np.clip((x - x_floor) / np.repeat(self.step, x.shape[-1]).reshape(x.shape),
                          0, self.bins - 1).astype(int)
            return self._prob2feature(self._idx_prob(idx), type_=type_)
        else:
            err = f'KDE核密度估计，预测输入格式不符合预期，x.shape={x.shape}, self.dims={self.dims}'
            raise ModelException(err)
