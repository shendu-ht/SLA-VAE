# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : feature_ext.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""

import random

import numpy as np

from src.common.constant import TSFParams
from src.common.exception import FeatureException
from src.common.utils_smooth import ext_smooth, ma_smooth
from src.common.utils_ts import sr_algo
from src.common.utils_window import p_window_ext, f_window_ext, e_window_ext
from src.model.kde import KDE


class TSAnomalousPattern:
    """
    Extract anomalous features of time series data
    """

    __slots__ = ['freq', 'sample_size', 'operators', 'kwargs']

    def __init__(self, freq, **kwargs):
        """initialize."""

        if freq > TSFParams.FREQ_MIN:
            err = f'The acquisition time interval of time series should be less than 10min，input freq={freq}min.'
            raise FeatureException(err)
        self.freq = freq
        self.sample_size = kwargs.get('kde_sample_size', TSFParams.KDE_SAMPLE_SIZE)
        self.operators = {
            'periodic_window': self._periodic_window,
            'periodic_diff_window': self._periodic_diff_window,
            'diff_window': self._diff_window,
            'evt_window': self._evt_window,
            'sr_window': self._sr_window,
            'pr_window': self._pr_window,
        }
        self.kwargs = kwargs

    def reset_kwargs(self, **kwargs):
        """reset params"""
        self.kwargs = kwargs

    def _periodic_window(self, x: np.ndarray, idx: list):
        """
        Periodic window extraction

        :param x: x.shape can be [N,], [Batch/Metric, N]
        :param idx: corresponding index list of values expected to extract anomalous features
        :return: (window_obs, cur_obs)
                  window_obs: used to fit kernel density distribution
                        window_obs.shape = [idx_len, 1, M], when dim = 1
                                           [idx_len, Batch/Metric, 1, M], when dim = 2
                  cur_obs: target value needed to extracted anomalous features.
                        cur_obs.shape = [idx_len, 1, 1], when dim = 1
                                        [idx_len, Batch/Metric, 1, 1], when dim = 2
        """

        pre_window = self.kwargs.get('tsa_pre_window_size', TSFParams.PRE_WINDOW_SIZE) // self.freq
        post_window = self.kwargs.get('tsa_post_window_size', TSFParams.POST_WINDOW_SIZE) // self.freq

        # samples collected in a day
        day_window = 1440 // self.freq

        dim = len(x.shape)
        idx_len = len(idx)
        if dim == 1:
            window_obs = np.zeros(shape=(idx_len, 1, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, 1, 1))

            # Due to the low complexity of window extraction，no need for performance optimization.
            for i in range(idx_len):
                # the last value is cur_obs
                x_window = p_window_ext(x, pre_window, post_window, day_window, idx[i]).tolist()
                cur_obs[i, 0, 0] = x_window[-1]

                x_window = x_window[:-1]
                while len(x_window) < self.sample_size and not x_window:
                    x_window += x_window
                window_obs[i, 0, :] = random.sample(x_window, self.sample_size)
        elif dim == 2:
            window_obs = np.zeros(shape=(idx_len, x.shape[0], 1, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, x.shape[0], 1, 1))

            for i in range(idx_len):
                x_window = p_window_ext(x, pre_window, post_window, day_window, idx[i])
                cur_obs[i, :, 0, 0] = x_window[:, -1]

                x_window = x_window[:, :-1]
                while x_window.shape[1] < self.sample_size and not x_window.shape[0]:
                    x_window = np.concatenate((x_window, x_window), axis=1)
                idx_s = random.sample(list(range(x_window.shape[1])), self.sample_size)
                window_obs[i, :, 0, :] = x_window[:, idx_s]
        else:
            err = f'Periodic window extraction, dim>=3 is not supported.'
            raise FeatureException(err)
        return window_obs, cur_obs

    def _periodic_diff_window(self, x: np.ndarray, idx: list):
        """
        Periodic differential window extraction

        :param x:
        :param idx:
        :return: (window_obs, cur_obs)
                        window_obs.shape = [idx_len, 2, M], when dim = 1
                                           [idx_len, Batch/Metric, 2, M], when dim = 2
                        cur_obs.shape = [idx_len, 2, 1], when dim = 1
                                        [idx_len, Batch/Metric, 2, 1], when dim = 2
        """

        focus_window = self.kwargs.get('diff_focus_window', TSFParams.FOCUS_WINDOW_SIZE)
        day_window = 1440 // self.freq

        dim = len(x.shape)
        idx_len = len(idx)
        if dim == 1:
            window_obs = np.zeros(shape=(idx_len, 2, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, 2, 1))
            for i in range(idx_len):
                cur_day = f_window_ext(x, day_window, idx[i])
                pre_day = f_window_ext(x, day_window, idx[i] - day_window)
                if pre_day.shape[0] < cur_day.shape[0]:
                    pre_day = np.concatenate((cur_day[: -pre_day.shape[0]], pre_day), axis=0)

                x_window = cur_day - ext_smooth(pre_day, **self.kwargs)
                # keep the variance of values in focus window
                ext_smooth(x_window[:-focus_window], **self.kwargs)
                # transformed cur_obs
                cur_obs[i, 0, 0], cur_obs[i, 1, 0] = x_window[-1], x_window[-1]

                x_window = x_window[:-1]
                # up and fall window_obs used to fit estimation
                window_obs[i, 0, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, 1, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        elif dim == 2:
            window_obs = np.zeros(shape=(idx_len, x.shape[0], 2, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, x.shape[0], 2, 1))
            for i in range(idx_len):
                cur_day = f_window_ext(x, day_window, idx[i])
                pre_day = f_window_ext(x, day_window, idx[i] - day_window)
                if pre_day.shape[1] < cur_day.shape[1]:
                    pre_day = np.concatenate((cur_day[:, : -pre_day.shape[1]], pre_day), axis=1)

                x_window = cur_day - ext_smooth(pre_day, **self.kwargs)
                ext_smooth(x_window[:, :-focus_window], **self.kwargs)
                cur_obs[i, :, 0, 0], cur_obs[i, :, 1, 0] = x_window[:, -1], x_window[:, -1]

                x_window = x_window[:, :-1]
                window_obs[i, :, 0, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, :, 1, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        else:
            err = f'Periodic differential window extraction, dim>=3 is not supported.'
            raise FeatureException(err)
        return window_obs, cur_obs

    def _diff_window(self, x: np.ndarray, idx: list):
        """
        Differential window extraction

        :param x:
        :param idx:
        :return: (window_obs, cur_obs)
                        window_obs.shape = [idx_len, 2, M], when dim = 1
                                           [idx_len, Batch/Metric, 2, M], when dim = 2
                        cur_obs.shape = [idx_len, 2, 1], when dim = 1
                                        [idx_len, Batch/Metric, 2, 1], when dim = 2
        """

        focus_window = self.kwargs.get('diff_focus_window', TSFParams.FOCUS_WINDOW_SIZE)
        day_window = 1440 // self.freq

        dim = len(x.shape)
        idx_len = len(idx)
        if dim == 1:
            window_obs = np.zeros(shape=(idx_len, 2, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, 2, 1))
            for i in range(idx_len):
                cur_day = f_window_ext(x, day_window + focus_window + 1, idx[i])
                # remove extreme values to avoid deviation.
                # Pointer operation, cur_day will be smoothed
                ext_smooth(cur_day, **self.kwargs)

                x_window = np.diff(cur_day)
                cur_obs[i, 0, 0], cur_obs[i, 1, 0] = x_window[-1], x_window[-1]

                x_window = x_window[:-1]
                window_obs[i, 0, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, 1, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        elif dim == 2:
            window_obs = np.zeros(shape=(idx_len, x.shape[0], 2, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, x.shape[0], 2, 1))
            for i in range(idx_len):
                cur_day = f_window_ext(x, day_window + focus_window + 1, idx[i])
                ext_smooth(cur_day, **self.kwargs)

                x_window = np.diff(cur_day)
                cur_obs[i, :, 0, 0], cur_obs[i, :, 1, 0] = x_window[:, -1], x_window[:, -1]
                x_window = x_window[:, :-1]
                window_obs[i, :, 0, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, :, 1, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        else:
            err = f'Differential window extraction, dim>=3 is not supported.'
            raise FeatureException(err)
        return window_obs, cur_obs

    def _evt_window(self, x: np.ndarray, idx: list):
        """
        Extreme window extraction

        :param x:
        :param idx:
        :return: (window_obs, cur_obs)
                        window_obs.shape = [idx_len, 2, M], when dim = 1
                                           [idx_len, Batch/Metric, 2, M], when dim = 2
                        cur_obs.shape = [idx_len, 2, 1], when dim = 1
                                        [idx_len, Batch/Metric, 2, 1], when dim = 2
        """

        focus_window = self.kwargs.get('series_focus_window', TSFParams.FOCUS_WINDOW_SIZE)
        day_window = 1440 // self.freq

        dim = len(x.shape)
        idx_len = len(idx)
        if dim == 1:
            window_obs = np.zeros(shape=(idx_len, 2, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, 2, 1))
            for i in range(idx_len):
                x_window = f_window_ext(x, day_window + focus_window, idx[i])
                ext_smooth(x_window, **self.kwargs)

                cur_obs[i, 0, 0], cur_obs[i, 1, 0] = x_window[-1], x_window[-1]
                x_window = x_window[:-1]
                window_obs[i, 0, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, 1, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        elif dim == 2:
            window_obs = np.zeros(shape=(idx_len, x.shape[0], 2, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, x.shape[0], 2, 1))
            for i in range(idx_len):
                x_window = f_window_ext(x, day_window + focus_window, idx[i])
                ext_smooth(x_window, **self.kwargs)

                cur_obs[i, :, 0, 0], cur_obs[i, :, 1, 0] = x_window[:, -1], x_window[:, -1]
                x_window = x_window[:, :-1]
                window_obs[i, :, 0, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, :, 1, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        else:
            err = f'Extreme window extraction, dim>=3 is not supported.'
            raise FeatureException(err)
        return window_obs, cur_obs

    def _sr_window(self, x: np.ndarray, idx: list):
        """
        Spectral residual window extraction

        :param x:
        :param idx:
        :return: (window_obs, cur_obs)
                        window_obs.shape = [idx_len, 3, M], when dim = 1
                                           [idx_len, Batch/Metric, 3, M], when dim = 2
                        cur_obs.shape = [idx_len, 3, 1], when dim = 1
                                        [idx_len, Batch/Metric, 3, 1], when dim = 2
        """

        focus_window = self.kwargs.get('series_focus_window', TSFParams.FOCUS_WINDOW_SIZE)
        day_window = 1440 // self.freq

        # Transform original time series into spectral residual
        sr = sr_algo(x, day_window, **self.kwargs)

        dim = len(x.shape)
        idx_len = len(idx)

        if dim == 1:
            window_obs = np.zeros(shape=(idx_len, 3, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, 3, 1))
            for i in range(idx_len):
                x_window = f_window_ext(sr, day_window + focus_window, idx[i])
                cur_obs[i, 0, 0], cur_obs[i, 1, 0], cur_obs[i, 2, 0] = x_window[-1], x_window[-1], x_window[-1]

                x_window = x_window[:-1]
                while x_window.shape[0] < self.sample_size and not x_window.shape[0]:
                    x_window = np.concatenate((x_window, x_window), axis=0)
                idx_s = random.sample(list(range(x_window.shape[0])), self.sample_size)
                window_obs[i, 0, :] = x_window[idx_s]
                window_obs[i, 1, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, 2, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        elif dim == 2:
            window_obs = np.zeros(shape=(idx_len, x.shape[0], 3, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, x.shape[0], 3, 1))
            for i in range(idx_len):
                x_window = f_window_ext(sr, day_window + focus_window, idx[i])
                cur_obs[i, :, 0, 0] = x_window[:, -1]
                cur_obs[i, :, 1, 0] = x_window[:, -1]
                cur_obs[i, :, 2, 0] = x_window[:, -1]

                x_window = x_window[:, :-1]
                while x_window.shape[1] < self.sample_size and not x_window.shape[0]:
                    x_window = np.concatenate((x_window, x_window), axis=1)
                idx_s = random.sample(list(range(x_window.shape[1])), self.sample_size)
                window_obs[i, :, 0, :] = x_window[:, idx_s]
                window_obs[i, :, 1, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, :, 2, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        else:
            err = f'Spectral residual window extraction, dim>=3 is not supported.'
            raise FeatureException(err)
        return window_obs, cur_obs

    def _pr_window(self, x: np.ndarray, idx: list):
        """
        Prediction error window extraction

        :param x:
        :param idx:
        :return: (window_obs, cur_obs)
                        window_obs.shape = [idx_len, 2, M], when dim = 1
                                           [idx_len, Batch/Metric, 2, M], when dim = 2
                        cur_obs.shape = [idx_len, 2, 1], when dim = 1
                                        [idx_len, Batch/Metric, 2, 1], when dim = 2
        """

        focus_window = self.kwargs.get('series_focus_window', TSFParams.FOCUS_WINDOW_SIZE)
        day_window = 1440 // self.freq

        dim = len(x.shape)
        idx_len = len(idx)
        if dim == 1:
            window_obs = np.zeros(shape=(idx_len, 2, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, 2, 1))
            for i in range(idx_len):
                cur_day = f_window_ext(x, day_window + focus_window, idx[i])
                ext_smooth(cur_day, **self.kwargs)

                # adopt the moving average as prediction for operating efficiency
                x_window = cur_day - ma_smooth(cur_day)
                cur_obs[i, 0, 0], cur_obs[i, 1, 0] = abs(x_window[-1]), x_window[-1]
                x_window = x_window[:-1]
                window_obs[i, 0, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, 1, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        elif dim == 2:
            window_obs = np.zeros(shape=(idx_len, x.shape[0], 2, self.sample_size))
            cur_obs = np.zeros(shape=(idx_len, x.shape[0], 2, 1))
            for i in range(idx_len):
                cur_day = f_window_ext(x, day_window + focus_window, idx[i])
                ext_smooth(cur_day, **self.kwargs)

                x_window = cur_day - ma_smooth(cur_day)
                cur_obs[i, :, 0, 0], cur_obs[i, :, 1, 0] = x_window[:, -1], x_window[:, -1]
                x_window = x_window[:, :-1]
                window_obs[i, :, 0, :] = e_window_ext(x_window, self.sample_size, type_='up')
                window_obs[i, :, 1, :] = e_window_ext(x_window, self.sample_size, type_='fall')
        else:
            err = f'Prediction error window extraction, dim>=3 is not supported.'
            raise FeatureException(err)
        return window_obs, cur_obs

    def kpi_feature_ext(self, x: np.ndarray, idx: list):
        """
        Anomalous feature extraction for single KPI time series.

        :param x: x.shape=[N,]
        :param idx: index list of target values expected to extract anomalous features
        :return:
        """

        dim = len(x.shape)
        if dim != 1:
            err = f'Single KPI time series feature extraction，input x.shape={x.shape} did not meet expectations.'
            raise FeatureException(err)

        kpi_operators = self.kwargs.get('kpi_operators', TSFParams.KPI_OPERATORS)
        window_obs, cur_obs = None, None

        for operator in kpi_operators:
            window_obs_op, cur_obs_op = self.operators[operator](x, idx)
            window_obs = window_obs_op if window_obs is None else np.concatenate((window_obs, window_obs_op), axis=1)
            cur_obs = cur_obs_op if cur_obs is None else np.concatenate((cur_obs, cur_obs_op), axis=1)

        # Adopt KDE to estimate distribution
        kde = KDE()
        kde.fit(window_obs)

        # one-way distribution [left], one-way distribution [right], two-way distribution
        # different features have different dependencies of distributions.
        # Converting complex problems into matrix operations can reduce running time
        bi_predict = kde.predict(cur_obs, type_='bi')
        left_predict = kde.predict(cur_obs, type_='left')
        right_predict = kde.predict(cur_obs, type_='right')

        kde_direct = self.kwargs.get('kde_direction', TSFParams.KDE_DIRECTION)
        features = np.copy(cur_obs)
        features[:, kde_direct['bi'], :] = bi_predict[:, kde_direct['bi'], :]
        features[:, kde_direct['left'], :] = left_predict[:, kde_direct['left'], :]
        features[:, kde_direct['right'], :] = right_predict[:, kde_direct['right'], :]
        features[:, kde_direct['abs'], :] = np.abs(features[:, kde_direct['abs'], :])
        return features.reshape(features.shape[:-1])

    def multi_kpi_feature_ext(self, x: np.ndarray, idx: list):
        """
        Anomalous feature extraction for multiple KPI time series.

        :param x: x.shape=[Batch, N], where "Batch" is the number of KPI time series.
        :param idx:
        :return:
        """

        dim = len(x.shape)
        if dim != 2:
            err = f'Multiple KPI time series feature extraction，input x.shape={x.shape} did not meet expectations.'
            raise FeatureException(err)

        multi_kpi_operators = self.kwargs.get('multi_kpi_operators', TSFParams.MULTI_KPI_OPERATORS)
        window_obs, cur_obs = None, None
        for operator in multi_kpi_operators:
            window_obs_op, cur_obs_op = self.operators[operator](x, idx)
            window_obs = window_obs_op if window_obs is None else np.concatenate((window_obs, window_obs_op), axis=2)
            cur_obs = cur_obs_op if cur_obs is None else np.concatenate((cur_obs, cur_obs_op), axis=2)

        kde = KDE()
        kde.fit(window_obs)
        bi_predict = kde.predict(cur_obs, type_='bi')
        left_predict = kde.predict(cur_obs, type_='left')
        right_predict = kde.predict(cur_obs, type_='right')

        kde_direct = self.kwargs.get('multi_kde_direction', TSFParams.MULTI_KDE_DIRECTION)
        features = np.copy(cur_obs)
        features[:, :, kde_direct['bi'], :] = bi_predict[:, :, kde_direct['bi'], :]
        features[:, :, kde_direct['left'], :] = left_predict[:, :, kde_direct['left'], :]
        features[:, :, kde_direct['right'], :] = right_predict[:, :, kde_direct['right'], :]
        features[:, :, kde_direct['abs'], :] = np.abs(features[:, :, kde_direct['abs'], :])
        return features.reshape(features.shape[:-1])
