# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : multi_kpi_ad.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description:
"""
import json
import logging
import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.constant import MultiKPIADParams
from src.common.exception import ServiceException
from src.common.util_thre_rec import th_rec_with_label
from src.common.utils_dataframe import drop_duplicate
from src.common.utils_time_series import series_verify
from src.common.utils_timestamp import time_series_process
from src.feature.feature_ext import TSAnomalousPattern
from src.model.semi_vae import SemiVAE


def load_data(data_path: str = None):
    """load training data for multiple kpi anomaly detection"""

    if data_path is None:
        data_path = os.path.join(MultiKPIADParams.ABSOLUTE_PATH, MultiKPIADParams.SAMPLE_PATH)
    fea_num = MultiKPIADParams.FEATURE_NUM

    with open(data_path, 'r') as fr:
        x, y = [], []
        for line in fr.readlines():
            vs = line.strip().split('|')
            x.append([float(v) for v in vs[:-1]])
            y.append(float(vs[-1]))
    x, y = np.array(x), np.array(y)
    return x.reshape([x.shape[0], -1, fea_num]), y.reshape([-1, 1])


def _anomaly_score(x_mu: np.ndarray, x_log_var: np.ndarray):
    """anomaly score calculation"""

    return -0.5 * np.sum(1 + x_log_var - np.power(x_mu, 2) - np.exp(x_log_var), axis=1)


class MultiKpiAnomalyDetection:
    """Anomaly detection model for multiple KPI anomaly detection."""

    def __init__(self, in_dim: List[int], latent_dim: int = None, hidden: List[int] = None, **kwargs):

        if latent_dim is None:
            latent_dim = MultiKPIADParams.LATENT_LEN
        if hidden is None:
            hidden = MultiKPIADParams.HIDDEN_LAYERS

        self.semi_vae = SemiVAE(in_dim, latent_dim, hidden, **kwargs)

        # initial threshold
        self.threshold = 0.0
        self.kwargs = kwargs

    def reset_kwargs(self, **kwargs):
        """Resetting params is supported."""

        self.kwargs = kwargs

    def fit(self, x: np.ndarray, y: np.ndarray, test_size=0.2, epochs=50, batch_size=256, lr=1e-3):
        """Anomaly detection model training"""

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        self.semi_vae.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)

        # threshold recommendation
        _, x_mu, x_log_var = self.semi_vae.predict(x_test)
        scores = _anomaly_score(x_mu, x_log_var)
        self.threshold, _ = th_rec_with_label(scores, y_test.reshape(-1, ))

    def predict(self, x: np.ndarray):
        """Anomaly detection model predicting"""

        x_pred, x_mu, x_log_var = self.semi_vae.predict(x)
        scores = _anomaly_score(x_mu, x_log_var)
        anomaly = (scores > self.threshold).astype(float)
        return anomaly, scores, x_pred

    def dumps_model(self, path: str = None):
        """store the parameters and threshold"""

        if path is None:
            path = os.path.join(MultiKPIADParams.ABSOLUTE_PATH, MultiKPIADParams.PARAM_PATH)

        with open(path, 'wb') as fwb:
            fwb.write(pickle.dumps({
                'model': self.semi_vae.state_dict(),
                'threshold': self.threshold
            }))

    def loads_model(self, path: str = None):
        """load the parameters and threshold"""

        if path is None:
            path = os.path.join(MultiKPIADParams.ABSOLUTE_PATH, MultiKPIADParams.PARAM_PATH)

        with open(path, 'rb') as frb:
            params = pickle.load(frb)
            self.semi_vae.load_state_dict(params['model'])
            self.threshold = params['threshold']


def multi_kpi_ad_api(dataframe: pd.DataFrame, model_path, **kwargs):
    """

    :param dataframe: data: pd.DataFrame, columns = ['timestamp', 'value']
                value: single KPI time series.
                timestamp: the corresponding timestamp [in seconds].
    :param model_path: model path
    :param kwargs:
    :return:
    """

    metric_list = kwargs.get('multi_kpi_names', MultiKPIADParams.METRIC_LIST)
    fea_num = kwargs.get('multi_kpi_feature_num', MultiKPIADParams.FEATURE_NUM)
    fea_name = [f'{metric}#f{i + 1}' for metric in metric_list for i in range(fea_num)]

    try:
        df, target_idx, freq = time_series_process(drop_duplicate(dataframe), by='timestamp', **kwargs)
        metrics = np.array(df[metric_list]).T

        tsf_ext = TSAnomalousPattern(freq)
        features = tsf_ext.multi_kpi_feature_ext(series_verify(metrics), target_idx)

        kpi_ad = MultiKpiAnomalyDetection(list(features.shape[1:]), **kwargs)
        kpi_ad.loads_model(model_path)

        anomaly, scores, x_pred = kpi_ad.predict(features)
        fea_name = [f'f{i}' for i in range(features.shape[1])]
        fea_json = [json.dumps({name: f for name, f in zip(fea_name, fea.reshape(-1))}) for fea in features]
        fea_pred_json = [json.dumps({name: float(f) for name, f in zip(fea_name, fea.reshape(-1))}) for fea in x_pred]
        ad_result_df = pd.DataFrame({
            'is_anomaly': anomaly,
            'anomaly_score': scores,
            'features': fea_json,
            'pred_features': fea_pred_json
        })

        return pd.concat((df.iloc[-ad_result_df.shape[0]:].reset_index(drop=True), ad_result_df), axis=1)
    except ServiceException as e:
        module_logger = logging.getLogger(__name__)
        module_logger.error(
            f'kpi anomaly detection api running err，\ninput shape={dataframe.shape}，data detail\n{dataframe[-10:]}', e
        )
        columns = ['is_anomaly', 'anomaly_score', 'features', 'pred_features']
        return pd.DataFrame(columns=dataframe.columns.tolist() + columns)
