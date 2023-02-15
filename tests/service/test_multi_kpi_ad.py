# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : test_multi_kpi_ad.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description:
"""

import datetime
import time
import unittest

import numpy as np
import pandas as pd

from src.common.constant import MultiKPIADParams
from src.service.multi_kpi_ad import load_data, MultiKpiAnomalyDetection, multi_kpi_ad_api


class MultiADModelTest(unittest.TestCase):
    """Test anomaly detection model for multiple KPI anomaly detection."""

    def setUp(self) -> None:
        self.x, self.y = load_data()

        t = datetime.datetime.today().timestamp() // 60 * 60
        ts = np.arange(t - 2 * 24 * 60 * 60, t, 60)
        self.num = 1
        self.kwargs = {'__start_time__': ts[-self.num]}
        metric_list = MultiKPIADParams.METRIC_LIST
        v = {metric: np.random.random(size=(2880,)) for metric in metric_list}
        v['timestamp'] = ts
        self.dataframe = pd.DataFrame(v)

    def test_multi_ad_model(self):
        multi_kpi_ad = MultiKpiAnomalyDetection(list(self.x.shape[1:]))
        multi_kpi_ad.loads_model()
        anomaly, scores, x_cons = multi_kpi_ad.predict(x=self.x[:5])

        self.assertTrue(all((anomaly == 0) | (anomaly == 1)))
        self.assertTrue(all(scores >= 0))
        self.assertTrue(all((x_cons >= 0).reshape(-1, )))

        multi_kpi_ad.fit(self.x, self.y, epochs=1)
        anomaly, scores, x_cons = multi_kpi_ad.predict(x=self.x[:5])

        self.assertTrue(all((anomaly == 0) | (anomaly == 1)))
        self.assertTrue(all(scores >= 0))
        self.assertTrue(all((x_cons >= 0).reshape(-1, )))

    def test_multi_kpi_ad_api(self):
        s_time = time.time()
        df_pred = multi_kpi_ad_api(self.dataframe, None, **self.kwargs)
        e_time = time.time()
        self.assertTrue(e_time - s_time < 0.1)

        self.assertTrue(e_time - s_time < 0.1)
        self.assertTrue(all(df_pred['is_anomaly'] == 0))
