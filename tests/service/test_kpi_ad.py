# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : test_kpi_ad.py
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

from src.service.kpi_ad import load_data, KpiAnomalyDetection, kpi_ad_api


class ADModelTest(unittest.TestCase):
    """Test anomaly detection model for single KPI anomaly detection."""

    def setUp(self) -> None:
        self.x, self.y = load_data()

        t = datetime.datetime.today().timestamp() // 60 * 60
        ts = np.arange(t - 2 * 24 * 60 * 60, t, 60)

        self.num = 1
        self.kwargs = {
            '__start_time__': ts[-self.num]
        }
        self.dataframe = pd.DataFrame({'timestamp': ts, 'value': np.random.random(size=(2880,))})

    def test_ad_model(self):
        kpi_ad = KpiAnomalyDetection(list(self.x.shape[1:]), self.y.shape[1])
        kpi_ad.loads_model()
        anomaly, scores, x_cons = kpi_ad.predict(x=self.x[:5])
        self.assertTrue(all((anomaly == 0) | (anomaly == 1)))
        self.assertTrue(all(scores >= 0))
        self.assertTrue(all((x_cons >= 0).reshape(-1, )))

        kpi_ad.fit(x=self.x, y=self.y, epochs=1)
        anomaly, scores, x_cons = kpi_ad.predict(x=self.x[:5])

        self.assertTrue(all((anomaly == 0) | (anomaly == 1)))
        self.assertTrue(all(scores >= 0))
        self.assertTrue(all((x_cons >= 0).reshape(-1, )))

    def test_kpi_ad_api(self):
        s_time = time.time()
        df_pred = kpi_ad_api(self.dataframe, None, **self.kwargs)
        e_time = time.time()
        self.assertTrue(e_time - s_time < 0.1)
        self.assertTrue(all(df_pred['is_anomaly'] == 0))
