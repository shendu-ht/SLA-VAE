# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : test_feature_ext.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""

import unittest

import numpy as np

from src.common.constant import TSFParams
from src.common.exception import FeatureException
from src.feature.feature_ext import TSAnomalousPattern


class SeriesAFTest(unittest.TestCase):
    """Tests for time series anomalous pattern extraction"""

    def setUp(self) -> None:
        metric_num = 10
        series_num = 2880
        self.freq = 1
        self.x1 = np.random.random(size=(series_num,))
        self.x2 = np.random.random(size=(metric_num, series_num))
        self.x3 = np.random.random(size=(metric_num, series_num, 1))
        self.idx = np.arange(2870, 2880)

    def test_kpi_feature(self):
        """Test single kpi feature extraction"""

        tsa = TSAnomalousPattern(freq=self.freq)
        f = tsa.kpi_feature_ext(self.x1, list(self.idx))

        self.assertTrue(f.shape == (self.idx.shape[0], 12))
        self.assertTrue(all((f <= 1).reshape(-1) & (f >= 0).reshape(-1)))

        self.assertRaises(FeatureException, tsa.kpi_feature_ext, self.x2, list(self.idx))
        self.assertRaises(FeatureException, tsa.kpi_feature_ext, self.x3, list(self.idx))

    def test_multi_kpi_feature(self):
        """Test multiple kpi feature extraction"""

        kwargs = {
            'multi_kpi_operators': TSFParams.KPI_OPERATORS,
            'multi_kde_direction': TSFParams.KDE_DIRECTION
        }

        tsa = TSAnomalousPattern(freq=self.freq)
        tsa.reset_kwargs(**kwargs)

        f = tsa.multi_kpi_feature_ext(self.x2, list(self.idx))
        self.assertTrue(f.shape[:2] == (self.idx.shape[0], self.x2.shape[0]))
        self.assertTrue(all((f <= 1).reshape(-1) & (f >= 0).reshape(-1)))

        self.assertRaises(FeatureException, tsa.multi_kpi_feature_ext, self.x1, list(self.idx))
        self.assertRaises(FeatureException, tsa.multi_kpi_feature_ext, self.x3, list(self.idx))
