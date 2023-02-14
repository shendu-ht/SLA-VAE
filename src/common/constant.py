# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : constant.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/14
@LastModifiedTime: 2023/2/14
@Description:
"""


class CSDParams:
    """Default Hyperparameters to Detect Whether the Time Series is Constant"""

    # Strict Detection or Not
    IS_STRICT = False

    # Diff the Time Series or Not
    IS_DIFF = True

    # The Constant Detection Ratio
    CONSTANT_RATIO_THRESHOLD = 0.95


class SRParams:
    """Default Hyperparameters of Spectral Residual Algorithm"""

    # The Extent Window Size of SR
    EXTENT_SIZE = 15


class CleanParams:
    """Default Hyperparameters of Time Series Cleaning"""

    """extremum_search"""

    # Extremum Detection Method
    ES_METHOD = 'NSigma'

    # The N sigma
    N_SIGMA = 3

    # The Anomaly Type: ['up', 'down', 'bi']
    ANOMALY_TYPE = 'bi'

    # The Boxplot IQR
    BOXPLOT_IQR = 1.5


class SmoothParams:
    """Default Hyperparameters of Time Series Smoothing"""

    """Move Average Smooth"""

    # Move Average Window Size
    WINDOW_SIZE = 5

    # Move Average Method ['avg', 'exp']
    MA_METHOD = 'avg'

    # Decay Param of 'exp' Method
    DECAY = 0.8

    """Indices Smooth"""

    # The Index to Start Time Series Smooth
    START_INDEX = 0

    """Extremum Smooth"""

    # Whether to process the diff anomalies
    DIFF_SMOOTH = False


class TSFParams:
    """Default Hyperparameters of Time Series Feature Extraction"""

    """Anomalous Pattern"""

    # The minimal sample frequency [maximal collection time interval △t]
    FREQ_MIN = 10

    # The sample size of data sent to kde
    KDE_SAMPLE_SIZE = 120

    # The Periodic Window Size
    PRE_WINDOW_SIZE = 2 * 60  # 2h
    POST_WINDOW_SIZE = 1 * 60  # 1h

    # No Smoothing Data in Focus Window
    FOCUS_WINDOW_SIZE = 1 * 60  # 1h

    # The Operators for Single KPI Metric [More features to represent anomalous pattern]
    KPI_OPERATORS = [
        'periodic_window',
        'periodic_diff_window',
        'diff_window',
        'evt_window',
        'sr_window',
        'pr_window'
    ]

    # The Hypothesis Direction for Each Feature
    KDE_DIRECTION = {
        'bi': [0],
        'left': [2, 4, 6, 9, 11],
        'right': [1, 3, 5, 8, 10],
        'abs': [7]
    }

    # The Operators for Multi Time Series Metric
    MULTI_KPI_OPERATORS = [
        'periodic_window',
        'pr_window',
        'sr_window'
    ]

    MULTI_KDE_DIRECTION = {
        'bi': [0],
        'left': [2, 5],
        'right': [1, 4],
        'abs': [3]
    }


class KDEParams:
    """Default Hyperparameters of Kernel Density Estimation"""

    # Default Sample
    SAMPLE_SIZE = 120

    # Minimum to Avoid Constant
    EPSILON = 1e-6

    # The N Sigma
    N_SIGMA = 3

    # Significance of Hypothesis Test
    SIGNIFICANCE = 1e-3
