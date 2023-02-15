# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : util_thre_rec.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description:
"""

import numpy as np

from src.common.exception import CommonException


def th_rec_with_label(scores: np.ndarray, labels: np.ndarray, n_r=0.95, a_r=0.05, ws=10, epochs=5):
    """threshold recommendation for binary classification problem"""

    if scores.shape != labels.shape:
        err = f'Threshold recommendation，unexpected input shape, ' \
              f'scores.shape={scores.shape}, labels.shape={labels.shape}'
        raise CommonException(err)

    if np.unique(labels).shape[0] != 2:
        err = f'Threshold recommendation，the number of label type is unexpected, label={np.unique(labels)}'
        raise CommonException(err)

    th_n = np.percentile(scores[labels == 0], n_r * 100)
    th_a = np.percentile(scores[labels == 1], a_r * 100)

    initial_th_l, initial_th_r = min(th_n, th_a), max(th_n, th_a)

    # windows
    scores = np.repeat(scores, ws).reshape(scores.shape[0], ws)
    labels = np.repeat(labels, ws).reshape(labels.shape[0], ws)

    pre_score, threshold = 0, 0
    for _ in range(epochs):

        ths = np.linspace(initial_th_l, initial_th_r, ws)
        ths_ext = np.repeat(ths, scores.shape[0]).reshape(ws, scores.shape[0]).T
        pred = (scores > ths_ext).astype(float)

        precision = np.sum(((pred == 1) & (labels == 1)).astype(float), axis=0) / np.sum(pred, axis=0)
        recall = np.sum(((pred == 1) & (labels == 1)).astype(float), axis=0) / np.sum(labels, axis=0)
        f1_scores = 2 * precision * recall / (precision + recall)

        f1_score, idx = np.max(f1_scores), np.argmax(f1_scores)
        if f1_score <= pre_score:
            break
        threshold, pre_score = ths[idx], f1_score
        initial_th_l = ths[idx - 1] if idx > 0 else ths[idx]
        initial_th_r = ths[idx + 1] if idx < ws - 1 else ths[idx]

    return threshold, pre_score
