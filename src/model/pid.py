# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : pid.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description: PID Control to ensure the stability of KL convergence.
                refer to: ControlVAE: Controllable Variational Autoencoder
                https://proceedings.mlr.press/v119/shao20b/shao20b.pdf
"""

import numpy as np


class PIControl:
    """Feedback System, PI Control"""

    def __init__(self):
        self.i_p = 0.0
        self.beta = 0.0
        self.error = 0.0

    def pi(self, err, beta_min=1, n=1, kp=1e-2, ki=1e-4):
        beta_i = None
        for i in range(n):
            p_i = kp / (1.0 + np.exp(err))
            i_i = self.i_p - ki * err

            if self.beta < 1.0:
                i_i = self.i_p
            beta_i = p_i + i_i + beta_min

            self.i_p = i_i
            self.beta = beta_i
            self.error = err

            if beta_i < beta_min:
                beta_i = beta_min
        return beta_i
