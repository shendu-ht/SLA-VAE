# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : test_semi_vae.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description:
"""

import unittest

import numpy as np

from src.model.semi_vae import SemiVAE


class SemiVAETest(unittest.TestCase):
    """Tests for semi-supervised variational auto-encoder"""

    def setUp(self) -> None:
        sample_size = 1000
        self.in_dim = [10, 6]
        self.x = np.random.random(size=(sample_size, self.in_dim[0], self.in_dim[1]))
        self.y = np.zeros(shape=(sample_size,))

    def test_semi_vae(self):
        latent_dim = 32
        hidden = [128, 64, 32]
        vae = SemiVAE(self.in_dim, latent_dim=latent_dim, hidden=hidden)
        vae.fit(self.x, self.y, epochs=1)

        num = 5
        y_pred, _, x_log_var = vae.predict(self.x[:num])

        self.assertTrue(y_pred.shape == (num, self.in_dim[0], self.in_dim[1]))
        self.assertTrue(all((y_pred >= 0).reshape(-1, )) & all((y_pred <= 1).reshape(-1, )))
        self.assertTrue(all((x_log_var >= 0).reshape(-1, )))
