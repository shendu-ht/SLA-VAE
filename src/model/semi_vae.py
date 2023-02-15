# !/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project    : SLA-VAE
@File       : semi_vae.py
@Author     : boyue.ht
@Version    : 1.0
@CreateTime : 2023/2/15
@LastModifiedTime: 2023/2/15
@Description:
"""

from typing import List

import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.model.pid import PIControl


def _loss_semi_vae(x: torch.Tensor, x_hat: torch.Tensor, y: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor):
    """Loss function of semi-supervised vae"""

    n = np.prod(x.shape[1:])

    x, x_hat = x.view(-1, n), x_hat.view(-1, n)
    normal_indices = torch.where(y == 0)[0]
    anomaly_indices = torch.where(y == 1)[0]

    x_n = torch.index_select(x, dim=0, index=normal_indices).detach()
    x_hat_n = torch.index_select(x_hat, dim=0, index=normal_indices)
    mu_n = torch.index_select(mu, dim=0, index=normal_indices)
    log_var_n = torch.index_select(log_var, dim=0, index=normal_indices)

    x_a = torch.index_select(x, dim=0, index=anomaly_indices).detach()
    x_hat_a = torch.index_select(x_hat, dim=0, index=anomaly_indices)

    bce_loss_n = F.binary_cross_entropy(x_hat_n, x_n, reduction='sum') / x_hat_n.shape[0] if x_hat_n.shape[0] else 0
    bce_loss_a = F.binary_cross_entropy(x_hat_a, x_a, reduction='sum') / x_hat_a.shape[0] if x_hat_a.shape[0] else 0
    bce_loss = bce_loss_n - bce_loss_a
    kld_loss = -0.5 * torch.sum(1 + log_var_n - mu_n.pow(2) - log_var_n.exp()) / log_var_n.shape[0] if log_var_n.shape[
        0] else 0
    return bce_loss, kld_loss


def _sample(mu: torch.Tensor, log_var: torch.Tensor):
    """Sample function of vae"""

    std = torch.exp(0.5 * log_var)
    return torch.randn_like(std).mul(std).add_(mu)


class SemiVAE(nn.Module):
    """Semi-supervised variational auto-encoder"""

    def __init__(self, in_dim: List[int], latent_dim: int, hidden: List[int], **kwargs):
        """

        :param in_dim: input dimension
        :param latent_dim: latent var dimension
        :param hidden: hidden layers of encoder and decoder layers
        :param kwargs:
        """

        super(SemiVAE, self).__init__()

        self.in_dim = in_dim

        n = len(hidden)
        self.encode_layers = nn.ModuleList()
        self.encode_layers.append(nn.Linear(in_features=np.prod(in_dim), out_features=hidden[0]))
        for i in range(n - 1):
            self.encode_layers.append(nn.Linear(in_features=hidden[i], out_features=hidden[i + 1]))

        self.mu = nn.Linear(in_features=hidden[-1], out_features=latent_dim)
        self.log_var = nn.Linear(in_features=hidden[-1], out_features=latent_dim)

        self.latent_layer = nn.Linear(in_features=latent_dim, out_features=hidden[-1])
        self.decoder_layers = nn.ModuleList()
        for i in range(n - 1):
            self.decoder_layers.append(nn.Linear(in_features=hidden[n - i - 1], out_features=hidden[n - i - 2]))
        self.decoder_layers.append(nn.Linear(in_features=hidden[0], out_features=np.prod(in_dim)))

        self.kwargs = kwargs

    def encoder(self, x: torch.Tensor):
        """
        semi-vae encoder

        :param x: support multiple types of input, e.g.,
                    Single KPI time series: [N,]
                    Multiple KPI time series: [M, N], ...
        :return:
        """

        x = torch.flatten(x, start_dim=1)
        for fc in self.encode_layers:
            x = F.relu(fc(x))

        x_mu = self.mu(x)
        x_log_var = nn.Softplus()(self.log_var(x))
        return x_mu, x_log_var

    def decoder(self, x: torch.Tensor):
        """
        semi-vae decoder

        :param x: latent var shape=[N, latent_dim]
        :return:
        """

        x = self.latent_layer(x)
        for fc in self.decoder_layers:
            x = fc(F.relu(x))

        # Since the input is in a range of [0, 1], sigmoid is adopted.
        x = torch.sigmoid(x)

        # [N, np.prod(in_dim)] -> [N, D1, D2, ...]
        x = x.view([x.shape[0]] + self.in_dim)
        return x

    def forward(self, x: torch.Tensor):
        """forward propagation"""

        x_mu, x_log_var = self.encoder(x)
        z = _sample(x_mu, x_log_var)
        x_hat = self.decoder(z)
        return x_hat, x_mu, x_log_var

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=50, batch_size=256, lr=1e-3, desired_kl=10, kp=1e-2, ki=1e-4):
        """semi-vae model training"""

        data_loader = DataLoader(TensorDataset(torch.Tensor(x), torch.Tensor(y)), batch_size=batch_size, shuffle=True)
        opt = optim.Adam(self.parameters(), lr=lr)

        pi = PIControl()

        for _ in range(epochs):
            # train_loss = 0
            for x_batch, y_batch in tqdm(data_loader):
                opt.zero_grad()
                x_hat, mu, log_var = self.forward(x_batch)
                bce_loss, kld_loss = _loss_semi_vae(x_batch, x_hat, y_batch, mu, log_var)
                beta = pi.pi(desired_kl - kld_loss.item(), kp=kp, ki=ki)
                loss = bce_loss + beta * kld_loss
                loss.backward()
                opt.step()
        return

    def predict(self, x: np.ndarray):
        """semi-vae model predicting"""

        x = torch.Tensor(x)
        x_hat, x_mu, x_log_var = self.forward(x)
        return x_hat.detach().numpy(), x_mu.detach().numpy(), x_log_var.detach().numpy()
