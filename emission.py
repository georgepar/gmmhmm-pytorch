import math

import torch
import torch.nn as nn

from util import input_to_tensor


class DiscreteEmissionModel(nn.Module):
    def __init__(self, emission_probs, dtype=torch.float, device="cpu"):
        super(DiscreteEmissionModel, self).__init__()
        self.eps = 1e-6
        self.dtype = dtype
        self.n_observations = emission_probs.shape[0]
        self.n_states = emission_probs.shape[1]
        self.device = device
        self.probs = nn.Parameter(
            torch.Tensor(self.n_states, self.n_observations).type(dtype),
            requires_grad=False,
        )
        self.reset_parameters(emission_probs)
        self.to(self.device)

    def reset_parameters(self, emission_probs):
        emission_probs = input_to_tensor(emission_probs, dtype=self.dtype)
        self.probs.data = emission_probs

    def log_prob(self, x):
        return torch.log(self.probs[x])

    def prob(self, x):
        return self.probs[x]

    def estimate_emissions(self, observation_sequences, obs_gammas):
        sum_marginal = torch.zeros([self.n_states], dtype=self.dtype).to(self.device)

        sum_emission_scores = torch.zeros(
            [self.n_observations, self.n_states], dtype=self.dtype
        ).to(self.device)

        for x, gammas in zip(observation_sequences, obs_gammas):
            sum_marginal += gammas.sum(0)

            # One hot encoding buffer that you create out of the loop and just keep reusing
            seq_one_hot = torch.zeros(
                [x.size(0), self.n_observations], dtype=self.dtype
            ).to(self.device)

            seq_one_hot.scatter_(1, x.unsqueeze(1), 1)
            emission_scores = torch.matmul(seq_one_hot.transpose_(1, 0), gammas)
            sum_emission_scores += emission_scores

        return sum_emission_scores / sum_marginal

    def m_step(self, x, gamma):
        emissions = self.estimate_emissions(x, gamma)

        return emissions

    def step(self, x, gamma):
        emissions = self.m_step(x, gamma)
        converged = self.has_converged(self.probs, emissions)
        self.update_(probs=emissions)

        return converged

    def has_converged(self, prev_emissions, new_emissions):
        delta = torch.max(torch.abs(new_emissions - prev_emissions)).item()

        return delta < self.eps

    def update_(self, probs=None):
        if probs is not None:
            self.probs.data = probs

    def forward(self, x):
        return self.log_prob(x)


class GaussianEmissionModel(nn.Module):
    def __init__(
        self, n_states, n_feats, mu=None, sigma=None, dtype=torch.float, device="cpu"
    ):
        super(GaussianEmissionModel, self).__init__()
        self.eps = 1e-6
        self.dtype = dtype
        self.n_feats = n_feats
        self.n_states = n_states
        self.device = device

        self.mu = nn.Parameter(
            torch.Tensor(self.n_states, self.n_feats).type(dtype), requires_grad=False
        )
        self.sigma = nn.Parameter(
            torch.Tensor(self.n_states, self.n_feats, self.n_feats).type(dtype),
            requires_grad=False,
        )

        self.reset_parameters(mu=mu, sigma=sigma)
        self.to(self.device)

    def reset_parameters(self, mu=None, sigma=None):
        if mu is not None:
            self.mu.data = mu
        else:
            self.mu.normal_()

        if sigma is not None:
            self.sigma.data = sigma
        else:
            x = torch.rand(*self.sigma.shape)
            s = torch.matmul(x, x.transpose(-1, -2)) + 1e-3
            self.sigma.data = s

    def log_prob(self, x):
        """
        x: N x n_features
        returns: N x n_states
        """
        identities = (
            torch.eye(self.n_feats).repeat(self.n_states, 1, 1).to(self.device)
        )
        L = torch.cholesky(self.sigma, upper=False)
        L_inv, _ = torch.triangular_solve(L, identities, upper=False)
        precisions = L_inv.contiguous()  # Inverse sigma

        mu = self.mu

        exp_term = torch.zeros(x.size(0), self.n_states).to(self.device)

        for k in range(self.n_states):
            y = torch.mm(x, precisions[k]) - torch.mv(precisions[k], mu[k])
            exp_term[:, k] = torch.sum(y * y, dim=1)

        log_det = torch.sum(
            torch.log(precisions.view(self.n_states, -1)[:, :: self.n_feats + 1]),
            dim=1,
        )

        logp = -0.5 * (self.n_feats * math.log(2 * math.pi) + exp_term) + log_det

        return logp

    def prob(self, x):
        logp = self.log_prob(x)

        return torch.exp(logp)

    def estimate_mu(self, observation_sequences, obs_gammas, marginal):
        sum_mu = torch.zeros((self.n_states, self.n_feats), dtype=self.dtype).to(
            self.device
        )

        for x, gamma in zip(observation_sequences, obs_gammas):
            # x -> N x n_feats
            # gamma -> N x n_states
            sum_mu += torch.matmul(gamma.t(), x)

        return sum_mu / marginal.unsqueeze(1)

    def estimate_sigma(self, observation_sequences, obs_gammas, mu, marginal):
        sum_sigma = torch.zeros(
            (self.n_states, self.n_feats, self.n_feats), dtype=self.dtype
        ).to(self.device)

        for x, gamma in zip(observation_sequences, obs_gammas):
            # x -> N x n_feats
            # gamma -> N x n_states
            # mu -> n_states x n_feats
            diff = x.unsqueeze(1) - mu.unsqueeze(0)  # N x n_states x n_feats
            # outer prod -> N x n_states x n_feats x n_feats
            outer = torch.einsum("nsi, nsj -> nsij", diff, diff)
            outer = outer * gamma.unsqueeze(-1).unsqueeze(-1)
            outer = outer.sum(0)
            sum_sigma += outer

        sigma = sum_sigma / marginal.unsqueeze(-1).unsqueeze(-1)

        return sigma

    def m_step(self, xs, gammas):
        marginal = torch.zeros((self.n_states), dtype=self.dtype).to(self.device)
        for g in gammas:
            marginal += g.sum(0)

        mu = self.estimate_mu(xs, gammas, marginal)
        sigma = self.estimate_sigma(xs, gammas, mu, marginal)

        return mu, sigma

    def step(self, xs, gammas):
        mu, sigma = self.m_step(xs, gammas)
        converged_mu = self.has_converged(self.mu, mu)
        converged_sigma = self.has_converged(self.sigma, sigma)
        self.update_(mu=mu, sigma=sigma)

        return converged_mu and converged_sigma

    def has_converged(self, prev_emissions, new_emissions):
        delta = torch.max(torch.abs(new_emissions - prev_emissions)).item()
        print(delta)
        return delta < self.eps

    def update_(self, mu=None, sigma=None):
        if mu is not None:
            self.mu.data = mu

        if sigma is not None:
            self.sigma.data = sigma

    def forward(self, x):
        return self.log_prob(x)
