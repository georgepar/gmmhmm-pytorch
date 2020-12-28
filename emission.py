import math

import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from gmm import DiagonalCovarianceGMM, FullCovarianceGMM
from util import input_to_tensor, is_close


class DiscreteEmissionModel(nn.Module):
    def __init__(self, emission_probs, dtype=torch.float, device="cpu"):
        super(DiscreteEmissionModel, self).__init__()
        self.eps = 1e-3
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

    def prefit(self, x):
        # No real need to prefit this for now.

        return self

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
        return is_close(prev_emissions, new_emissions, eps=self.eps)

    def update_(self, probs=None):
        if probs is not None:
            self.probs.data = probs

    def forward(self, x):
        return self.log_prob(x)


class FullGaussianEmissionModel(nn.Module):
    def __init__(
        self, n_states, n_feats, mu=None, sigma=None, dtype=torch.float, device="cpu"
    ):
        super(FullGaussianEmissionModel, self).__init__()
        self.eps = 1e-3
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

    def estimate_sequence_states(self, x):
        all_sequence_items = torch.cat(x)
        kmu = KMeans(n_clusters=self.n_states, n_init=5).fit(all_sequence_items)
        assigned_states = kmu.predict(all_sequence_items)

        return all_sequence_items, assigned_states

    def fit_initial_state_distributions(self, all_sequence_items, assigned_states):
        for s in range(self.n_states):
            assigned_items = all_sequence_items[assigned_states == s]
            mu_mle = torch.mean(assigned_items, dim=0)
            self.mu[s, :] = mu_mle
            diff = assigned_items - self.mu[s, :].unsqueeze(0)  # N * f
            self.sigma[s, :, :] = (
                1.0 / (assigned_items.size(0) - 1) * torch.matmul(diff.t(), diff)
            )

        return self

    def prefit(self, x):
        """
        x: list of sequences as torch.tensors (seq_length, n_features)
        """
        all_sequence_items, assigned_states = self.estimate_sequence_states(x)
        self = self.fit_initial_state_distributions(all_sequence_items, assigned_states)
        print("Prefited:")
        print(self.mu)
        print(self.sigma)

        return self

    def log_prob(self, x):
        """
        x: N x n_features
        returns: N x n_states
        """
        identities = torch.eye(self.n_feats).repeat(self.n_states, 1, 1).to(self.device)
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
        reg = 1e-3 * torch.eye(self.n_feats, dtype=self.dtype).to(self.device)
        sigma += reg.unsqueeze(0)

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

        print("*" * 60)
        print("New mu: ")
        print(mu)
        print()
        print("*" * 60)
        print("New sigma: ")
        print(sigma)
        print()

        converged_mu = self.has_converged(self.mu, mu)
        converged_sigma = self.has_converged(self.sigma, sigma)
        self.update_(mu=mu, sigma=sigma)

        return converged_mu and converged_sigma

    def has_converged(self, prev_parameter, new_parameter):
        return is_close(prev_parameter, new_parameter, eps=self.eps)

    def update_(self, mu=None, sigma=None):
        if mu is not None:
            self.mu.data = mu

        if sigma is not None:
            self.sigma.data = sigma

    def forward(self, x):
        return self.log_prob(x)


class DiagonalGaussianEmissionModel(nn.Module):
    def __init__(
        self, n_states, n_feats, mu=None, sigma=None, dtype=torch.float, device="cpu"
    ):
        super(DiagonalGaussianEmissionModel, self).__init__()
        self.eps = 1e-6
        self.dtype = dtype
        self.n_feats = n_feats
        self.n_states = n_states
        self.device = device

        self.mu = nn.Parameter(
            torch.Tensor(self.n_states, self.n_feats).type(dtype), requires_grad=False
        )
        self.sigma = nn.Parameter(
            torch.Tensor(self.n_states, self.n_feats).type(dtype),
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
            self.sigma.fill_(1)

    def estimate_sequence_states(self, x):
        all_sequence_items = torch.cat(x)
        kmu = KMeans(n_clusters=self.n_states, n_init=5).fit(all_sequence_items)
        assigned_states = kmu.predict(all_sequence_items)

        return all_sequence_items, assigned_states

    def fit_initial_state_distributions(self, all_sequence_items, assigned_states):
        for s in range(self.n_states):
            assigned_items = all_sequence_items[assigned_states == s]
            mu_mle = torch.mean(assigned_items, dim=0)
            var_mle = torch.var(assigned_items, dim=0)
            self.mu[s, :] = mu_mle
            self.sigma[s, :] = var_mle

        return self

    def prefit(self, x):
        """
        x: list of sequences as torch.tensors (seq_length, n_features)
        """
        all_sequence_items, assigned_states = self.estimate_sequence_states(x)
        self = self.fit_initial_state_distributions(all_sequence_items, assigned_states)

        return self

    def log_prob(self, x):
        """
        x: N x n_features
        returns: N x n_states
        """
        precisions = torch.rsqrt(self.sigma)  # n_states x n_features
        mu = self.mu  # n_states x n_features

        exp_term = torch.zeros(x.size(0), self.n_states).to(self.device)

        x = x.unsqueeze(1)  # N x 1 x n_features
        mu = self.mu.unsqueeze(0)  # 1 x n_states x n_features
        precisions = precisions.unsqueeze(0)  # 1 x n_states x n_features

        # This is outer product
        exp_term = torch.sum(
            (mu * mu + x * x - 2 * x * mu) * (precisions ** 2), dim=2, keepdim=True
        )  #
        log_det = torch.sum(torch.log(precisions), dim=2, keepdim=True)

        logp = -0.5 * (self.n_feats * math.log(2 * math.pi) + exp_term) + log_det

        return logp.squeeze()  # N x n_states

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
            # sum_mu += torch.matmul(gamma.t(), x)
            sum_mu += torch.einsum("ns, nf -> sf", gamma, x)

        return sum_mu / marginal.unsqueeze(1)

    def estimate_sigma(self, observation_sequences, obs_gammas, mu, marginal):
        sum_sigma = torch.zeros((self.n_states, self.n_feats), dtype=self.dtype).to(
            self.device
        )

        for x, gamma in zip(observation_sequences, obs_gammas):
            # x -> N x n_feats
            # gamma -> N x n_states
            # mu -> n_states x n_feats
            diff = x.unsqueeze(1) - mu.unsqueeze(0)  # N x s x f
            sum_sigma += (gamma.unsqueeze(-1) * diff * diff).sum(0)

        sigma = sum_sigma / marginal.unsqueeze(-1)

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

    def has_converged(self, prev_parameter, new_parameter):
        return is_close(prev_parameter, new_parameter, eps=self.eps)

    def update_(self, mu=None, sigma=None):
        if mu is not None:
            self.mu.data = mu

        if sigma is not None:
            self.sigma.data = sigma

    def forward(self, x):
        return self.log_prob(x)


class GMMEmissionModel(nn.Module):
    def __init__(
        self,
        n_mixtures,
        n_features,
        n_states,
        init="random",
        device="cpu",
        covariance_type="diagonal",
        n_iter=1000,
        delta=1e-3,
        warm_start=False,
    ):
        super(GMMEmissionModel, self).__init__()
        gmm_cls = {
            "diag": DiagonalCovarianceGMM,
            "diagonal": DiagonalCovarianceGMM,
            "full": FullCovarianceGMM,
        }

        if covariance_type not in gmm_cls.keys():
            raise ValueError(
                "covariance_type can only be one of {}".format(list(gmm_cls.keys()))
            )
        self.n_states = n_states
        self.n_mixtures = n_mixtures
        self.n_features = n_features
        self.device = device
        self.covariance_type = covariance_type
        self.gmms = [
            gmm_cls[covariance_type](
                n_mixtures,
                n_features,
                init=init,
                device=device,
                n_iter=n_iter,
                delta=delta,
                warm_start=warm_start,
            )

            for _ in range(n_states)
        ]

    def assignment_prob(self, x):
        wlogp = torch.stack([gmm.weighted_log_prob(x) for gmm in self.gmms])

        if wlogp.ndim == 2:
            wlogp = wlogp.unsqueeze(1)
        assignment_probs = (
            torch.exp(wlogp).permute(1, 0, 2).contiguous()
        )  # -> N * n_states * n_mixtures

        return assignment_probs / assignment_probs.sum(1).unsqueeze(1)

    def prob(self, x):
        wlogp = torch.stack([gmm.weighted_log_prob(x) for gmm in self.gmms])

        if wlogp.ndim == 2:
            wlogp = wlogp.unsqueeze(1)
        assignment_probs = (
            torch.exp(wlogp).permute(1, 0, 2).contiguous()
        )  # -> N * n_states * n_mixtures

        probs = []

        for s in range(self.n_states):
            probs.append(assignment_probs[:, s, :].matmul(self.gmms[s].pi))

        probs = torch.stack(probs, -1)

        return probs

    def estimate_new_gammas(self, x, gammas):
        probs = self.assignment_prob(x)  # N * n_states * n_mixtures
        new_g = probs / gammas.unsqueeze(-1)

        return new_g

    def estimate_pi(self, obs_gammas, new_gammas):
        sum_nominator = torch.zeros(
            [self.n_states, self.n_mixtures], dtype=torch.float64
        ).to(self.device)

        sum_denom = torch.zeros([self.n_states], dtype=torch.float64).to(self.device)

        for g, new_g in zip(obs_gammas, new_gammas):
            sum_nominator += new_g.sum(0)  # sum along N
            sum_denom += g.sum(0)  # sum along N

        pi = sum_nominator / sum_denom.unsqueeze(-1)

        return pi  # n_states * n_mixtures

    def estimate_mu(self, observation_sequences, new_gammas, marginal):
        # marginal -> sum_observations(sum_time(new_gammas)) -> (n_states * n_mixtures)

        sum_nominator = torch.zeros(
            [self.n_states, self.n_mixtures, self.n_features], dtype=torch.float64
        ).to(self.device)

        for x, new_g in zip(observation_sequences, new_gammas):
            # x -> N * n_features
            # new_g: N * n_states * n_mixtures -> n_states * n_mixtures * N
            num = torch.matmul(new_g.permute(1, 2, 0).contiguous(), x)
            sum_nominator += num

        mu = sum_nominator / marginal.unsqueeze(-1)

        return mu

    def estimate_sigma(self, observation_sequences, new_gammas, mu, marginal):
        # marginal -> sum_observations(sum_time(new_gammas)) -> (n_states * n_mixtures)

        # TODO: implement Sigma update rule
        sum_num = torch.zeros(
            [self.n_states, self.n_mixtures, self.n_features, self.n_features],
            dtype=torch.float64,
        ).to(self.device)

        for x, new_g in zip(observation_sequences, new_gammas):
            # x -> N * n_features
            # mu -> n_states * n_mixtures * n_features
            N = x.size(0)
            mu_rep = mu.unsqueeze(0).repeat(N, 1, 1, 1)
            diff = (
                x.view(N, 1, 1, self.n_features) - mu_rep
            )  # N * n_states * n_mixtures * n_features

            # Outer product (x - mu) * (x - mu)^T
            # N * n_states * n_mixtures * n_features * n_features
            xmu2 = torch.einsum("nsmi,nsmj->nsmij", (diff, diff))

            # new_g: N * n_states * n_mixtures
            num = new_g.unsqueeze(-1).unsqueeze(-1) * xmu2
            num = num.sum(0)
            sum_num += num

        sigma = sum_num / marginal.unsqueeze(-1).unsqueeze(-1)

        return sigma

    def m_step(self, x, gammas):
        new_g = self.estimate_new_gammas(x, gammas)

        return new_g

    def step(self, observation_sequences, obs_gammas):
        prev_ll = [
            sum([gmm.log_likelihood(x) for x in observation_sequences])

            for gmm in self.gmms
        ]

        new_obs_gammas = []

        for x, gammas in zip(observation_sequences, obs_gammas):
            new_g = self.m_step(x, gammas)
            new_obs_gammas.append(new_g)

        marginal = torch.zeros(
            [self.n_states, self.n_mixtures], dtype=torch.float64
        ).to(self.device)

        pi = self.estimate_pi(obs_gammas, new_obs_gammas)
        mu = self.estimate_mu(observation_sequences, new_obs_gammas, marginal)
        sigma = self.estimate_sigma(observation_sequences, new_obs_gammas, mu, marginal)

        self.update_(mu=mu, sigma=sigma, pi=pi)

        converged = True

        for i, gmm in enumerate(self.gmms):
            new_ll = sum([gmm.log_likelihood(x) for x in observation_sequences])
            converged = converged and self.has_converged(prev_ll[i], new_ll)

        return converged

    def update_(self, mu=None, sigma=None, pi=None):
        """
        mu: (n_states, n_mixtures, n_features)
        pi: (n_states, n_mixtures)
        sigma: (n_states, n_mixtures, n_features, n_features)
        """

        for state in range(self.n_states):
            if self.covariance_type in ["diag", "diagonal"]:
                s = torch.diag(sigma[state])
            else:
                s = sigma[state]
            self.gmms[state].update_(mu=mu[state], sigma=s, pi=pi[state])
