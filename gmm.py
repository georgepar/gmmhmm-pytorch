import math

import torch
import torch.nn as nn
from sklearn import cluster


class GMMBase(nn.Module):
    """Gaussian mixture model with torch operations"""

    def __init__(
        self,
        n_mixtures,
        n_features,
        init="random",
        device="cpu",
        n_iter=1000,
        delta=1e-3,
        warm_start=False
    ):
        super(GMMBase, self).__init__()
        self.device = device
        self.n_mixtures = n_mixtures
        self.n_features = n_features

        if init not in ["kmeans", "random"]:
            raise ValueError("init can be kmeans or random")
        self.init = init

        self.mu = nn.Parameter(torch.Tensor(n_mixtures, n_features), requires_grad=False)
        self.sigma = None  # To be initialized by subclasses
        self.pi = nn.Parameter(torch.Tensor(n_mixtures), requires_grad=False)
        self.converged_ = False
        self.eps = 1e-6
        self.delta = delta
        self.warm_start = warm_start
        self.n_iter = n_iter

    def reset_parameters(self, x=None):
        if self.init == "random" or x is None:
            self.mu.normal_()
            self.reset_sigma()
            self.pi.fill_(1.0 / self.n_mixtures)  # Uniform mixture weights
        elif self.init == "kmeans":
            centroids = (
                cluster.KMeans(n_clusters=self.n_mixtures, n_init=1)
                .fit(x)
                .cluster_centers_
            )
            centroids = torch.tensor(centroids).to(self.device)
            self.update_(mu=centroids)

    def reset_sigma(self):
        raise NotImplementedError(
            "You must use one of the GMM subclasses (DiagonalCovarianceGMM, FullCovarianceGMM)"
        )

    def estimate_precisions(self):
        raise NotImplementedError(
            "You must use one of the GMM subclasses (DiagonalCovarianceGMM, FullCovarianceGMM)"
        )

    def log_prob(self, x):
        raise NotImplementedError(
            "You must use one of the GMM subclasses (DiagonalCovarianceGMM, FullCovarianceGMM)"
        )

    def weighted_log_prob(self, x):
        logp = self.log_prob(x)
        wlogp = logp + torch.log(self.pi)

        return wlogp

    def log_likelihood(self, x):
        wlogp = self.weighted_log_prob(x)
        per_sample_log_likelihood = torch.logsumexp(wlogp, dim=1)

        return per_sample_log_likelihood.sum()

    def e_step(self, x):
        wlogp = self.weighted_log_prob(x).unsqueeze(-1)
        log_likelihood = torch.logsumexp(wlogp, dim=1, keepdim=True)
        Q = wlogp - log_likelihood

        return Q.squeeze()

    def estimate_mu(self, x, pi, responsibilities):
        nk = pi * x.size(0)
        mu = torch.sum(responsibilities * x, dim=0, keepdim=True) / nk

        return mu

    def estimate_pi(self, x, responsibilities):
        pi = torch.sum(responsibilities, dim=0, keepdim=True) + self.eps
        pi = pi / x.size(0)

        return pi

    def m_step(self, x, Q):
        x = x.unsqueeze(1)

        resp = torch.exp(Q).unsqueeze(-1)
        pi = self.estimate_pi(x, resp)
        mu = self.estimate_mu(x, pi, resp)
        sigma = self.estimate_sigma(x, mu, pi, resp)

        pi = pi.squeeze()
        mu = mu.squeeze()
        sigma = sigma.squeeze()

        return pi, mu, sigma

    def step(self, x):
        Q = self.e_step(x)
        prev_ll = self.log_likelihood(x)
        pi, mu, sigma = self.m_step(x, Q)
        self.update_(pi=pi, mu=mu, sigma=sigma)
        new_ll = self.log_likelihood(x)

        return self.has_converged(prev_ll, new_ll)

    def has_converged(self, prev_ll, new_ll):
        if (new_ll.abs().item() == float("Inf")) or (new_ll == float("nan")):
            self.reset_parameters()
            print("GMM diverged. Reseting")
            self.reset_parameters()

            return False

        return new_ll - prev_ll < self.delta

    def estimate_sigma(self, x, mu, pi, responsibilities):
        raise NotImplementedError(
            "You must use one of the GMM subclasses (DiagonalCovarianceGMM, FullCovarianceGMM)"
        )

    def update_(self, mu=None, sigma=None, pi=None):
        if mu is not None:
            self.mu.data = mu

        if sigma is not None:
            self.sigma.data = sigma

        if pi is not None:
            self.pi.data = pi

    def fit(self, x, y=None):
        if not self.warm_start:
            if self.converged_:
                self.reset_parameters(x=x)

        itr = 0

        while (itr < self.n_iter + 1):

            has_converged = self.step(x)

            if has_converged:
                break
            itr += 1

        self.converged_ = True

        return self

    def forward(self, x):
        p = self.weighted_log_prob(x)

        return p


class DiagonalCovarianceGMM(GMMBase):
    """Gaussian mixture model with torch operations"""

    def __init__(self, n_mixtures, n_features, init="random", device="cpu", n_iter=1000, delta=1e-3,
                 warm_start=False):
        super(DiagonalCovarianceGMM, self).__init__(n_mixtures, n_features, init=init,
                                                    device=device, n_iter=n_iter, delta=delta,
                                                    warm_start=warm_start)

        self.sigma = nn.Parameter(
            torch.Tensor(n_mixtures, n_features), requires_grad=False
        )
        self.reset_parameters()
        self.to(self.device)

    def reset_sigma(self):
        self.sigma.fill_(1)

    def estimate_precisions(self):
        return torch.rsqrt(self.sigma)

    def log_prob(self, x):
        precisions = self.estimate_precisions()

        x = x.unsqueeze(1)
        mu = self.mu.unsqueeze(0)
        precisions = precisions.unsqueeze(0)

        # This is outer product
        exp_term = torch.sum(
            (mu * mu + x * x - 2 * x * mu) * (precisions ** 2), dim=2, keepdim=True
        )
        log_det = torch.sum(torch.log(precisions), dim=2, keepdim=True)

        logp = -0.5 * (self.n_features * math.log(2 * math.pi) + exp_term) + log_det

        return logp.squeeze()

    def estimate_sigma(self, x, mu, pi, responsibilities):
        nk = pi * x.size(0)
        x2 = (responsibilities * x * x).sum(0, keepdim=True) / nk
        mu2 = mu * mu
        xmu = (responsibilities * mu * x).sum(0, keepdim=True) / nk
        sigma = x2 - 2 * xmu + mu2 + self.eps

        return sigma



class FullCovarianceGMM(GMMBase):
    """Gaussian mixture model with torch operations"""
    def __init__(self, n_mixtures, n_features, init="random", device="cpu", n_iter=1000, delta=1e-3,
                 warm_start=False):
        super(FullCovarianceGMM, self).__init__(n_mixtures, n_features, init=init,
                                                device=device, n_iter=n_iter, delta=delta,
                                                warm_start=warm_start)

        self.sigma = nn.Parameter(
            torch.Tensor(n_mixtures, n_features, n_features), requires_grad=False
        )
        self.reset_parameters()
        self.to(self.device)

    def reset_sigma(self):
        x = torch.rand(*self.sigma.shape)
        s = torch.matmul(x, x.transpose(-1, -2)) + 1e-3
        self.sigma.data = s

    def estimate_precisions(self):
        identities = (torch
                      .eye(self.n_features)
                      .unsqueeze(0)
                      .repeat(self.n_mixtures, 1, 1)
                      .to(self.device))
        L = torch.cholesky(self.sigma, upper=False)
        L_inv, _ = torch.triangular_solve(L, identities, upper=False)

        return L_inv.contiguous()

    def log_prob(self, x):
        precisions = self.estimate_precisions()

        mu = self.mu
        exp_term = torch.zeros(x.size(0), self.n_mixtures).to(self.device)

        for k in range(self.n_mixtures):
            y = torch.mm(x, precisions[k]) - torch.mv(precisions[k], mu[k])
            exp_term[:, k] = torch.sum(y * y, dim=1)

        log_det = torch.sum(
            torch.log(
                precisions.view(self.n_mixtures, -1)[:, ::self.n_features + 1]
            ), dim=1
        )

        logp = -0.5 * (self.n_features * math.log(2 * math.pi) + exp_term) + log_det

        return logp.squeeze()

    def estimate_sigma(self, x, mu, pi, responsibilities):
        x = x.squeeze()
        mu = mu.squeeze()
        pi = pi.squeeze()

        nk = pi * x.size(0)
        n_mixtures, n_features = self.n_mixtures, self.n_features
        sigma = torch.zeros((n_mixtures, n_features, n_features)).to(self.device)
        responsibilities = responsibilities.squeeze()

        for k in range(n_mixtures):
            diff = x - mu[k]
            sigma[k] = torch.mm(responsibilities[:, k] * diff.t(), diff) / nk[k]
            sflat = sigma[k].view(sigma[k].numel())
            sflat[::n_features + 1] += 1e-6  # Add to diagonal to ensure positive definite
            sigma[k] = sflat.view(n_features, n_features)

        return sigma


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
        warm_start=False
    ):
        super(GMMEmissionModel, self).__init__()
        gmm_cls = {
            "diag": DiagonalCovarianceGMM,
            "diagonal": DiagonalCovarianceGMM,
            "full": FullCovarianceGMM
        }

        if covariance_type not in gmm_cls.keys():
            raise ValueError(
                "covariance_type can only be one of {}"
                .format(list(gmm_cls.keys()))
            )

        self.n_states = n_states
        self.n_mixtures = n_mixtures
        self.n_features = n_features
        self.device = device

        self.gmms = [
            gmm_cls["covariance_type"](
                n_mixtures, n_features, init=init, device=device,
                n_iter=n_iter, delta=delta, warm_start=warm_start
            )

            for _ in range(n_states)
        ]

    def observation_prob(self, x):
        wlogp = torch.stack([gmm.weighted_log_prob(x) for gmm in self.gmms])
        assignment_probs = torch.exp(wlogp) # n_states * n_mixtures

        return assignment_probs / assignment_probs.sum(1)

    def estimate_new_gammas(self, observation_sequences, obs_gammas):
        new_gammas = []

        for x, gammas in zip(observation_sequences, obs_gammas):
            probs = self.observation_prob(x).transpose(1, 0, 2) # N * n_states * n_mixtures
            new_gammas.append(probs / gammas.unsqueeze(0))

        return new_gammas

    def estimate_pi(self, obs_gammas, new_gammas):
        sum_pi = torch.zeros(
            [self.n_states, self.n_mixtures], dtype=torch.float64
        ).to(self.device)

        for g, new_g in zip(obs_gammas, new_gammas):
            num = new_g.sum(0)
            denom = g.sum(0).unsqueeze(0)
            pi = num / denom
            sum_pi += pi

        return sum_pi

    def estimate_mu(self, observation_sequences, new_gammas):
        sum_mu = torch.zeros(
            [self.n_states, self.n_mixtures, self.n_features], dtype=torch.float64 
        ).to(self.device)

        for x, new_g in zip(observation_sequences, new_gammas):
            # x -> N * n_features
            # new_g -> N * n_states * n_mixtures
            num = torch.matmul(new_g.transpose(1, 2, 0), x)
            denom = new_g.sum(0)
            mu = num / denom.unsqueeze(0)
            sum_mu += mu

        return sum_mu

    def estimate_sigma(self, observation_sequences, new_gammas, mu):
        sum_sigma = torch.zeros(
            [self.n_states, self.n_mixtures, self.n_features, self.n_features],
            dtype=torch.float64
        ).to(self.device)
