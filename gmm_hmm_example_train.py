import numpy as np
import pandas as pd
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from gmm import GMMEmissionModel
from hmm import HMM


def sample_gmm(which_state, pi, mu, sigma):
    """
    pis = torch.tensor (n_mixtures)
    sigmas = torch.tensor (n_mixtures * n_features * n_features)
    mus = torch.tensor (n_mixtures * n_features * n_features)
    """
    which_mixture = torch.multinomial(pi[which_state], 1)
    sig = sigma[which_state, which_mixture, :, :]
    mu = mu[which_state, which_mixture, :]

    mvn = MultivariateNormal(mu, covariance_matrix=sig)

    return mvn.sample()

def reset_sigma(n_states, n_mixtures, n_feats):
    x = torch.rand(n_states, n_mixtures, n_feats, n_feats)
    s = torch.matmul(x, x.transpose(-1, -2)) + 1e-3
    return s


def generate_HMM_observation(num_obs, n_feats, pi0, T, pi, mu, sigma):
    obs = torch.zeros((num_obs, n_feats))
    states = torch.zeros(num_obs, dtype=torch.long)
    states[0] = torch.multinomial(pi0, 1)
    obs[0] = sample_gmm(states[0], pi, mu, sigma)

    for t in range(1, num_obs):
        states[t] = torch.multinomial(T[states[t - 1], :], 1)
        obs[t] = sample_gmm(states[t], pi, mu, sigma)

    return obs


def train_single_observation():
    n_states, n_mixtures, n_feats = 3, 2, 4
    True_pi0 = torch.tensor([.7, .2, .1])
    True_A = torch.tensor([[0.75, 0.15, 0.1], [0.12, 0.68, 0.2], [0.4, 0.3, 0.3]])
    True_pi = torch.tensor([[0.8, 0.2], [0.5, 0.5], [0.25, 0.75]])
    True_mu = torch.rand(n_states, n_mixtures, n_feats)
    True_sigma = reset_sigma(n_states, n_mixtures, n_feats)
    obs_seq = generate_HMM_observation(
        50, n_feats, True_pi0, True_A, True_pi, True_mu, True_sigma
    )


    emission_model = GMMEmissionModel(n_mixtures, n_feats, n_states, device="cpu",
                                      covariance_type="full")
    model = HMM(
        emission_model,
        n_states=n_states,
        device="cpu",
        n_iter=1000,
        dtype=torch.float,
    ).fit([obs_seq])

    print("*" * 60)
    print("True Transition Matrix: ")
    print(True_A)
    print()
    print("Transition Matrix: ")
    print(model.A)
    print("*" * 60)
    print("True state priors: ")
    print(True_pi0)
    print()
    print("State Priors: ")
    print(model.pi0)
    print("*" * 60)
    print("True mixture weights: ")
    print(True_pi)
    print()
    print("Mixture weights: ")
    print([gmm.pi for gmm in emission_model.gmms])
    print("*" * 60)
    print("True mu: ")
    print(True_mu)
    print()
    print("mu: ")
    print([gmm.mu for gmm in emission_model.gmms])
    print("*" * 60)
    print("True sigma: ")
    print(True_sigma)
    print()
    print("sigma: ")
    print([gmm.sigma for gmm in emission_model.gmms])
    print("*" * 60)

    print("Reached Convergence: ")
    print(model.converged_)
    print("*" * 60)


if __name__ == "__main__":
    train_single_observation()
    print("=" * 60)
