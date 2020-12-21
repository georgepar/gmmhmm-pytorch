import numpy as np
import pandas as pd
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from emission import DiagonalGaussianEmissionModel as GaussianEmissionModel
from hmm import HMM


def sample_gaussian(which_state, mu, sigma):
    """
    sigmas = torch.tensor (n_states * n_features * n_features)
    mus = torch.tensor (n_states * n_features)
    """
    sig = sigma[which_state, :, :]
    mu = mu[which_state, :]

    mvn = MultivariateNormal(mu, covariance_matrix=sig)

    return mvn.sample()


def reset_sigma(n_states, n_feats):
    s = torch.eye(n_feats, n_feats) * random.random()
    s = s.unsqueeze(0).repeat(n_states, 1, 1)
    return s


def generate_HMM_observation(num_obs, n_feats, pi0, T, mu, sigma):
    obs = torch.zeros((num_obs, n_feats))
    states = torch.zeros(num_obs, dtype=torch.long)
    states[0] = torch.multinomial(pi0, 1)
    obs[0] = sample_gaussian(states[0], mu, sigma)

    for t in range(1, num_obs):
        states[t] = torch.multinomial(T[states[t - 1], :], 1)
        obs[t] = sample_gaussian(states[t], mu, sigma)

    return obs


def train_single_observation():
    n_states, n_feats = 2, 3
    True_pi0 = torch.tensor([0.7, 0.3])
    True_A = torch.tensor([[0.75, 0.25], [0.12, 0.88]])
    True_mu = torch.rand(n_states, n_feats)
    True_sigma = reset_sigma(n_states, n_feats)
    obs_seq = generate_HMM_observation(
        200, n_feats, True_pi0, True_A, True_mu, True_sigma
    )

    emission_model = GaussianEmissionModel(n_states, n_feats, device="cpu")
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
    print("True mu: ")
    print(True_mu)
    print()
    print("mu: ")
    print(emission_model.mu)
    print("*" * 60)
    print("True sigma: ")
    print(True_sigma)
    print()
    print("sigma: ")
    print(emission_model.sigma)
    print("*" * 60)

    print("Reached Convergence: ")
    print(model.converged_)
    print("*" * 60)


def train_multi_observation():
    n_states, n_feats = 2, 3
    True_pi0 = torch.tensor([0.7, 0.3])
    True_A = torch.tensor([[0.75, 0.25], [0.12, 0.88]])
    True_mu = torch.rand(n_states, n_feats)
    True_sigma = reset_sigma(n_states, n_feats)
    obs_seq = [generate_HMM_observation(
        200, n_feats, True_pi0, True_A, True_mu, True_sigma
    ).to("cpu") for _ in range(100)]

    emission_model = GaussianEmissionModel(n_states, n_feats, device="cpu")
    model = HMM(
        emission_model,
        n_states=n_states,
        device="cpu",
        n_iter=1000,
        dtype=torch.float,
    ).fit(obs_seq)

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
    print("True mu: ")
    print(True_mu)
    print()
    print("mu: ")
    print(emission_model.mu)
    print("*" * 60)
    print("True sigma: ")
    print(True_sigma)
    print()
    print("sigma: ")
    print(emission_model.sigma)
    print("*" * 60)

    print("Reached Convergence: ")
    print(model.converged_)
    print("*" * 60)


if __name__ == "__main__":
    #train_single_observation()
    print("=" * 60)
    train_multi_observation()
