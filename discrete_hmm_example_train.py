import numpy as np
import pandas as pd
import torch

from hmm import HMM, DiscreteEmissionModel


def generate_HMM_observation(num_obs, pi, T, E):
    def drawFrom(probs):
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]

    obs = np.zeros(num_obs)
    states = np.zeros(num_obs)
    states[0] = drawFrom(pi)
    obs[0] = drawFrom(E[:, int(states[0])])

    for t in range(1, num_obs):
        states[t] = drawFrom(T[int(states[t - 1]), :])
        obs[t] = drawFrom(E[:, int(states[t])])

    obs = torch.from_numpy(obs).long().to("cuda")

    return obs


def train_single_observation():
    True_pi = np.array([0.5, 0.5])
    True_T = np.array([[0.85, 0.15], [0.12, 0.88]])
    True_E = np.array([[0.8, 0.0], [0.1, 0.0], [0.1, 1.0]])

    obs_seq = generate_HMM_observation(200, True_pi, True_T, True_E)

    init_pi = np.array([0.5, 0.5])
    init_T = np.array([[0.5, 0.5], [0.5, 0.5]])
    init_E = np.array([[0.3, 0.2], [0.3, 0.5], [0.4, 0.3]])

    emission_model = DiscreteEmissionModel(init_E, device="cuda", dtype=torch.float64)
    model = HMM(
        emission_model,
        n_states=2,
        A=init_T,
        pi0=init_pi,
        device="cuda",
        n_iter=1000,
        dtype=torch.float64,
    ).fit([obs_seq])

    print("*" * 60)
    print("Transition Matrix: ")
    print(model.A)
    print()
    print("Emission Matrix: ")
    print(emission_model.probs)
    print()
    print("Priors: ")
    print(model.pi0)
    print("Reached Convergence: ")
    print(model.converged_)
    print("*" * 60)


def train_multi_observation():
    True_pi = np.array([0.5, 0.5])
    True_T = np.array([[0.85, 0.15], [0.12, 0.88]])
    True_E = np.array([[0.8, 0.0], [0.1, 0.0], [0.1, 1.0]])

    obs_seqs = [
        generate_HMM_observation(200, True_pi, True_T, True_E) for _ in range(100)
    ]

    init_pi = np.array([0.5, 0.5])
    init_T = np.array([[0.5, 0.5], [0.5, 0.5]])
    init_E = np.array([[0.3, 0.2], [0.3, 0.5], [0.4, 0.3]])

    emission_model = DiscreteEmissionModel(init_E, device="cuda", dtype=torch.float64)
    model = HMM(
        emission_model,
        n_states=2,
        A=init_T,
        pi0=init_pi,
        device="cuda",
        n_iter=1000,
        dtype=torch.float64,
    ).fit(obs_seqs)

    print("*" * 60)
    print("Transition Matrix: ")
    print(model.A)
    print()
    print("Emission Matrix: ")
    print(emission_model.probs)
    print()
    print("Priors: ")
    print(model.pi0)
    print("Reached Convergence: ")
    print(model.converged_)
    print("*" * 60)


if __name__ == "__main__":
    train_single_observation()
    print("=" * 60)
    train_multi_observation()
