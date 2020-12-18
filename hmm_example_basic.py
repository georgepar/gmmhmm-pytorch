import numpy as np
import pandas as pd
import torch

from hmm import HMM, DiscreteEmissionModel


def dptable(state_prob, states):
    print(" ".join(("%8d" % i) for i in range(state_prob.shape[0])))
    for i, prob in enumerate(state_prob.T):
        print("%.7s: " % states[i] + " ".join("%.7s" % ("%f" % p) for p in prob))


def viterbi_example():
    p0 = np.array([0.6, 0.4])
    emi = np.array([[0.5, 0.1], [0.4, 0.3], [0.1, 0.6]])
    trans = np.array([[0.7, 0.3], [0.4, 0.6]])

    states = {0: "Healthy", 1: "Fever"}
    obs = {0: "normal", 1: "cold", 2: "dizzy"}

    obs_seq = np.array([0, 0, 1, 2, 2])

    emission_model = DiscreteEmissionModel(emi, device="cpu")
    model = HMM(emission_model, n_states=2, A=trans, pi0=p0, device="cpu")

    states_seq, state_prob = model.viterbi(obs_seq)

    print("Observation sequence: ", [obs[o] for o in obs_seq])
    df = pd.DataFrame(torch.t(state_prob).cpu().numpy(), index=["Healthy", "Fever"])
    print(df)
    print(states_seq)
    print("Most likely States: ", [states[s.item()] for s in states_seq])

def forward_backward_example():

    p0 = np.array([0.5, 0.5])
    emi = np.array([[0.9, 0.2], [0.1, 0.8]])
    trans = np.array([[0.7, 0.3], [0.3, 0.7]])

    states = {0: "rain", 1: "no_rain"}
    obs = {0: "umbrella", 1: "no_umbrella"}

    obs_seq = np.array([1, 1, 0, 0, 0, 1])

    emission_model = DiscreteEmissionModel(emi, device="cpu")
    model = HMM(emission_model, n_states=2, A=trans, pi0=p0, device="cpu")

    obs_prob_seq = emission_model.prob(obs_seq)
    alphas, betas = model.forward_backward_alg(obs_prob_seq)

    posterior = alphas * betas
    # marginal per timestep
    marginal = torch.sum(posterior, 1)

    # Normalize porsterior into probabilities
    posterior = posterior / marginal.view(-1, 1)

    results = {
        "Forward": alphas.cpu().numpy(),
        "Backward": betas.cpu().numpy(),
        "Posterior": posterior.cpu().numpy(),
    }

    for k, v in results.items():
        inferred_states = np.argmax(v, axis=1)
        print()
        print(k)
        dptable(v, states)
        print()

    print("*" * 60)
    print("Most likely Final State: ", states[inferred_states[-1]])
    print("*" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("Running viterbi example:")
    viterbi_example()
    print("=" * 60)
    print("=" * 60)
    print("Running forward_backward example:")
    forward_backward_example()
