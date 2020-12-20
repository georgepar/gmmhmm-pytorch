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

    def e_step(self, x):
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
