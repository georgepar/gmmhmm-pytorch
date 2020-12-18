import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gmm import DiagonalCovarianceGMM, FullCovarianceGMM


class DiscreteEmissionModel(nn.Module):
    def __init__(self, emission_probs, device="cpu"):
        super(DiscreteEmissionModel, self).__init__()
        self.eps = 1e-6
        self.n_observations = emission_probs.shape[0]
        self.n_states = emission_probs.shape[1]
        self.device = device
        self.probs = nn.Parameter(emission_probs, requires_grad=False)
        self.to(self.device)

    def log_prob(self, x):
        return torch.log(self.probs[x])

    def prob(self, x):
        return self.probs[x]

    def e_step(self, x):
        return self.probs[x]

    def estimate_emissions(self, observation_sequences, obs_gammas):
        sum_marginal = torch.zeros(
            [self.n_states], dtype=torch.float64
        ).to(self.device)

        sum_emission_scores = torch.zeros(
            [self.n_observations, self.n_states], dtype=torch.float64
        ).to(self.device)

        for x, gammas in zip(observation_sequences, obs_gammas):
            sum_marginal += gammas.sum(0)

            # One hot encoding buffer that you create out of the loop and just keep reusing
            seq_one_hot = torch.zeros(
                [x.size(0), self.n_observations],
                dtype=torch.float64
            ).to(self.device)

            seq_one_hot.scatter_(
                1, torch.tensor(x).unsqueeze(1), 1
            )
            emission_scores = torch.matmul(
                seq_one_hot.transpose_(1, 0), gammas
            )
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
        delta = torch.max(torch.abs(new_emissions - prev_emissions)).item() < self.eps

        return delta < self.eps

    def update_(self, probs=None):
        if probs is not None:
            self.probs.data = probs

    def forward(self, x):
        return self.log_prob(x)


class DiagonalGMMEmissionModel(DiagonalCovarianceGMM):
    pass


class FullGMMEmissionModel(FullCovarianceGMM):
    pass


class HMM(nn.Module):
    """Some Information about HMM"""
    def __init__(self, emission_model, n_states=None, A=None, pi0=None, n_iter=100, device="cpu"):
        super(HMM, self).__init__()
        self.n_iter = n_iter
        self.eps = 1e-6
        self.device = device

        if n_states is None and A is None:
            raise ValueError("You must either pass the number of states or the Transition Matrix")

        if n_states is None:
            n_states = A.shape[0]

        self.A = nn.Parameter(torch.Tensor(n_states, n_states), requires_grad=False)
        self.pi0 = nn.Parameter(torch.Tensor(n_states), requires_grad=False)
        self.emission_model = emission_model
        self.converged_ = False

        self.to(self.device)

    def _forward_alg(self, observation_probs):
        N = observation_probs.size(0)

        alphas = torch.zeros((N, self.n_states)).to(self.device)
        alphas[0] = self.pi0 * observation_probs[0]

        scale = torch.zeros([N], dtype=torch.float64).to(self.device)

        scale[0] = 1.0 / alphas[0].sum()
        alphas[0] = alphas[0] * scale

        for t in range(1, N):
            # transition prior
            prior_prob = torch.mm(
                alphas[t - 1].unsqueeze(0), self.A
            ).squeeze()
            # forward belief propagation
            alphas[t] = prior_prob * observation_probs[t]

            scale[t] = 1.0 / alphas[t].sum()
            alphas[t] = alphas[t] * scale

        return alphas, scale

    def _backward_alg(self, observation_probs, scale):
        N = observation_probs.size(0)

        betas = torch.zeros((N, self.n_states)).to(self.device)

        # initialize with state ending priors
        betas[N - 1] = torch.ones(
            [self.n_states], dtype=torch.float64
        ) * scale[N - 1]

        for t in range(N - 2, -1, -1):
            for i in range(self.n_states):
                betas[t, i] = (
                    betas[t + 1] * observation_probs[t + 1]
                ).dot(self.A[i, :])
            betas[t] = betas[t] * scale[t]

        return betas

    def forward_backward_alg(self, observation_probs):
        alphas, scale = self._forward_alg(observation_probs)
        betas = self._backward_alg(observation_probs, scale)

        return alphas, betas

    def calculate_gammas(self, alphas, betas):
        joint_prob = alphas * betas
        gammas = joint_prob / joint_prob.sum(1).unsqueeze(1)

        return gammas

    def calculate_ksi(self, betas, gammas, obs_seq):
        N = gammas.size(0)
        ksi = torch.zeros([N - 1, self.n_states, self.n_states])

        for t in range(N - 1):
            for i in range(self.n_states):
                next_obs = self.emission_model.prob(obs_seq[t + 1])
                ksi[t, i, :] = gammas[t, i] * self.A[i, :] * next_obs * betas[t + 1]
                ksi[t, i, :] = ksi[t, i, :] / betas[t, i]

        return ksi

    def estimate_priors(self, obs_gammas):
        sum_priors = torch.zeros([self.nstates], dtype=torch.float64).to(self.device)

        for gammas in obs_gammas:
            sum_priors += gammas[0]

        return sum_priors / len(obs_gammas)

    def estimate_transition_matrix(self, obs_gammas, obs_ksi):
        sum_ksi = torch.zeros(
            [self.n_states, self.n_states], dtype=torch.float64
        ).to(self.device)

        sum_gammas = torch.zeros(
            [self.n_states], dtype=torch.float64
        ).to(self.device)

        for gammas, ksi in zip(obs_gammas, obs_ksi):
            sum_ksi += torch.sum(ksi, dim=0)
            sum_gammas += torch.sum(gammas[:-1], dim=0)

        return sum_ksi / sum_gammas

    def e_step(self, observation_sequence):
        observation_probs = self.emission_model.probs(observation_sequence)

        alphas, betas = self.forward_backward_alg(observation_probs)

        return alphas, betas

    def m_step(self, observation_sequence, alphas, betas):
        """
        https://imaging.mrc-cbu.cam.ac.uk/methods/BayesianStuff?action=AttachFile&do=get&target=bilmes-em-algorithm.pdf
        """
        gammas = self.calculate_gammas(alphas, betas)
        ksi = self.calculate_ksi(betas, gammas, observation_sequence)

        return gammas, ksi

    def has_converged(self, prev_parameter, new_parameter):
        delta = torch.max(torch.abs(new_parameter - prev_parameter)).item() < self.eps

        return delta < self.eps

    def step(self, observation_sequences):
        obs_gammas, obs_ksi = [], []

        for observation_sequence in observation_sequences:
            alphas, betas = self.e_step(observation_sequence)
            gammas, ksi = self.m_step(observation_sequence, alphas, betas)
            obs_gammas.append(gammas)
            obs_ksi.append(ksi)

        new_A = self.estimate_transition_matrix(obs_gammas, obs_ksi)
        new_pi0 = self.estimate_priors(obs_gammas)

        emissions_converged = self.emission_model.step(
            observation_sequences, obs_gammas
        )
        A_converged = self.has_converged(self.A, new_A)
        pi0_converged = self.has_converged(self.pi0, new_pi0)
        converged = emissions_converged and A_converged and pi0_converged

        self.update_(transition_matrix=new_A, priors=new_pi0)

        return converged

    def update_(self, transition_matrix=None, priors=None):
        if transition_matrix is not None:
            self.A.data = transition_matrix

        if priors is not None:
            self.pi0.data = priors

    def baum_welch(self, observation_sequence):
        for step in range(self.n_iter + 1):
            has_converged = self.step(observation_sequence)

            if has_converged:
                print("Converged at iteration {}".format(step))

                break
        self.converged_ = True

    def fit(self, observation_sequences):
        self.baum_welch(observation_sequences)

        return self

    def forward(self, x):

        return x
