import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from util import input_to_tensor


class HMM(nn.Module):
    """Some Information about HMM"""

    def __init__(
        self,
        emission_model,
        n_states=None,
        A=None,
        pi0=None,
        n_iter=100,
        dtype=torch.float,
        device="cpu",
    ):
        super(HMM, self).__init__()
        self.n_iter = n_iter
        self.eps = 1e-6
        self.device = device
        self.dtype = dtype

        if n_states is None and A is None:
            raise ValueError(
                "You must either pass the number of states or the Transition Matrix"
            )

        if n_states is None:
            n_states = A.shape[0]

        self.n_states = n_states

        self.A = nn.Parameter(
            torch.Tensor(n_states, n_states).type(dtype), requires_grad=False
        )
        self.pi0 = nn.Parameter(torch.Tensor(n_states).type(dtype), requires_grad=False)
        self.emission_model = emission_model
        self.converged_ = False

        self.reset_parameters(A=A, pi0=pi0)

        self.to(self.device)

    def reset_parameters(self, A=None, pi0=None):
        self.A.normal_()
        self.pi0.fill_(1.0 / self.n_states)

        if A is not None:
            A = input_to_tensor(A, dtype=self.dtype)
            self.update_(transition_matrix=A)

        if pi0 is not None:
            pi0 = input_to_tensor(pi0, dtype=self.dtype)
            self.update_(priors=pi0)

    def _forward_alg(self, observation_probs):
        N = observation_probs.size(0)

        alphas = torch.zeros((N, self.n_states), dtype=self.dtype).to(self.device)
        alphas[0] = self.pi0 * observation_probs[0]

        scale = torch.zeros([N], dtype=self.dtype).to(self.device)

        scale[0] = 1.0 / alphas[0].sum()
        alphas[0] = alphas[0] * scale[0]

        for t in range(1, N):
            # transition prior
            prior_prob = torch.mm(alphas[t - 1].unsqueeze(0), self.A).squeeze()
            # forward belief propagation
            alphas[t] = prior_prob * observation_probs[t]

            scale[t] = 1.0 / alphas[t].sum()
            alphas[t] = alphas[t] * scale[t]

        return alphas, scale

    def _backward_alg(self, observation_probs, scale):
        N = observation_probs.size(0)

        betas = torch.zeros((N, self.n_states), dtype=self.dtype).to(self.device)

        # initialize with state ending priors
        betas[N - 1] = (
            torch.ones([self.n_states], dtype=self.dtype).to(self.device) * scale[N - 1]
        )

        for t in range(N - 2, -1, -1):
            for i in range(self.n_states):
                betas[t, i] = (betas[t + 1] * observation_probs[t + 1]).dot(
                    self.A[i, :]
                )
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

    def calculate_ksi(self, alphas, betas, obs_seq):
        N = alphas.size(0)
        ksi = torch.zeros((N - 1, self.n_states, self.n_states), dtype=self.dtype).to(
            self.device
        )

        for t in range(N - 1):
            next_prob = self.emission_model.prob(obs_seq[t + 1].unsqueeze(0))
            tmp = torch.matmul(alphas[t].unsqueeze(0), self.A) * next_prob.unsqueeze(0)
            denom = torch.matmul(tmp, betas[t + 1].unsqueeze(1)).squeeze()

            for i in range(self.n_states):
                num = alphas[t, i] * self.A[i, :] * next_prob * betas[t + 1]
                ksi[t, i, :] = num / denom

        return ksi

    def estimate_priors(self, obs_gammas):
        sum_priors = torch.zeros([self.n_states], dtype=self.dtype).to(self.device)

        for gammas in obs_gammas:
            sum_priors += gammas[0]

        return sum_priors / len(obs_gammas)

    def estimate_transition_matrix(self, obs_gammas, obs_ksi):
        sum_ksi = torch.zeros([self.n_states, self.n_states], dtype=self.dtype).to(
            self.device
        )

        sum_gammas = torch.zeros([self.n_states], dtype=self.dtype).to(self.device)

        for gammas, ksi in zip(obs_gammas, obs_ksi):
            sum_ksi += torch.sum(ksi, dim=0)
            sum_gammas += torch.sum(gammas[:-1], dim=0)

        return sum_ksi / sum_gammas.unsqueeze(1)

    def e_step(self, observation_sequence):
        observation_probs = self.emission_model.prob(observation_sequence)

        alphas, betas = self.forward_backward_alg(observation_probs)

        return alphas, betas

    def m_step(self, observation_sequence, alphas, betas):
        """
        https://imaging.mrc-cbu.cam.ac.uk/methods/BayesianStuff?action=AttachFile&do=get&target=bilmes-em-algorithm.pdf
        """
        gammas = self.calculate_gammas(alphas, betas)
        ksi = self.calculate_ksi(alphas, betas, observation_sequence)

        return gammas, ksi

    def has_converged(self, prev_parameter, new_parameter):
        delta = torch.max(torch.abs(new_parameter - prev_parameter)).item()
        print(delta)
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
        current_iteration = 0
        has_converged = False

        for _ in tqdm(range(self.n_iter), desc="Training HMM...", total=self.n_iter):
            has_converged = self.step(observation_sequence)
            current_iteration += 1

            if has_converged:
                break

        if has_converged:
            print("Converged at iteration {}".format(current_iteration))
            self.converged_ = True
        else:
            print("Could not converge after {} iterations".format(current_iteration))

    def fit(self, observation_sequences):
        self.baum_welch(observation_sequences)

        return self

    def viterbi(self, observation_sequence):
        N = len(observation_sequence)
        v_prob = torch.zeros((N, self.n_states), dtype=self.dtype).to(self.device)

        backpointer = torch.zeros((N, self.n_states), dtype=torch.long).to(self.device)

        obs_prob = self.emission_model.log_prob(observation_sequence)
        v_prob[0, :] = torch.log(self.pi0) + obs_prob[0, :]

        logA = torch.log(self.A)

        for t in range(1, N):
            # belief probagation
            belief = v_prob[t - 1, :].view(-1, 1) + logA
            best_belief, best_pointer = torch.max(belief, 0)
            v_prob[t, :] = best_belief + obs_prob[t]
            backpointer[t, :] = best_pointer

        best_path = torch.zeros([N], dtype=torch.long).to(self.device)
        best_path[N - 1] = torch.argmax(v_prob[N - 1, :], dim=0)

        for t in range(N - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        v_prob = torch.exp(v_prob)

        return best_path, v_prob

    def forward(self, x):

        return x
