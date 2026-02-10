from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import NamedTuple

import torch
import torch.nn as nn


class Proposal(NamedTuple):
    sample: torch.Tensor  # (B: batch, N: site indices, o: orbitals)
    log_prob: torch.Tensor  # (B,)
    log_prob_rev: torch.Tensor  # (B,)


class ProposalSampler(ABC):
    @abstractmethod
    def sample(self, current: torch.Tensor) -> Proposal:
        """
        Generate a new sample given the current sample in the Markov chain.

        Parameters
        ----------
        `current` : `torch.Tensor`
            Current sample(s) in the Markov chain as an integer `torch.Tensor`.
            Shape `(B, N, o)` where `B` is the batch size, `N` is the site index, and `o` is the orbital
            index (intrinsic degrees of freedom) of each sample.

        Returns
        -------
        `Proposal`
            A `Proposal` named tuple containing:
            - `sample`: New sample(s) generated as a integer `torch.Tensor`, shape `(B, N, o)`.
            - `log_prob`: Log probabilities of the proposal distribution, shape `(B,)`.
            - `log_prob_rev`: Log probabilities of the reverse proposal distribution, shape `(B,)`.
        """
        pass


class Sample(NamedTuple):
    """
    A realized sample in the chain with its target log-probability.
    """

    sample: torch.Tensor  # (B, N, o)
    log_prob: torch.Tensor  # (B,)


class MetropolisHastingsSampler:
    def __init__(self, q: ProposalSampler) -> None:
        """
        Metropolis-Hastings sampler using a given proposal sampler.

        Parameters
        ----------
            q: An instance of `ProposalSampler` to generate proposals.
        """
        self.q = q

    def next(self, g: nn.Module, current: Sample) -> Sample:
        """
        Get the next accepted sample in the Markov chain using the Metropolis-Hastings algorithm.

        Parameters
        ----------
        `g` : `nn.Module`
            A neural network module that computes the coefficients of the samples, the output must
            have shape `(B, N, 2)` where the last dimension contains the real and imaginary parts.
        `current` : `Sample`
            Current sample as the last element of the Markov chain. Contains:
            - `sample`: Integer `torch.Tensor` of shape `(B, N, o)`.
            - `log_prob`: Target log probabilities of the current sample, shape `(B,)`.

        Returns
        -------
        `Sample`
            A `Sample` named tuple containing:
            - `sample`: Next accepted sample(s) in the Markov chain as an integer `torch.Tensor`, shape `(B, N, o)`.
            - `log_prob`: Target log probabilities of the accepted samples, shape `(B,)`.
        """
        proposal = self.q.sample(current.sample)

        current_log_prob = torch.as_tensor(current.log_prob)
        proposal_target_log_prob = self._target_log_prob(g, proposal.sample)

        if current_log_prob.ndim == 0:
            current_log_prob = current_log_prob.unsqueeze(0)
        if proposal_target_log_prob.ndim == 0:
            proposal_target_log_prob = proposal_target_log_prob.unsqueeze(0)

        if current_log_prob.shape != proposal_target_log_prob.shape:
            raise ValueError(
                "log_prob_fn must return shape (B,) matching current sample batch."
            )

        proposal_log_prob = torch.as_tensor(proposal.log_prob)
        proposal_log_prob_rev = torch.as_tensor(proposal.log_prob_rev)

        if proposal_log_prob.ndim == 0:
            proposal_log_prob = proposal_log_prob.unsqueeze(0)
        if proposal_log_prob_rev.ndim == 0:
            proposal_log_prob_rev = proposal_log_prob_rev.unsqueeze(0)

        if proposal_log_prob.shape != proposal_log_prob_rev.shape:
            raise ValueError(
                "Proposal log-probabilities must have matching shape (B,)."
            )

        log_accept_ratio = (
            proposal_target_log_prob
            - current_log_prob
            + proposal_log_prob_rev
            - proposal_log_prob
        )
        rand_u = torch.rand(
            log_accept_ratio.shape,
            device=log_accept_ratio.device,
            dtype=log_accept_ratio.dtype,
        )
        accept = torch.log(rand_u) < log_accept_ratio

        accept_mask = accept.view((-1,) + (1,) * (current.sample.ndim - 1))
        next_sample = torch.where(accept_mask, proposal.sample, current.sample)
        next_log_prob = torch.where(accept, proposal_target_log_prob, current_log_prob)

        return Sample(sample=next_sample, log_prob=next_log_prob)

    @staticmethod
    def _target_log_prob(g: nn.Module, samples: torch.Tensor) -> torch.Tensor:
        """
        Compute log |psi|^2 from a model output of shape (B, 2) or (2,).
        """
        out = g(samples)
        out = torch.as_tensor(out)
        if out.ndim == 1:
            if out.shape[0] != 2:
                raise ValueError(
                    "Model output must have shape (2,) for a single sample."
                )
            out = out.unsqueeze(0)
        if out.ndim != 2 or out.shape[1] != 2:
            raise ValueError("Model output must have shape (B, 2).")

        re = out[:, 0]
        im = out[:, 1]
        return torch.log(re * re + im * im)


class RandomWalkSampler(ProposalSampler):
    def sample(self, current: torch.Tensor) -> Proposal:
        """
        Perform a sweep walk proposal over all sites in the sample from `0 -> N` then repeat. Each sweep will
        select the next site index `r_0`, and select the next orbital index `o`, and choose a random site `r_1`, and
        swap their values at orbital `o` (orbital swaps only occur between sites for the same orbital).

        Parameters
        ----------
        `current` : `torch.Tensor`
            Current sample(s) in the Markov chain as an integer `torch.Tensor`.
            Shape `(B, N, o)` where `B` is the batch size, `N` is the site index, and `o` is the orbital
            index (intrinsic degrees of freedom) of each sample.
        """
        if not hasattr(self, "_r0"):
            self._r0 = 0
            self._o = 0

        bsz, n_sites, n_orbitals = current.shape

        r0 = self._r0
        o = self._o
        if r0 >= n_sites:
            r0 = 0
        if o >= n_orbitals:
            o = 0

        self._r0 = r0 + 1
        self._o = o
        if self._r0 >= n_sites:
            self._r0 = 0
            self._o = (o + 1) % n_orbitals

        device = current.device
        r1 = torch.randint(0, n_sites, (bsz,), device=device)

        proposal = current.clone()
        b_idx = torch.arange(bsz, device=device)
        v0 = proposal[b_idx, r0, o].clone()
        v1 = proposal[b_idx, r1, o].clone()
        proposal[b_idx, r0, o] = v1
        proposal[b_idx, r1, o] = v0

        log_q = -math.log(n_sites)
        log_prob = torch.full(
            (bsz,), log_q, device=device, dtype=torch.get_default_dtype()
        )
        return Proposal(sample=proposal, log_prob=log_prob, log_prob_rev=log_prob)
