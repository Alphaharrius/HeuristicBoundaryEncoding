import math

import torch

from hbe.mcmc import RandomWalkSampler


def test_random_walk_shapes_and_logprob():
    sampler = RandomWalkSampler()
    current = torch.zeros((4, 5, 3), dtype=torch.long)

    proposal = sampler.sample(current)

    assert proposal.sample.shape == current.shape
    assert proposal.log_prob.shape == (current.shape[0],)
    assert proposal.log_prob_rev.shape == (current.shape[0],)
    expected = -math.log(current.shape[1])
    assert torch.allclose(
        proposal.log_prob, torch.full_like(proposal.log_prob, expected)
    )
    assert torch.allclose(proposal.log_prob_rev, proposal.log_prob)


def test_random_walk_same_orbital_swap_only():
    sampler = RandomWalkSampler()
    bsz, n_sites, n_orbitals = 3, 4, 2
    base = torch.arange(bsz * n_sites * n_orbitals).reshape(bsz, n_sites, n_orbitals)

    proposal = sampler.sample(base)
    diff = proposal.sample != base

    for b in range(bsz):
        idx = diff[b].nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        assert idx.shape[0] == 2
        assert idx[0, 1].item() == idx[1, 1].item()
        original_vals = base[b][idx[:, 0], idx[:, 1]]
        proposal_vals = proposal.sample[b][idx[:, 0], idx[:, 1]]
        assert torch.equal(original_vals.sort().values, proposal_vals.sort().values)


def test_random_walk_orbital_sweep_order():
    sampler = RandomWalkSampler()
    n_sites, n_orbitals = 5, 3
    current = torch.zeros((1, n_sites, n_orbitals), dtype=torch.long)

    for _ in range(n_sites):
        sampler.sample(current)

    assert sampler._r0 == 0
    assert sampler._o == 1
