"""The policy architecture: weight sharing, availability masking, and SB3 compatibility.

The two properties worth testing are the ones the architecture exists for. Weight sharing
is only real if the per-asset parameter count does not grow with the universe, and
permutation equivariance is only real if swapping two assets' inputs swaps their logits --
both are easy to *claim* from a diagram and easy to lose in the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.agents.policy import (
    MASK_LOGIT,
    PortfolioExtractor,
    SharedAssetActorCritic,
    count_parameters,
    policy_kwargs_from,
)
from src.env.observation import ObservationSpec

pytest.importorskip("stable_baselines3")


def _spec(n_assets: int = 6, n_etf: int = 3, n_macro: int = 4) -> ObservationSpec:
    """A synthetic layout, so these tests do not need the data pipeline."""
    spec = ObservationSpec(
        tickers=tuple(f"T{i}" for i in range(n_assets)),
        macro=tuple(f"m{i}" for i in range(n_macro)),
        global_cross_sectional=(),
        per_asset_etf=tuple(f"f{i}" for i in range(n_etf)),
        per_asset_cross_sectional=(),
    )
    return ObservationSpec(**{**spec.__dict__, "names": spec.build_names()})


def _policy(spec: ObservationSpec) -> SharedAssetActorCritic:
    obs_space = spaces.Box(-10.0, 10.0, (spec.size,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, (spec.n_assets + 1,), dtype=np.float32)
    return SharedAssetActorCritic(obs_space, act_space, lambda _: 3e-4, obs_spec=spec,
                                  asset_hidden=16, embed_dim=8, context_dim=16,
                                  head_hidden=16, trunk=(32, 32))


def _obs(spec: ObservationSpec, batch: int = 4, *, available=None) -> torch.Tensor:
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(batch, spec.size)).astype(np.float32)
    col = spec.per_asset_fields.index("pf_available")
    start, F = spec.per_asset_slice.start, spec.n_per_asset
    for j in range(spec.n_assets):
        bit = 1.0 if available is None else float(available[j])
        obs[:, start + j * F + col] = bit
    return torch.as_tensor(obs)


# --------------------------------------------------------------- weight sharing


def test_the_per_asset_parameter_count_does_not_grow_with_the_universe():
    """The whole claim of the shared encoder. A flat head would fail this outright."""
    small = count_parameters(_policy(_spec(n_assets=6)))
    large = count_parameters(_policy(_spec(n_assets=24)))
    # The critic trunk reads the concatenated embeddings, so it does grow. The actor path
    # must not, so the total growth is far below the 4x the asset count grew by.
    assert large < small * 2.0, (
        f"{small} -> {large} parameters for 6 -> 24 assets; the per-asset weights are "
        "not being shared")


def test_the_head_is_permutation_equivariant():
    """Swap two assets' input blocks and their logits must swap, with the rest unmoved.

    This is what makes "what it learns about one sector ETF transfers to the others"
    a property of the network rather than a hope.
    """
    spec = _spec(n_assets=5)
    policy = _policy(spec).eval()
    obs = _obs(spec, batch=1)

    F, start = spec.n_per_asset, spec.per_asset_slice.start
    swapped = obs.clone()
    a, b = 1, 3
    block_a = obs[:, start + a * F:start + (a + 1) * F].clone()
    block_b = obs[:, start + b * F:start + (b + 1) * F].clone()
    swapped[:, start + a * F:start + (a + 1) * F] = block_b
    swapped[:, start + b * F:start + (b + 1) * F] = block_a

    with torch.no_grad():
        base = policy.get_distribution(obs).distribution.mean[0]
        perm = policy.get_distribution(swapped).distribution.mean[0]

    # Column 0 is CASH and is not part of the permutation.
    assert base[0].item() == pytest.approx(perm[0].item(), abs=1e-5)
    assert base[1 + a].item() == pytest.approx(perm[1 + b].item(), abs=1e-5)
    assert base[1 + b].item() == pytest.approx(perm[1 + a].item(), abs=1e-5)
    untouched = [i for i in range(spec.n_assets) if i not in (a, b)]
    for j in untouched:
        assert base[1 + j].item() == pytest.approx(perm[1 + j].item(), abs=1e-5)


# ------------------------------------------------------------------- masking


def test_unavailable_assets_are_pushed_to_the_action_floor():
    spec = _spec(n_assets=5)
    policy = _policy(spec).eval()
    available = [1, 0, 1, 0, 1]
    obs = _obs(spec, batch=3, available=available)
    with torch.no_grad():
        mean = policy.get_distribution(obs).distribution.mean
    for j, bit in enumerate(available):
        column = mean[:, 1 + j]
        if bit:
            assert not torch.allclose(column, torch.full_like(column, MASK_LOGIT))
        else:
            assert torch.allclose(column, torch.full_like(column, MASK_LOGIT)), (
                f"asset {j} is unavailable but was not masked")


def test_cash_is_never_masked():
    """CASH is synthetic and always available -- it is the agent's outside option."""
    spec = _spec(n_assets=4)
    policy = _policy(spec).eval()
    obs = _obs(spec, batch=2, available=[0, 0, 0, 0])
    with torch.no_grad():
        mean = policy.get_distribution(obs).distribution.mean
    assert not torch.allclose(mean[:, 0], torch.full_like(mean[:, 0], MASK_LOGIT))


def test_masking_can_be_turned_off_for_the_ablation():
    spec = _spec(n_assets=4)
    obs_space = spaces.Box(-10.0, 10.0, (spec.size,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, (spec.n_assets + 1,), dtype=np.float32)
    policy = SharedAssetActorCritic(obs_space, act_space, lambda _: 3e-4, obs_spec=spec,
                                    asset_hidden=8, embed_dim=8, context_dim=8,
                                    head_hidden=8, trunk=(16,),
                                    mask_unavailable=False).eval()
    obs = _obs(spec, batch=2, available=[0, 0, 0, 0])
    with torch.no_grad():
        mean = policy.get_distribution(obs).distribution.mean
    assert not torch.allclose(mean, torch.full_like(mean, MASK_LOGIT))


def test_the_pooled_context_ignores_unavailable_assets():
    """An ETF that does not exist yet must not shift the universe summary."""
    spec = _spec(n_assets=5)
    extractor = PortfolioExtractor(
        spaces.Box(-10.0, 10.0, (spec.size,), dtype=np.float32), spec,
        asset_hidden=8, embed_dim=8, context_dim=8).eval()
    obs = _obs(spec, batch=1, available=[1, 1, 1, 0, 0])
    start, F = spec.per_asset_slice.start, spec.n_per_asset
    changed = obs.clone()
    for j in (3, 4):                       # scribble over the unavailable blocks
        col = spec.per_asset_fields.index("pf_available")
        block = torch.randn(F)
        block[col] = 0.0
        changed[0, start + j * F:start + (j + 1) * F] = block
    with torch.no_grad():
        a, b = extractor(obs), extractor(changed)
    context_start = spec.n_assets * 8
    assert torch.allclose(a[:, context_start:], b[:, context_start:], atol=1e-6), (
        "changing an unavailable asset's features moved the pooled context")


# ---------------------------------------------------------------- SB3 contract


def test_the_policy_satisfies_sb3s_contract():
    """`evaluate_actions` must agree with `forward` on the log-prob of the action it
    sampled -- PPO's importance ratio is meaningless otherwise (D9)."""
    spec = _spec(n_assets=5)
    policy = _policy(spec).eval()
    obs = _obs(spec, batch=6)
    with torch.no_grad():
        actions, values, log_prob = policy(obs)
        values2, log_prob2, entropy = policy.evaluate_actions(obs, actions)
    assert actions.shape == (6, spec.n_assets + 1)
    assert torch.allclose(values, values2, atol=1e-6)
    assert torch.allclose(log_prob, log_prob2, atol=1e-5)
    assert entropy.shape == (6,)


def test_deterministic_prediction_is_deterministic():
    spec = _spec(n_assets=5)
    policy = _policy(spec).eval()
    obs = _obs(spec, batch=3)
    with torch.no_grad():
        a = policy._predict(obs, deterministic=True)
        b = policy._predict(obs, deterministic=True)
    assert torch.equal(a, b)


def test_policy_kwargs_come_from_config():
    spec = _spec()
    kwargs = policy_kwargs_from({"policy": {"embed_dim": 11, "mask_unavailable": False}},
                                spec)
    assert kwargs["embed_dim"] == 11
    assert kwargs["mask_unavailable"] is False
    assert kwargs["obs_spec"] is spec
