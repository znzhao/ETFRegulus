"""The policy: a shared per-asset encoder with availability masking.

Specified in reference/rl-training.md section 2. The observation is
`[ global | per-asset (K x F) | portfolio | params ]`, and the whole point of the
architecture is that the per-asset block is processed by *one* set of weights applied K
times, so what the network learns about one sector ETF transfers to the others.

Two deliberate departures from the sketch in that document, both explained where they
happen:

* **The actor head is per-asset too, not a flat `Linear(trunk -> K+1)`.** A flat head has a
  separate weight vector per asset, which throws away the equivariance the encoder just
  bought and makes "adding a ticker later does not require relearning from scratch" false.
  Here `logit_i = head([e_i, context])` with `head` shared across assets, so the model is
  permutation-equivariant end to end and the per-asset parameter count is independent of K.
* **Availability masking pushes the action mean to a large negative rather than `-inf`.**
  The action space is bounded `[-1, 1]` (see `src/env/etf_env.py`), so `-inf` is not
  available and would produce NaN in the Gaussian log-prob besides. A mean of -5 sits five
  standard deviations below the top of the range, and after the environment's `LOGIT_SCALE`
  it is worth `e^-50` of relative weight -- indistinguishable from zero, and differentiable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from src.env.observation import ObservationSpec

#: Action-mean floor for an asset that does not exist yet. See the module docstring.
MASK_LOGIT = -5.0


def _mlp(sizes: list[int], activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes, sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers)


class PortfolioExtractor(BaseFeaturesExtractor):
    """Split the observation, encode assets with shared weights, build a context vector.

    Emits a flat tensor of `[per-asset embeddings (K*E) | context (C)]` because SB3's
    contract requires a flat feature vector; `SharedAssetActorCritic` reshapes it back.
    Keeping the per-asset embeddings *un-pooled* in the output is what lets the actor head
    stay per-asset -- a pooled-only feature vector would force a flat head.
    """

    def __init__(self, observation_space: spaces.Box, obs_spec: ObservationSpec,
                 asset_hidden: int = 128, embed_dim: int = 64, context_dim: int = 128):
        self.obs_spec = obs_spec
        K, F = obs_spec.n_assets, obs_spec.n_per_asset
        self.K, self.F = K, F
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        super().__init__(observation_space, features_dim=K * embed_dim + context_dim)

        self.asset_encoder = _mlp([F, asset_hidden, embed_dim])

        # Context: what the whole portfolio and the world look like. Mean AND max pooling
        # -- mean carries "what does the universe look like on average", max carries "is
        # there anything extreme out there", and a single pool loses one of them.
        n_global = len(obs_spec.global_block)
        n_portfolio = obs_spec.portfolio_slice.stop - obs_spec.portfolio_slice.start
        n_params = obs_spec.param_slice.stop - obs_spec.param_slice.start
        self.context_encoder = _mlp(
            [2 * embed_dim + n_global + n_portfolio + n_params, context_dim, context_dim])

        self._slices = (obs_spec.macro_slice, obs_spec.per_asset_slice,
                        obs_spec.portfolio_slice, obs_spec.param_slice)
        # Where the availability bit sits inside one asset's block.
        self._avail_col = obs_spec.per_asset_fields.index("pf_available")

    def split(self, obs: torch.Tensor):
        g, pa, pf, pr = self._slices
        assets = obs[:, pa.start:pa.stop].view(-1, self.K, self.F)
        return obs[:, g.start:g.stop], assets, obs[:, pf.start:pf.stop], obs[:, pr.start:pr.stop]

    def availability(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, K) float mask, straight out of the observation.

        This works only because observations are NOT re-normalized by `VecNormalize`
        (see `src/agents/normalization.py`): the bit is still exactly 0 or 1 here.
        """
        _, assets, _, _ = self.split(obs)
        return assets[:, :, self._avail_col]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        glob, assets, portfolio, params = self.split(obs)
        # One encoder, applied K times. Folding K into the batch is the cheap way to do it.
        embeddings = self.asset_encoder(assets.reshape(-1, self.F))
        embeddings = embeddings.view(-1, self.K, self.embed_dim)

        # Unavailable assets must not pollute the pooled context. Mean over available
        # only; max with the missing ones pushed to -inf-ish.
        mask = assets[:, :, self._avail_col].unsqueeze(-1)
        n_avail = mask.sum(dim=1).clamp(min=1.0)
        pooled_mean = (embeddings * mask).sum(dim=1) / n_avail
        pooled_max = (embeddings.masked_fill(mask == 0, -1e4)).max(dim=1).values
        pooled_max = torch.nan_to_num(pooled_max, neginf=0.0)

        context = self.context_encoder(
            torch.cat([pooled_mean, pooled_max, glob, portfolio, params], dim=1))
        return torch.cat([embeddings.reshape(-1, self.K * self.embed_dim), context], dim=1)


class _AssetHeadExtractor(nn.Module):
    """Stands in for SB3's `MlpExtractor`.

    `forward_actor` returns the K+1 action means directly, so the policy's `action_net`
    becomes an identity and the per-asset head *is* the final layer. `forward_critic`
    returns an ordinary trunk latent, because a value function is a single scalar about the
    whole portfolio and has nothing to gain from equivariance.
    """

    def __init__(self, K: int, embed_dim: int, context_dim: int,
                 head_hidden: int = 128, trunk: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.K, self.embed_dim, self.context_dim = K, embed_dim, context_dim
        self.latent_dim_pi = K + 1
        self.latent_dim_vf = trunk[-1]

        # Shared across assets: [asset embedding, context] -> one logit.
        self.asset_head = nn.Sequential(
            nn.Linear(embed_dim + context_dim, head_hidden), nn.Tanh(),
            nn.Linear(head_hidden, 1))
        # CASH has no per-asset block of its own, so it gets its own small head off the
        # context. It is the outside option, not one of the K.
        self.cash_head = nn.Sequential(
            nn.Linear(context_dim, head_hidden), nn.Tanh(), nn.Linear(head_hidden, 1))
        self.value_trunk = _mlp([K * embed_dim + context_dim, *trunk])

    def _split_features(self, features: torch.Tensor):
        n = self.K * self.embed_dim
        return features[:, :n].view(-1, self.K, self.embed_dim), features[:, n:]

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        embeddings, context = self._split_features(features)
        ctx = context.unsqueeze(1).expand(-1, self.K, -1)
        asset_logits = self.asset_head(
            torch.cat([embeddings, ctx], dim=2).reshape(-1, self.embed_dim + self.context_dim)
        ).view(-1, self.K)
        return torch.cat([self.cash_head(context), asset_logits], dim=1)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_trunk(features)

    def forward(self, features: torch.Tensor):
        return self.forward_actor(features), self.forward_critic(features)


class SharedAssetActorCritic(ActorCriticPolicy):
    """PPO policy with the shared per-asset head and availability masking."""

    def __init__(self, observation_space, action_space, lr_schedule: Callable,
                 *, obs_spec: ObservationSpec, asset_hidden: int = 128,
                 embed_dim: int = 64, context_dim: int = 128, head_hidden: int = 128,
                 trunk: tuple[int, ...] = (256, 256), mask_unavailable: bool = True,
                 **kwargs):
        self.obs_spec = obs_spec
        self._arch = dict(asset_hidden=asset_hidden, embed_dim=embed_dim,
                          context_dim=context_dim, head_hidden=head_hidden,
                          trunk=tuple(trunk))
        self.mask_unavailable = bool(mask_unavailable)
        kwargs.pop("net_arch", None)
        kwargs["features_extractor_class"] = PortfolioExtractor
        kwargs["features_extractor_kwargs"] = dict(
            obs_spec=obs_spec, asset_hidden=asset_hidden, embed_dim=embed_dim,
            context_dim=context_dim)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _AssetHeadExtractor(
            self.obs_spec.n_assets, self._arch["embed_dim"], self._arch["context_dim"],
            head_hidden=self._arch["head_hidden"], trunk=self._arch["trunk"])

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # The per-asset head already produced the action means. A Linear on top would
        # reintroduce exactly the per-asset weights the head exists to avoid.
        self.action_net = nn.Identity()

    # ------------------------------------------------------------------ masking

    def _mask(self, obs: torch.Tensor, mean_actions: torch.Tensor) -> torch.Tensor:
        if not self.mask_unavailable:
            return mean_actions
        avail = self.features_extractor.availability(obs)          # (B, K)
        # Column 0 is CASH, which is always available -- it is synthetic.
        full = torch.cat([torch.ones_like(avail[:, :1]), avail], dim=1)
        return torch.where(full > 0.5, mean_actions,
                           torch.full_like(mean_actions, MASK_LOGIT))

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        mean_actions = self._mask(obs, self.action_net(latent_pi))
        distribution = self._distribution_from(mean_actions)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, values, distribution.log_prob(actions)

    def _distribution_from(self, mean_actions: torch.Tensor):
        return self.action_dist.proba_distribution(mean_actions, self.log_std)

    def get_distribution(self, obs: torch.Tensor):
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._distribution_from(self._mask(obs, self.action_net(latent_pi)))

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._distribution_from(self._mask(obs, self.action_net(latent_pi)))
        return (self.value_net(latent_vf), distribution.log_prob(actions),
                distribution.entropy())

    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)


def policy_kwargs_from(resolved: dict, obs_spec: ObservationSpec) -> dict:
    """Read the architecture out of config, so an ablation is a config change."""
    net = (resolved.get("policy", {}) or {})
    return dict(
        obs_spec=obs_spec,
        asset_hidden=int(net.get("asset_hidden", 128)),
        embed_dim=int(net.get("embed_dim", 64)),
        context_dim=int(net.get("context_dim", 128)),
        head_hidden=int(net.get("head_hidden", 128)),
        trunk=tuple(net.get("trunk", (256, 256))),
        mask_unavailable=bool(net.get("mask_unavailable", True)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
