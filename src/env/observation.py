"""The observation layout: what the policy sees, in what order, and why it is legal.

Two jobs, both of which exist to catch a silent index shift:

1. **Declare the layout.** A single flat `float32` vector in the fixed order given by
   reference/env-mdp.md section 1:

       [ macro | cross_sectional | per_asset (K x F) | portfolio | params ]

   Per-asset blocks follow the canonical ticker order in `config/universe.yaml`, which
   never changes -- appending a ticker is a policy-invalidating change, and the manifest
   hash is what catches it.

2. **Enforce the Markov floor.** The market half of the observation is configurable
   (`config/observation.yaml`, open question Q6). The portfolio and parameter halves are
   NOT: they are built from `MANDATORY_FIELDS` below, and `validate()` fails if any is
   missing. A missing feasibility field does not make the problem harder, it makes it
   non-Markov, and PPO's assumptions break silently rather than loudly.

Every field is addressable by name through `index_of`, so a test can assert what is at
each position instead of trusting a comment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

MANIFEST_PATH = Path("data/features/feature_manifest.json")

#: The Markov sufficiency floor from reference/env-mdp.md section 1. The observation must
#: be sufficient to reconstruct which actions the projection will alter; each of these
#: carries part of that. Not configurable, by design.
MANDATORY_PER_ASSET: tuple[str, ...] = (
    "pf_weight",            # where the portfolio is now
    "pf_position",          # shares > 0
    "pf_locked",            # the lock floor is active for this asset
    "pf_lock_remaining",    # remaining calendar days / N
    "pf_unlock_proximity",  # 1 - remaining/N, 0 when unlocked
    "pf_available",         # the availability mask -- what carries pre-inception info
)

MANDATORY_GLOBAL: tuple[str, ...] = (
    "pf_cash_weight",
    "pf_nav_norm",          # NAV relative to episode start
    "pf_peak_norm",         # the running peak, which the episode INHERITS
    "pf_drawdown",          # D_t
    "pf_drawdown_budget",   # B_t = D_max - D_t. Mandatory, not optional.
    "pf_locked_count",      # how many positions are locked / K
    "pf_locked_nav_frac",   # fraction of NAV that cannot be sold
    "pf_wavg_lock_days",    # value-weighted remaining lock, in units of N
)

MANDATORY_PARAMS: tuple[str, ...] = ("param_hold_days", "param_max_drawdown")

MANDATORY_FIELDS = MANDATORY_PER_ASSET + MANDATORY_GLOBAL + MANDATORY_PARAMS


class ObservationError(ValueError):
    """The declared observation does not agree with the feature manifest."""


@dataclass(frozen=True)
class ObservationSpec:
    """The resolved layout. Immutable: it is part of a trained policy's identity."""

    tickers: tuple[str, ...]
    macro: tuple[str, ...]
    global_cross_sectional: tuple[str, ...]
    per_asset_etf: tuple[str, ...]
    per_asset_cross_sectional: tuple[str, ...]
    hold_days_divisor: float = 365.0
    clip: float = 10.0
    names: tuple[str, ...] = field(default=(), repr=False)

    # ------------------------------------------------------------------ geometry

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def per_asset_market(self) -> tuple[str, ...]:
        return self.per_asset_etf + self.per_asset_cross_sectional

    @property
    def per_asset_fields(self) -> tuple[str, ...]:
        """Market features first, then the mandatory portfolio fields, per asset."""
        return self.per_asset_market + MANDATORY_PER_ASSET

    @property
    def n_per_asset(self) -> int:
        return len(self.per_asset_fields)

    @property
    def global_block(self) -> tuple[str, ...]:
        return self.macro + self.global_cross_sectional

    @property
    def macro_slice(self) -> slice:
        return slice(0, len(self.global_block))

    @property
    def per_asset_slice(self) -> slice:
        start = len(self.global_block)
        return slice(start, start + self.n_assets * self.n_per_asset)

    @property
    def portfolio_slice(self) -> slice:
        start = self.per_asset_slice.stop
        return slice(start, start + len(MANDATORY_GLOBAL))

    @property
    def param_slice(self) -> slice:
        start = self.portfolio_slice.stop
        return slice(start, start + len(MANDATORY_PARAMS))

    @property
    def size(self) -> int:
        return self.param_slice.stop

    # -------------------------------------------------------------------- naming

    def build_names(self) -> tuple[str, ...]:
        out: list[str] = [f"g:{c}" for c in self.global_block]
        for ticker in self.tickers:
            out.extend(f"a:{ticker}:{c}" for c in self.per_asset_fields)
        out.extend(f"p:{c}" for c in MANDATORY_GLOBAL)
        out.extend(f"p:{c}" for c in MANDATORY_PARAMS)
        return tuple(out)

    def index_of(self, name: str) -> int:
        """Position of a named field. The addressable form of the layout comment."""
        names = self.names or self.build_names()
        try:
            return names.index(name)
        except ValueError as exc:
            raise ObservationError(f"no observation field named {name!r}") from exc

    def asset_slice(self, ticker: str) -> slice:
        j = self.tickers.index(ticker)
        start = self.per_asset_slice.start + j * self.n_per_asset
        return slice(start, start + self.n_per_asset)

    def describe(self) -> dict:
        return {
            "size": self.size,
            "n_assets": self.n_assets,
            "per_asset_fields": list(self.per_asset_fields),
            "per_asset_market": len(self.per_asset_market),
            "per_asset_portfolio": len(MANDATORY_PER_ASSET),
            "global_market": len(self.global_block),
            "portfolio_global": len(MANDATORY_GLOBAL),
            "params": len(MANDATORY_PARAMS),
            "blocks": {
                "macro": [self.macro_slice.start, self.macro_slice.stop],
                "per_asset": [self.per_asset_slice.start, self.per_asset_slice.stop],
                "portfolio": [self.portfolio_slice.start, self.portfolio_slice.stop],
                "params": [self.param_slice.start, self.param_slice.stop],
            },
        }

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_config(cls, resolved: dict, tickers: Sequence[str],
                    *, manifest_path: Path = MANIFEST_PATH,
                    validate: bool = True) -> "ObservationSpec":
        obs = resolved.get("observation")
        if not obs:
            raise ObservationError(
                "no `observation:` block in the config; the observation selection is "
                "declared, never inferred -- see config/observation.yaml (Q6)"
            )
        per_asset = obs.get("per_asset", {})
        glob = obs.get("global", {})
        scaling = resolved.get("scaling", {}) or {}

        spec = cls(
            tickers=tuple(tickers),
            macro=tuple(glob.get("macro", ())),
            global_cross_sectional=tuple(glob.get("cross_sectional", ())),
            per_asset_etf=tuple(per_asset.get("etf", ())),
            per_asset_cross_sectional=tuple(per_asset.get("cross_sectional", ())),
            hold_days_divisor=float(scaling.get("hold_days_divisor", 365.0)),
            clip=float(scaling.get("clip", 10.0)),
        )
        spec = cls(**{**spec.__dict__, "names": spec.build_names()})
        if validate:
            spec.validate(manifest_path)
        return spec

    # ------------------------------------------------------------------ validation

    def validate(self, manifest_path: Path = MANIFEST_PATH) -> dict:
        """Check every declared name against the manifest, and the Markov floor.

        Raises rather than warns. A feature the environment thinks it is reading but the
        pipeline never produced is a column of zeros the policy would learn around.
        """
        if not manifest_path.exists():
            raise ObservationError(
                f"{manifest_path} is missing; run "
                "`python -m scripts.s03_build_features --config config/features.yaml`"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        by_block: dict[str, set[str]] = {}
        for col in manifest["columns"]:
            by_block.setdefault(col["block"], set()).add(col["name"])

        problems: list[str] = []

        def check(names: Sequence[str], block: str, where: str) -> None:
            known = by_block.get(block, set())
            for n in names:
                if n not in known:
                    near = sorted(k for k in known if k.split("_")[0] == n.split("_")[0])
                    hint = f"  did you mean one of {near[:4]}?" if near else ""
                    problems.append(f"{where}: {n!r} is not a {block} column.{hint}")

        check(self.per_asset_etf, "etf", "observation.per_asset.etf")
        check(self.per_asset_cross_sectional, "cross_sectional",
              "observation.per_asset.cross_sectional")
        check(self.macro, "macro", "observation.global.macro")
        check(self.global_cross_sectional, "cross_sectional",
              "observation.global.cross_sectional")

        if not self.per_asset_market:
            problems.append("observation.per_asset selects no market features at all")

        # The canonical ticker order is part of the policy's identity, so a mismatch
        # against the manifest is an error and not something to quietly reorder.
        canonical = tuple(manifest["canonical_tickers"])
        if self.tickers != canonical:
            problems.append(
                f"ticker order differs from the manifest: {self.tickers} != {canonical}. "
                "Per-asset blocks are positional; reordering silently invalidates a "
                "trained policy."
            )

        # The Markov floor. These are code-defined, so this can only fail if someone
        # edits the constants -- which is exactly when it should fail.
        missing = [f for f in MANDATORY_FIELDS if not any(
            n.endswith(f) for n in self.build_names())]
        if missing:
            problems.append(f"mandatory Markov fields absent from the layout: {missing}")

        if problems:
            raise ObservationError(
                "the declared observation does not match "
                f"{manifest_path}:\n  " + "\n  ".join(problems)
            )

        return {
            "manifest_generated_at": manifest.get("generated_at"),
            "selected": {
                "per_asset_etf": len(self.per_asset_etf),
                "per_asset_cross_sectional": len(self.per_asset_cross_sectional),
                "macro": len(self.macro),
                "global_cross_sectional": len(self.global_cross_sectional),
            },
            "available": {k: len(v) for k, v in by_block.items()},
            "obs_dim": self.size,
        }


def finite_or_raise(obs: np.ndarray, spec: ObservationSpec, *, where: str) -> None:
    """No NaN, no inf, ever. Named to the exact field, because `nan at index 913` is not
    a debuggable message when the vector is several hundred wide."""
    bad = ~np.isfinite(obs)
    if bad.any():
        names = spec.names or spec.build_names()
        idx = np.flatnonzero(bad)[:8]
        detail = ", ".join(f"{names[k]}={obs[k]!r}" for k in idx)
        raise ObservationError(
            f"{where}: observation has {int(bad.sum())} non-finite value(s): {detail}"
        )
