"""Environment fixtures.

The bundle is session-scoped and built once. It reads the real curated prices and the real
feature frames rather than a fixture, because the thing under test *is* the alignment
between the price calendar, the feature calendar and the canonical ticker order -- a
synthetic fixture would agree with itself by construction and prove none of it.

The window is deliberately short. These tests are about mechanics, not about performance
over history, and the suite has a 60-second budget.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.env.etf_env import EnvConfig
from src.env.factory import build_bundle, make_env

CONFIG = "config/training.yaml"
START, END = "2008-01-02", "2009-12-31"   # inside fold_2012's training window

pytestmark = pytest.mark.skipif(
    not Path("data/features/feature_manifest.json").exists(),
    reason="run Stages 1-3 first")


def _available() -> bool:
    return (Path("data/features/feature_manifest.json").exists()
            and Path("data/curated/prices.parquet").exists())


needs_data = pytest.mark.skipif(not _available(), reason="run Stages 1-3 first")


@pytest.fixture(scope="session")
def bundle():
    if not _available():
        pytest.skip("run Stages 1-3 first")
    return build_bundle(CONFIG, start=START, end=END)


@pytest.fixture(scope="session")
def obs_spec(bundle):
    return bundle.spec


@pytest.fixture
def env(bundle):
    """Risk off by default: the envelope is exercised in its own tests and in Stage 6,
    and it costs about 1 ms of the 1.6 ms step."""
    cfg = EnvConfig(**{**bundle.env_cfg.__dict__,
                       "risk_enabled": False, "episode_lengths": (30,)})
    return make_env(bundle, seed=7, env_cfg=cfg)


@pytest.fixture
def risky_env(bundle):
    cfg = EnvConfig(**{**bundle.env_cfg.__dict__, "episode_lengths": (20,)})
    return make_env(bundle, seed=7, env_cfg=cfg)
