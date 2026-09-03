"""T16 -- the stage harness.

Everything downstream depends on this, so the three properties that make it trustworthy
are tested directly: `--dry-run` has no side effects, staleness detection fires, and a
manifest is written when a stage *fails* (a failed run is still a run, and it is recorded).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from src.cli import stage as stage_mod
from src.cli.stage import StageContext, outputs_are_fresh, stage
from src.config.loader import ConfigError, load_typed, resolve_config, strict_from_dict


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "runs"
    monkeypatch.setattr(stage_mod, "RUNS", d)
    return d


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"alpha": 1, "beta": "two"}), encoding="utf-8")
    return p


# ------------------------------------------------------------------- --dry-run


def test_dry_run_has_no_side_effects(runs_dir, cfg_file, tmp_path, capsys):
    """A dry run must not create the run dir, the log, the manifest, or any output."""
    out = tmp_path / "out.parquet"
    touched = []

    @stage(name="t_dry", config_default=str(cfg_file), outputs=[str(out)])
    def body(cfg, ctx):
        touched.append(True)
        out.write_text("side effect", encoding="utf-8")

    assert body(["--dry-run"]) == 0
    assert not touched, "the stage body ran during a dry run"
    assert not out.exists()
    assert not runs_dir.exists(), "a dry run created the run directory"

    printed = capsys.readouterr().out
    assert "dry run: t_dry" in printed
    assert '"alpha": 1' in printed, "the resolved config was not printed"
    assert str(out) in printed, "the output plan was not printed"


def test_dry_run_reports_missing_inputs_without_failing(runs_dir, cfg_file, tmp_path, capsys):
    missing = tmp_path / "data" / "curated" / "prices.parquet"

    @stage(name="t_dry2", config_default=str(cfg_file), inputs=[str(missing)])
    def body(cfg, ctx):
        raise AssertionError("must not run")

    assert body(["--dry-run"]) == 0
    assert "MISSING" in capsys.readouterr().out


# ------------------------------------------------------------------- staleness


def test_missing_input_refuses_and_names_the_producer(runs_dir, cfg_file, capsys):
    """Refusing is the point; naming the command to run first is what makes it useful."""

    @stage(
        name="t_stale", config_default=str(cfg_file),
        # A path that cannot exist, but that still resolves to the s02 producer, so the
        # test does not depend on whether the pipeline has been run in this working tree.
        inputs=["data/curated/__absent__.parquet"], upstream="s02_curate_data",
    )
    def body(cfg, ctx):
        raise AssertionError("must not run with a missing input")

    assert body([]) == 2
    err = capsys.readouterr().err
    assert "missing required input" in err
    assert "scripts.s02_curate_data" in err, "the fix command was not printed"
    assert not runs_dir.exists()


def test_force_overrides_a_missing_input(runs_dir, cfg_file, tmp_path):
    ran = []

    @stage(name="t_force", config_default=str(cfg_file),
           inputs=[str(tmp_path / "nope.parquet")])
    def body(cfg, ctx):
        ran.append(True)

    assert body(["--force"]) == 0
    assert ran


def test_fresh_outputs_short_circuit_and_force_reruns(runs_dir, cfg_file, tmp_path):
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    src.write_text("in", encoding="utf-8")
    dst.write_text("out", encoding="utf-8")
    import os
    import time

    os.utime(dst, (time.time() + 10, time.time() + 10))  # output newer than input

    runs = []

    @stage(name="t_fresh", config_default=str(cfg_file),
           inputs=[str(src)], outputs=[str(dst)])
    def body(cfg, ctx):
        runs.append(True)

    assert body([]) == 0
    assert not runs, "a stage with fresh outputs re-ran"
    assert body(["--force"]) == 0
    assert runs, "--force did not re-run the stage"


def test_outputs_are_fresh_detects_a_stale_output(tmp_path):
    import os
    import time

    src, dst = tmp_path / "a", tmp_path / "b"
    src.write_text("a", encoding="utf-8")
    dst.write_text("b", encoding="utf-8")
    os.utime(src, (time.time() + 10, time.time() + 10))  # input newer than output
    assert not outputs_are_fresh([src], [dst])
    assert not outputs_are_fresh([src], [tmp_path / "missing"])


# -------------------------------------------------------------------- manifest


def test_manifest_written_on_success(runs_dir, cfg_file, tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("payload", encoding="utf-8")

    @stage(name="t_ok", config_default=str(cfg_file), inputs=[str(src)])
    def body(cfg, ctx):
        ctx.record(rows=7)

    assert body(["--seed", "123"]) == 0
    manifest = json.loads(_only_manifest(runs_dir).read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["stage"] == "t_ok"
    assert manifest["seed"] == 123
    assert manifest["rows"] == 7
    assert manifest["inputs"][0]["sha256"], "input artifacts must be hashed"
    assert "wall_seconds" in manifest


def test_manifest_written_on_failure(runs_dir, cfg_file):
    """A stage that raises must still leave a manifest recording the failure."""

    @stage(name="t_fail", config_default=str(cfg_file))
    def body(cfg, ctx):
        raise ValueError("deliberate")

    assert body([]) == 1
    manifest = json.loads(_only_manifest(runs_dir).read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "deliberate" in manifest["error"]
    log = (_only_manifest(runs_dir).parent / "log.txt").read_text(encoding="utf-8")
    assert "Traceback" in log, "the traceback was not captured in the run log"


def test_run_id_and_config_snapshot(runs_dir, cfg_file):
    @stage(name="t_id", config_default=str(cfg_file))
    def body(cfg, ctx):
        pass

    assert body([]) == 0
    run_dir = _only_manifest(runs_dir).parent
    assert run_dir.name.startswith("t_id_")
    chash = run_dir.name.rsplit("_", 1)[1]
    assert len(chash) == 8
    snapshot = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot == {"alpha": 1, "beta": "two"}


def test_seed_is_applied(runs_dir, cfg_file):
    seen = []

    @stage(name="t_seed", config_default=str(cfg_file))
    def body(cfg, ctx):
        import numpy as np

        seen.append(np.random.rand())

    body(["--seed", "7"])
    body(["--seed", "7"])
    body(["--seed", "8"])
    assert seen[0] == seen[1], "the same seed produced different draws"
    assert seen[0] != seen[2]


def _only_manifest(runs_dir: Path) -> Path:
    found = sorted(runs_dir.glob("*/manifest.json"))
    assert len(found) == 1, f"expected exactly one manifest, found {found}"
    return found[0]


# ---------------------------------------------------------------- config layer


def test_unknown_key_is_an_error(tmp_path):
    """A typo in a hyperparameter name must not cost a training run."""

    @dataclass
    class Cfg:
        alpha: int
        beta: str = "b"

    assert strict_from_dict(Cfg, {"alpha": 1}).alpha == 1
    with pytest.raises(ConfigError, match="unknown key"):
        strict_from_dict(Cfg, {"alpha": 1, "alfa": 2})
    with pytest.raises(ConfigError, match="required key is missing"):
        strict_from_dict(Cfg, {"beta": "x"})
    with pytest.raises(ConfigError, match="expected int"):
        strict_from_dict(Cfg, {"alpha": "not an int"})


def test_extends_layers_and_child_wins(tmp_path):
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"a": 1, "nested": {"x": 1, "y": 2}}), encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({"extends": "base.yaml", "nested": {"y": 99}}), encoding="utf-8")
    resolved = resolve_config(tmp_path / "child.yaml")
    assert resolved == {"a": 1, "nested": {"x": 1, "y": 99}}


def test_circular_extends_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump({"extends": "b.yaml"}), encoding="utf-8")
    (tmp_path / "b.yaml").write_text(yaml.safe_dump({"extends": "a.yaml"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="circular"):
        resolve_config(tmp_path / "a.yaml")


def test_real_configs_load_and_validate():
    """The shipped configs must satisfy their own schemas."""
    from src.config.schema import FeaturesConfig, UniverseConfig

    universe, _ = load_typed("config/universe.yaml", UniverseConfig)
    features, _ = load_typed("config/features.yaml", FeaturesConfig)

    assert len(universe.tradable_tickers) == 24
    assert len(set(universe.tradable_tickers)) == 24, "duplicate ticker in the universe"
    assert universe.synthetic_asset == "CASH"
    assert universe.synthetic_asset not in universe.tradable_tickers
    assert features.universe_config == "config/universe.yaml"


def test_universe_config_matches_the_reference_document():
    """config/universe.yaml is canonical; reference/etf-universe.md is the rationale.

    They are kept in sync by this assertion rather than by care.
    """
    import re

    from src.config.schema import UniverseConfig

    universe, _ = load_typed("config/universe.yaml", UniverseConfig)
    doc = Path("reference/etf-universe.md").read_text(encoding="utf-8")

    block = re.search(r"```yaml\ntradable:\n(.*?)```", doc, re.S)
    assert block, "the machine-readable tradable block is missing from etf-universe.md"
    documented = set(re.findall(r"\b[A-Z]{2,4}\b", block.group(1))) - {"CASH"}
    assert documented == set(universe.tradable_tickers)
