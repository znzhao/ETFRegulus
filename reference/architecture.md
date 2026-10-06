# Architecture

Repo layout, the config system, the stage harness, and the artifact contracts that let every stage run
independently.

---

## 1. Layout

```text
ETFRegulus/
├── IMPLEMENTATION_PLAN.md      # the map
├── STATUS.md                   # the tracker
├── requirements.txt
├── reference/                  # this folder — all design depth
│
├── config/
│   ├── universe.yaml           # tradable + feature_only symbols, kinds, inception
│   ├── features.yaml           # feature definitions, windows, scaling policy
│   ├── constraints.yaml        # lock, drawdown, projection backend, risk estimators
│   ├── training.yaml           # PPO hyperparameters, curriculum stages
│   ├── evaluation.yaml         # walk-forward folds, metrics, acceptance thresholds
│   └── experiments/            # one file per named run; inherits from the above
│
├── data/                       # git-ignored (D6)
│   ├── raw/                    # exactly as fetched, never edited
│   ├── curated/                # aligned, validated, calendar-reindexed
│   └── features/               # feature matrices + per-fold scalers
│
├── scripts/                    # THE ENTRY POINTS (D1) — one file per stage
│   ├── s00_check_env.py
│   ├── s01_fetch_data.py
│   ├── s02_curate_data.py
│   ├── s03_build_features.py
│   ├── s04_simulate.py
│   ├── s05_run_baselines.py
│   ├── s06_smoke_env.py
│   ├── s07_train_ppo.py
│   ├── s08_walk_forward.py
│   ├── s09_stress.py
│   ├── s10_bootstrap.py
│   ├── s11_adversarial.py
│   ├── s12_report.py
│   └── s13_daily_inference.py  # reserved, not implemented (D4)
│
├── src/
│   ├── cli/stage.py            # shared harness: args, config, manifest, staleness
│   ├── config/                 # typed config loading + validation
│   ├── data/                   # loaders, alignment, validation
│   ├── features/               # etf, ohlcv, macro, cross_sectional, builder
│   ├── portfolio/              # ledger, valuation, execution, lock_manager
│   ├── constraints/            # availability, portfolio_constraints, risk_envelope,
│   │                           #   projector, safety_layer
│   ├── env/                    # etf_env, state_builder, transition, reset_sampler
│   ├── agents/                 # policy, normalization  (PPO itself comes from SB3, D2)
│   ├── training/               # curriculum, trainer, yearly_cv, model_selection
│   ├── evaluation/             # metrics, walk_forward, stress, bootstrap, adversarial
│   └── reporting/              # reports, plots
│
├── tests/                      # mirrors src/; see testing.md
└── artifacts/
    ├── runs/<run_id>/          # per-run manifests, checkpoints, tensorboard
    └── reports/                # generated reports and plots
```

`scripts/` is what makes D1 work: `src/` holds importable library
code with no side effects at import time; `scripts/` holds the things you run.

---

## 2. The stage harness

Every entry point is a thin file. All of them look like this:

```python
# scripts/s04_simulate.py
from src.cli.stage import stage

@stage(
    name="s04_simulate",
    config_default="config/sim/default.yaml",
    inputs=["data/curated/prices.parquet", "data/features/etf.parquet"],
    outputs=["artifacts/runs/{run_id}/trajectory.parquet"],
    upstream="s03_build_features",
)
def main(cfg, ctx):
    ...

if __name__ == "__main__":
    main()
```

The decorator provides, uniformly, for free:

| Behavior | Detail |
|---|---|
| `--config PATH` | Required-with-a-default; loaded, validated, and snapshotted |
| `--dry-run` | Print resolved config + input/output paths, then exit 0 without side effects |
| `--force` | Run even if outputs exist and are newer than inputs |
| `--seed N` | Seeds `random`, `numpy`, and `torch`; recorded in the manifest |
| Staleness check | If an input is missing or older than *its* inputs, fail with the exact command to run first |
| Run id | `<stage>_<UTC timestamp>_<config hash[:8]>` |
| Manifest | Written on both success and failure, with the outcome recorded |
| Logging | Structured to stdout and to `artifacts/runs/<run_id>/log.txt` |

**This harness is built in Stage 0.** Everything after depends on it, and it is roughly 150 lines.

---

## 3. Config system

YAML, layered, typed. `config/experiments/*.yaml` inherit and override the five base files:

```yaml
# config/experiments/ppo_stage3.yaml
extends: [training.yaml, constraints.yaml]
run_name: ppo_stage3_full_params
training:
  curriculum_stage: 3
  total_timesteps: 5_000_000
constraints:
  lock:
    hold_days: {values: [15, 18, 21, 25, 30, 36, 42, 50, 60]}   # the D16 operating range
  drawdown:
    max_drawdown: {values: [0.05, 0.10, 0.15, 0.20, 0.25]}
```

Rules:

- Configs load into **typed dataclasses** (`src/config/`), not raw dicts. An unknown key is an error, not a
  silent no-op — a typo in a hyperparameter name must not cost a training run.
- The resolved config is hashed; the hash goes into the run id and the manifest.
- No code reads an environment variable except the data layer, which reads API keys from `.env`.
- Anything a person might tune is in config. Nothing tunable is a literal in `src/`.

---

## 4. Artifact contracts

The stage boundaries. A stage may read only from its declared inputs, and its outputs must be complete on disk
before the next stage runs. This is what makes stages independently runnable.

### `data/curated/prices.parquet`

Long format, one row per `(session, ticker)`.

| Column | Type | Notes |
|---|---|---|
| `session` | `date` | tz-naive; from `XNYS` (see [data-pipeline.md](data-pipeline.md)) |
| `ticker` | `category` | |
| `close_adj` | `float64` | total return; **features and returns only** |
| `close_raw` | `float64` | quoted price; **execution and valuation only** |
| `open_raw`, `high_raw`, `low_raw` | `float64` | tradable tickers only |
| `volume` | `float64` | tradable tickers only |
| `is_tradable` | `bool` | false for feature-only symbols |

### `data/curated/inception.parquet`

| Column | Type |
|---|---|
| `ticker` | `category` |
| `first_session` | `date` |

The availability mask is derived from this and nowhere else.

### `data/features/{etf,cross_sectional,macro}.parquet`

Wide, indexed by `session` (and `ticker` for `etf`). Every column is documented in
[features.md](features.md). **Unscaled** — scaling is fold-dependent and lives in the scaler artifact.

### `data/features/scalers/<fold_id>.json`

Fitted on the training window of that fold only. Loading a scaler fitted on a window that overlaps the
evaluation period is a lookahead bug and is checked by a test (see [testing.md](testing.md)).

### `artifacts/runs/<run_id>/trajectory.parquet`

The universal backtest output — produced identically by Stage 4, 5, 8, and 9, so every downstream metric
function has exactly one input format.

| Column | Notes |
|---|---|
| `session` | |
| `nav` | total-return NAV at close |
| `peak_nav` | running maximum |
| `drawdown` | `1 - nav/peak_nav` |
| `cash` | |
| `w_<ticker>` | executed weight, one column per tradable ticker |
| `shares_<ticker>` | executed share count |
| `locked_<ticker>` | bool, at decision time |
| `unlock_date_<ticker>` | date or null |
| `proj_distance` | `||a_proj - a_raw||_1` |
| `safety_intervened` | bool — the risk envelope bound |
| `capital_preservation` | bool — drawdown-breached capital-preservation mode active |
| `n_param`, `dmax_param` | the conditioning parameters in force |
| `reward` | |

### `artifacts/runs/<run_id>/manifest.json`

```json
{
  "run_id": "s07_train_ppo_20260901T031500Z_a3f19c22",
  "stage": "s07_train_ppo",
  "git_sha": "634e2a3", "git_dirty": false,
  "config_hash": "a3f19c22", "config_path": "config/experiments/ppo_stage3.yaml",
  "seed": 42,
  "inputs": [{"path": "data/features/etf.parquet", "sha256": "...", "mtime": "..."}],
  "outputs": ["artifacts/runs/.../policy.zip"],
  "started_at": "...", "finished_at": "...", "status": "success",
  "python": "3.11.9", "packages": {"stable-baselines3": "2.x", "torch": "2.x"}
}
```

`git_dirty: true` on any run whose result is reported is a red flag; Stage 12 refuses to include such runs in
a report unless `--allow-dirty` is passed.

---

## 5. Determinism

Reproducibility is a hard acceptance criterion, so:

- One seed per run, from `--seed`, recorded in the manifest, applied to `random` / `numpy` / `torch` / the
  env's own RNG (`gymnasium.utils.seeding`).
- Each vectorized env worker derives its seed as `seed + worker_index` — never shares an RNG.
- The reset sampler takes its own RNG stream, so changing the number of workers does not change which initial
  states are drawn.
- Any nondeterminism that remains (GPU kernel nondeterminism) is documented, not hidden. Rerunning a training
  run with the same seed should reproduce metrics to within a tolerance stated in the report, and Stage 12
  reports the observed spread across seeds rather than a single number.
