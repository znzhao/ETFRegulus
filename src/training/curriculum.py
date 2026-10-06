"""The five-stage curriculum, and the gate between stages.

reference/rl-training.md section 4. Each stage adds exactly one source of difficulty, so a
regression localizes to the thing just added:

    1  N = 0, D_max fixed, flat start, risk off   -- portfolio mechanics, nothing else
    2  + the lock, N in {21, 30, 42}
    3  + the full (N, D_max) grid
    4  + the drawdown safety envelope
    5  + reservoir starts and sampled episode lengths

Stage 1 is the diagnostic one. With no lock, no envelope and a fixed ceiling, an agent that
cannot beat holding cash is not evidence of a hard problem -- it is evidence of a bug in
the simulator or the reward, and the curriculum exists partly to make that unambiguous.

**The gate is not advisory.** A stage advances only if it beats the `cash` baseline on its
own training window and records zero lock and zero feasibility violations. Continuing past
a violation trains the next stage against a constraint layer that is known broken, and the
resulting policy is not a weaker result, it is an invalid one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CurriculumStage:
    """One rung. `env` overrides land on `EnvConfig`, `ppo` on the PPO kwargs."""

    index: int
    name: str
    what_is_added: str
    env: dict = field(default_factory=dict)
    total_timesteps: int | None = None

    @property
    def config_name(self) -> str:
        return f"ppo_stage{self.index}.yaml"


#: The canonical ladder. `hold_days_values` / `max_drawdown_values` are *overrides*; a
#: stage that does not name one inherits the D16 grid from `config/constraints.yaml`, so
#: the operating range lives in exactly one place.
CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage(
        1, "mechanics", "portfolio mechanics only: no lock, fixed ceiling, no envelope",
        env=dict(hold_days_values=(0,), hold_days_weights=None,
                 max_drawdown_values=(0.20,), max_drawdown_weights=None,
                 episode_lengths=(252,), risk_enabled=False,
                 flat_start=True, stress_reset=False),
    ),
    CurriculumStage(
        2, "lock", "the holding lock, at three values around the primary",
        env=dict(hold_days_values=(21, 30, 42), hold_days_weights=None,
                 max_drawdown_values=(0.20,), max_drawdown_weights=None,
                 episode_lengths=(252,), risk_enabled=False, flat_start=True),
    ),
    CurriculumStage(
        3, "parameters", "the full (N, D_max) grid -- the policy must condition on both",
        env=dict(episode_lengths=(252,), risk_enabled=False, flat_start=True),
    ),
    CurriculumStage(
        4, "envelope", "the drawdown safety envelope inside env.step",
        env=dict(episode_lengths=(252,), risk_enabled=True, flat_start=True),
    ),
    CurriculumStage(
        5, "randomized", "reservoir initial states and sampled episode lengths",
        env=dict(episode_lengths=(63, 126, 252, 504), risk_enabled=True,
                 flat_start=False),
    ),
)

STAGES_BY_INDEX = {s.index: s for s in CURRICULUM}


@dataclass
class GateResult:
    """Whether a stage may advance, and why."""

    stage: int
    beats_cash: bool
    lock_violations: int
    feasibility_violations: int
    episode_log_return: float
    cash_log_return: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (self.beats_cash and self.lock_violations == 0
                and self.feasibility_violations == 0)

    def to_dict(self) -> dict:
        return {"stage": self.stage, "passed": self.passed,
                "beats_cash": self.beats_cash,
                "lock_violations": self.lock_violations,
                "feasibility_violations": self.feasibility_violations,
                "episode_log_return": self.episode_log_return,
                "cash_log_return": self.cash_log_return,
                "notes": self.notes}

    def explain(self) -> str:
        if self.passed:
            return (f"stage {self.stage} passed: episode log return "
                    f"{self.episode_log_return:+.4f} > cash {self.cash_log_return:+.4f}, "
                    "zero violations")
        why = []
        if not self.beats_cash:
            why.append(
                f"did not beat cash ({self.episode_log_return:+.4f} vs "
                f"{self.cash_log_return:+.4f}). On stage 1 in particular this points at "
                "the simulator or the reward, not at the policy")
        if self.lock_violations:
            why.append(f"{self.lock_violations} lock violation(s)")
        if self.feasibility_violations:
            why.append(f"{self.feasibility_violations} feasibility violation(s)")
        return f"stage {self.stage} FAILED the gate: " + "; ".join(why)


def evaluate_gate(stage: int, diagnostics: dict, *, cash_log_return: float = 0.0,
                  lock_violations: int = 0,
                  feasibility_violations: int = 0) -> GateResult:
    """`cash` earns exactly zero log return by construction, so the bar is > 0."""
    episode_return = float(diagnostics.get("episode_log_return", 0.0))
    result = GateResult(
        stage=stage,
        beats_cash=episode_return > cash_log_return,
        lock_violations=int(lock_violations),
        feasibility_violations=int(feasibility_violations),
        episode_log_return=episode_return,
        cash_log_return=float(cash_log_return),
    )
    if diagnostics.get("n_episodes", 0) < 10:
        result.notes.append(
            f"only {diagnostics.get('n_episodes', 0)} completed episodes in the window; "
            "the gate is measured on a small sample")
    rate = diagnostics.get("safety_intervened", 0.0)
    if rate > 0.9:
        result.notes.append(
            f"safety intervention rate {rate:.2f} -- the envelope may be "
            "over-calibrated (risk-envelope.md section 7)")
    cash_weight = diagnostics.get("cash_weight", 0.5)
    if cash_weight > 0.98 or cash_weight < 0.005:
        result.notes.append(
            f"mean cash weight {cash_weight:.3f} is pinned; the policy may have collapsed")
    return result
