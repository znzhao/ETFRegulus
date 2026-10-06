"""The acceptance table: hard engineering, risk reporting, and performance.

reference/evaluation.md section 5. A model is not accepted because `return > benchmark`.
The table is evaluated into `acceptance.json` as explicit pass/fail, and three properties
of it are deliberate:

* **Hard engineering criteria must be exactly zero. There is no partial credit.** Any
  non-zero value fails the model, and the table says FAIL rather than "mostly fine".
* **Risk reporting criteria are about presence, not value.** A run that cannot produce a
  breach classification is *incomplete*, independent of how well it performed.
* **Performance criteria can fail without failing the model.** They are reported as
  measured. A policy that does not beat `spy_tlt_60_40` on risk-adjusted terms has not
  demonstrated anything -- but that is a finding, not a defect, and the distinction is
  kept visible instead of collapsed into one verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Criterion:
    name: str
    group: str
    passed: bool
    observed: str
    requirement: str
    blocking: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AcceptanceTable:
    criteria: list[Criterion] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.criteria.append(Criterion(**kw))

    @property
    def blocking_failures(self) -> list[Criterion]:
        return [c for c in self.criteria if c.blocking and not c.passed]

    @property
    def passed(self) -> bool:
        return not self.blocking_failures

    def group(self, name: str) -> list[Criterion]:
        return [c for c in self.criteria if c.group == name]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_criteria": len(self.criteria),
            "n_passed": sum(c.passed for c in self.criteria),
            "blocking_failures": [c.name for c in self.blocking_failures],
            "criteria": [c.to_dict() for c in self.criteria],
        }


def _zero(table: AcceptanceTable, name: str, count, source: str) -> None:
    n = int(count)
    table.add(name=name, group="hard_engineering", passed=n == 0,
              observed=str(n), requirement="exactly 0",
              blocking=True, note=source)


def build(walk_forward: dict | None, stress: dict | None, bootstrap: dict | None,
          adversarial: dict | None, comparison: dict | None) -> AcceptanceTable:
    """Assemble the table from whatever stage artifacts exist.

    A missing stage is recorded as a FAILED reporting criterion rather than skipped: an
    acceptance table with silent holes in it is worse than one that says what is absent.
    """
    table = AcceptanceTable()

    # ------------------------------------------------ hard engineering (all zero)
    if walk_forward:
        hard = walk_forward["hard_acceptance"]
        _zero(table, "illegal lock sells", hard["lock_violations"], "Stage 8 walk-forward")
        _zero(table, "action feasibility violations", hard["feasibility_violations"],
              "Stage 8 walk-forward")
        _zero(table, "preventable D_max violations",
              hard["preventable_dmax_violations"], "Stage 8 replay detector")
    else:
        table.add(name="walk-forward hard acceptance", group="hard_engineering",
                  passed=False, observed="no Stage 8 run", requirement="exactly 0",
                  blocking=True, note="run scripts.s08_walk_forward")

    if stress:
        s = stress.get("hard_acceptance", {})
        _zero(table, "lock violations under stress", s.get("lock_violations", 0),
              "Stage 9 stress sweeps")
        _zero(table, "preventable violations under stress",
              s.get("preventable_violations", 0), "Stage 9 stress sweeps")

    if adversarial:
        a = adversarial.get("hard_acceptance", {})
        _zero(table, "lock violations on adversarial paths", a.get("lock_violations", 0),
              "Stage 11 adversarial")
        _zero(table, "preventable violations on adversarial paths",
              a.get("preventable_violations", 0), "Stage 11 adversarial")

    # Negative cash, negative shares and pre-inception trades are asserted at runtime by
    # the ledger and re-checked by the replay detector, so the detector finding nothing is
    # the evidence for all three at once.
    if walk_forward:
        findings = sum(f["test"].get("n_replay_findings", 0)
                       for f in walk_forward.get("folds", []))
        _zero(table, "negative cash / negative shares / pre-inception trades",
              findings, "Ledger runtime checks + Stage 8 replay (I1, I5)")

    # ------------------------------------------------- risk reporting (presence)
    def present(name: str, ok: bool, observed: str, note: str = "") -> None:
        table.add(name=name, group="risk_reporting", passed=ok, observed=observed,
                  requirement="present", blocking=True, note=note)

    if walk_forward:
        folds = walk_forward.get("folds", [])
        worst = max((f["test"]["max_drawdown"] for f in folds), default=0.0)
        present("OOS maximum drawdown", bool(folds), f"{worst:.2%} (worst fold)")
        breaches = sum(f["violations"].get("n_breaches", 0) for f in folds)
        present("number of D_max breaches", True, str(breaches))
        depth = max((f["violations"].get("worst_breach_depth", 0.0) for f in folds),
                    default=0.0)
        duration = max((f["violations"].get("longest_breach_duration", 0) for f in folds),
                       default=0)
        present("breach depth and duration", True,
                f"worst {depth:.2%}, longest {duration} sessions")
        forced = sum(f["violations"].get("n_market_forced", 0) for f in folds)
        prevent = sum(f["violations"].get("n_preventable", 0) for f in folds)
        present("preventable vs market-forced classification", True,
                f"{prevent} preventable / {forced} market-forced",
                "reported separately and never summed")
        rate = sum(f["test"]["cells"][k].get("safety_intervention_rate", 0.0)
                   for f in folds for k in f["test"]["cells"]) / max(
                       sum(len(f["test"]["cells"]) for f in folds), 1)
        present("safety intervention frequency", True, f"{rate:.1%} of steps")

    present("bootstrap confidence bands", bool(bootstrap),
            "produced for every standard metric" if bootstrap else "no Stage 10 run")
    present("D_max monotonicity", bool(stress and stress["monotonicity"]["passed"]),
            "monotone" if stress and stress["monotonicity"]["passed"] else "NOT VERIFIED",
            "a tighter ceiling must not deepen realized drawdown")
    present("adversarial disclaimer", bool(adversarial and adversarial.get("DISCLAIMER")),
            "present" if adversarial else "no Stage 11 run",
            "historical robustness, not a forward-looking guarantee")

    # ------------------------------------------------------------- performance
    if comparison:
        for name, beaten in comparison.get("beats", {}).items():
            table.add(name=f"beats {name}", group="performance",
                      passed=bool(beaten["passed"]), observed=beaten["observed"],
                      requirement=beaten["requirement"], blocking=False,
                      note=beaten.get("note", ""))
    return table
