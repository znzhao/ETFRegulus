"""The RL model on the user's own holdings (web app, real-holdings mode).

In "my holdings" mode a baseline's plan is its simulated target, moved onto the user's
book. The RL model can do better than that: it is a function of the portfolio state, so
the browser runs it on the USER's state -- their weights, which positions are still
locked and for how long, their drawdown from their own 12-month high, their holdings
limit -- and then puts the result through the same rule layer as the research simulator.

Everything runs in the browser (`app/rl.js` in the site repository); nothing about the
user's holdings leaves it. This module is the reference implementation:

* `model_doc` writes what the browser needs: the actor weights of every seed, the scaled
  market features of the data date, and the observation layout. The risk samples come
  from `risk.json`, which already exists.
* `user_plan` is the decision the browser must reproduce, built from the same engine
  pieces the simulator uses (`cap_holdings`, `AnalyticProjector`, the observation
  formulas). The site's tests compare `app/rl.js` against cases generated from it.

The user's state stands in for a fresh episode: the NAV reference of `pf_nav_norm` is the
current value (so `pf_nav_norm` = 1 and `pf_peak_norm` = peak / value), exactly as at an
environment reset, where the peak is inherited and the NAV is the starting point.
Execution is whole shares; sales come first, and buys are scaled down to the cash.
"""

from __future__ import annotations

import base64
import datetime as dt
import math

import numpy as np

from src.constraints.projector import AnalyticProjector
from src.constraints.risk_envelope import _loss_from_paths, risk_budget
from src.env.etf_env import LOGIT_SCALE
from src.sim.engine import cap_holdings, exit_dropped

#: Action-mean floor for an unavailable asset (`src.agents.policy.MASK_LOGIT`).
MASK_LOGIT = -5.0
#: The smallest hold-days denominator in the observation (`state_builder._MIN_N`).
MIN_N = 1.0

ACTOR_LAYERS = {
    "enc1": "features_extractor.asset_encoder.0", "enc2": "features_extractor.asset_encoder.2",
    "ctx1": "features_extractor.context_encoder.0", "ctx2": "features_extractor.context_encoder.2",
    "head1": "mlp_extractor.asset_head.0", "head2": "mlp_extractor.asset_head.2",
    "cash1": "mlp_extractor.cash_head.0", "cash2": "mlp_extractor.cash_head.2",
}


# --------------------------------------------------------------------------- the document

def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode("ascii")


def _unb64(s: str, shape) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float64).reshape(shape)


def actor_weights(model) -> dict:
    sd = model.policy.state_dict()
    out = {}
    for name, key in ACTOR_LAYERS.items():
        w, b = sd[key + ".weight"].cpu().numpy(), sd[key + ".bias"].cpu().numpy()
        out[name] = {"shape": list(w.shape), "w": _b64(w), "b": _b64(b)}
    return out


def model_doc(models, spec, store, market, *, data_date: dt.date, meta: dict, rules: dict,
              schema_version: int) -> dict:
    """What `app/rl.js` needs to run the ensemble at the data date."""
    row = store.index_of_session(market.sessions[-1])
    per_asset, glob = store.at(row)
    return {
        "schema_version": schema_version, "data_date": data_date.isoformat(), "model": meta,
        "tickers": list(spec.tickers),
        "n_market": len(spec.per_asset_market), "n_per_asset": spec.n_per_asset,
        "n_global": len(spec.global_block), "n_portfolio": spec.portfolio_slice.stop - spec.portfolio_slice.start,
        "holdings_param": bool(spec.holdings_param),
        "hold_days_divisor": spec.hold_days_divisor, "clip": spec.clip,
        "logit_scale": LOGIT_SCALE, "mask_logit": MASK_LOGIT,
        "rules": rules,
        "inputs": {
            "global": [round(float(x), 6) for x in np.asarray(glob, float)],
            "per_asset": [[round(float(x), 6) for x in r] for r in np.asarray(per_asset, float)],
            "available": [bool(x) for x in market.available[-1]],
        },
        "seeds": [actor_weights(m) for m in models],
    }


# --------------------------------------------------------------------------- the forward pass

def _layer(doc_layer):
    w = _unb64(doc_layer["w"], doc_layer["shape"])
    b = _unb64(doc_layer["b"], (doc_layer["shape"][0],))
    return w, b


def policy_mean(seed: dict, obs: np.ndarray, doc: dict) -> np.ndarray:
    """The actor's masked action mean -- `SharedAssetActorCritic`, in numpy."""
    K, F, n_g, n_p = len(doc["tickers"]), doc["n_per_asset"], doc["n_global"], doc["n_portfolio"]
    L = {k: _layer(v) for k, v in seed.items()}
    lin = lambda x, wb: x @ wb[0].T + wb[1]
    glob = obs[:n_g]
    assets = obs[n_g:n_g + K * F].reshape(K, F)
    pf = obs[n_g + K * F:n_g + K * F + n_p]
    pr = obs[n_g + K * F + n_p:]
    emb = np.tanh(lin(np.tanh(lin(assets, L["enc1"])), L["enc2"]))
    m = assets[:, doc["n_market"] + 5:doc["n_market"] + 6]          # pf_available
    pooled_mean = (emb * m).sum(0) / max(m.sum(), 1.0)
    pooled_max = np.where(m > 0, emb, -1e4).max(0)
    ctx = np.tanh(lin(np.tanh(lin(np.concatenate([pooled_mean, pooled_max, glob, pf, pr]), L["ctx1"])), L["ctx2"]))
    asset_logit = lin(np.tanh(lin(np.concatenate([emb, np.tile(ctx, (K, 1))], axis=1), L["head1"])), L["head2"])[:, 0]
    cash_logit = lin(np.tanh(lin(ctx, L["cash1"])), L["cash2"])[0]
    mean = np.concatenate([[cash_logit], asset_logit])
    avail = np.concatenate([[1.0], m[:, 0]])
    return np.where(avail > 0.5, mean, doc["mask_logit"])


def softmax_weights(mean: np.ndarray, scale: float) -> np.ndarray:
    a = np.clip(mean, -1.0, 1.0) * scale
    e = np.exp(a - a.max())
    return e / e.sum()


# --------------------------------------------------------------------------- the user's state

def user_state(doc: dict, user: dict) -> dict:
    """Weights, locks, drawdown of the user's book at the data date's closes."""
    tickers = doc["tickers"]
    prices = user["prices"]
    data_date = dt.date.fromisoformat(doc["data_date"])
    n = int(user["hold_days"])
    shares = np.zeros(len(tickers))
    last_buy: dict[str, dt.date] = {}
    for h in user["holdings"]:
        if h["t"] not in tickers or not prices.get(h["t"], 0) > 0:
            continue
        j = tickers.index(h["t"])
        shares[j] += max(0.0, float(h["sh"]))
        if h.get("d"):
            d = dt.date.fromisoformat(h["d"])
            last_buy[h["t"]] = max(d, last_buy.get(h["t"], d))
    px = np.array([float(prices.get(t, 0.0)) for t in tickers])
    cash = max(0.0, float(user["cash"]))
    nav = cash + float((shares * px).sum())
    peak = max(float(user.get("peak") or 0.0), nav)
    remaining = np.zeros(len(tickers))
    for j, t in enumerate(tickers):
        if shares[j] > 0 and t in last_buy:
            remaining[j] = max(0, (last_buy[t] + dt.timedelta(days=n) - data_date).days)
    w = np.zeros(len(tickers) + 1)
    if nav > 0:
        w[1:] = shares * px / nav
        w[0] = cash / nav
    lower = np.zeros_like(w)
    lower[1:] = np.where(remaining > 0, w[1:], 0.0)
    return {"shares": shares, "px": px, "cash": cash, "nav": nav, "peak": peak,
            "drawdown": 1.0 - nav / peak if peak > 0 else 0.0,
            "remaining": remaining, "weights": w, "lower": lower}


def observation(doc: dict, st: dict, *, hold_days: int, max_drawdown: float,
                max_holdings: int) -> np.ndarray:
    """`state_builder.build_observation` for the user's state (nav_at_reset = NAV)."""
    K, n_market = len(doc["tickers"]), doc["n_market"]
    avail = np.array(doc["inputs"]["available"], dtype=float)
    per_asset = np.array(doc["inputs"]["per_asset"], dtype=float)
    w = st["weights"][1:]
    rem = st["remaining"]
    locked = rem > 0
    n = max(float(hold_days), MIN_N)
    block = np.zeros((K, doc["n_per_asset"]))
    block[:, :n_market] = np.where(avail[:, None] > 0, per_asset, 0.0)
    block[:, n_market + 0] = w
    block[:, n_market + 1] = st["shares"] > 0
    block[:, n_market + 2] = locked
    block[:, n_market + 3] = rem / n
    block[:, n_market + 4] = np.where(locked, 1.0 - np.minimum(rem, n) / n, 0.0)
    block[:, n_market + 5] = avail
    lw = w[locked].sum() if locked.any() else 0.0
    wavg = float((w[locked] * rem[locked]).sum() / lw) if locked.any() and lw > 1e-12 else 0.0
    nav, peak = st["nav"], st["peak"]
    portfolio = [st["weights"][0], 1.0, peak / max(nav, 1e-9), st["drawdown"],
                 max_drawdown - st["drawdown"], locked.sum() / max(K, 1), float(lw), wavg / n]
    params = [hold_days / doc["hold_days_divisor"], max_drawdown]
    if doc["holdings_param"]:
        k = K if max_holdings <= 0 else min(max_holdings, K)
        params.append(k / K)
    obs = np.concatenate([doc["inputs"]["global"], block.reshape(-1), portfolio, params])
    obs = obs.astype(np.float32).astype(np.float64)
    return np.clip(obs, -doc["clip"], doc["clip"])


class PublishedRisk:
    """The risk envelope from the published samples (`risk.json`)."""

    def __init__(self, risk_doc: dict, budget: float):
        self.paths = [np.asarray(s["paths"], dtype=float) for s in risk_doc["samples"]]
        self.q, self.measure, self.agg = risk_doc["quantile"], risk_doc["measure"], risk_doc["aggregation"]
        self._budget = budget

    def stress_loss(self, w) -> float:
        losses = [_loss_from_paths(p, np.asarray(w, float), self.q, self.measure) for p in self.paths]
        if self.agg == "max":
            return max(losses)
        if self.agg == "mean":
            return float(np.mean(losses))
        return float(np.quantile(losses, 0.9))

    def budget(self) -> float:
        return self._budget


# --------------------------------------------------------------------------- the plan

def user_plan(doc: dict, risk_doc: dict, user: dict) -> dict:
    """The model's next-open plan for the user's own book. The browser must match it."""
    hold_days, d_max = int(user["hold_days"]), float(user["ceiling"]) / 100
    k = int(user.get("max_holdings") or 0)
    rules = doc["rules"]
    st = user_state(doc, user)
    obs = observation(doc, st, hold_days=hold_days, max_drawdown=d_max, max_holdings=k)
    raw = np.mean([softmax_weights(policy_mean(s, obs, doc), doc["logit_scale"])
                   for s in doc["seeds"]], axis=0)

    cur, lower = st["weights"], st["lower"]
    avail = np.concatenate([[True], np.array(doc["inputs"]["available"], dtype=bool)])
    a = cap_holdings(raw, lower, cur, k)
    cp = st["drawdown"] > d_max
    risk = PublishedRisk(risk_doc, risk_budget(st["nav"], st["peak"], d_max))
    proj = AnalyticProjector(alpha_tolerance=rules.get("alpha_tolerance", 1e-3),
                             over_budget_rule=rules["over_budget_rule"],
                             derisk_step=rules["derisk_step"]).project(
        a, lower_bounds=lower, available=avail, risk=risk, capital_preservation=cp,
        current_weights=cur)
    target, diag = exit_dropped(proj.weights, a, lower, k), proj.diagnostics

    # Whole-share execution, the engine's rules: the band keeps small changes (a forced
    # de-risking keeps only small buys; a full exit under a holdings limit always goes),
    # a locked position is never cut, and over budget an asset not deliberately raised
    # is not bought (D-F).
    px, sh = st["px"], st["shares"]
    band = float(rules["no_trade_band"])
    new = sh.copy()
    for j in range(len(sh)):
        if not px[j] > 0:
            continue
        gap = target[j + 1] - cur[j + 1]
        keep = band > 0 and abs(gap) < band and (not diag.derisk_forced or gap >= 0)
        if k > 0 and target[j + 1] <= 1e-12 and sh[j] > 0:
            keep = False
        if keep:
            continue
        want = math.floor(target[j + 1] * st["nav"] / px[j] + 1e-9)
        if st["remaining"][j] > 0:
            want = max(want, sh[j])
        if diag.no_risk_increase and target[j + 1] <= cur[j + 1] + 1e-9:
            want = min(want, sh[j])
        new[j] = want
    sells = float(np.maximum(sh - new, 0) @ px)
    buys_sh = np.maximum(new - sh, 0)
    cash_avail = st["cash"] + sells
    cost = float(buys_sh @ px)
    if cost > cash_avail + 1e-9:
        scale = cash_avail / cost
        new = np.where(new > sh, sh + np.floor(buys_sh * scale + 1e-9), new)
    cash_after = st["cash"] + float((sh - new) @ px)
    return {
        "nav": st["nav"], "peak": st["peak"], "drawdown": st["drawdown"],
        "budget": risk.budget(), "raw": raw, "target": target,
        "shares": new, "cash_after": cash_after,
        "risk_state": ("capital_preservation" if cp
                       else "over_budget" if diag.no_risk_increase else "normal"),
        "derisk_forced": bool(diag.derisk_forced),
    }
