"""Rendering the comparison report as a standalone page.

Kept out of `scripts/s12_report.py` so the stage stays readable: that file decides *what*
is in the report, this one decides how it looks.

Two pieces of information design that a plain table cannot do, and that matter because the
core artifact is a 13-year by 6-strategy matrix repeated eleven times:

* **Heat tinting.** Every numeric cell carries a background tint proportional to its value
  within its own table, gains and losses on opposite hues. A reader finds 2022 or the
  worst drawdown by looking, not by reading 78 numbers.
* **Stacked allocation bars.** The seven category weights per strategy are also drawn as
  one bar, so "what does this strategy actually hold" is a glance rather than a
  reconstruction from seven columns.

The page is emitted as a fragment (`<title>`, fonts, `<style>`, content) rather than a
full document: browsers render it standalone perfectly well, and it is also directly
publishable as an artifact, which wraps content in its own skeleton.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.categories import CATEGORIES

#: One muted hue per category. Chosen to stay distinguishable on both a near-white and a
#: near-black ground, and deliberately not a rainbow -- these are one portfolio's parts.
CATEGORY_COLORS: dict[str, str] = {
    "US Broad Equity": "#1f5f6b",
    "US Sector Equity": "#3d8b96",
    "Interest Rate": "#7a6a9e",
    "Credit": "#a8735a",
    "Commodities": "#b8973f",
    "International Equity": "#5b8c5a",
    "CASH": "#9aa3ad",
}

_POS = (18, 112, 90)     # deep green
_NEG = (163, 53, 44)     # brick
_NEU = (15, 110, 120)    # petrol, for magnitude-only scales


def _tint(value: float, vmax: float, rgb: tuple[int, int, int], cap: float = 0.30) -> str:
    """A background tint whose alpha is the value's share of the table's extreme.

    Alpha rather than a solid colour so the same tint reads on both the light and the dark
    ground without a second palette.
    """
    if not np.isfinite(value) or vmax <= 1e-12:
        return ""
    alpha = min(abs(value) / vmax, 1.0) * cap
    if alpha < 0.012:
        return ""
    return f"background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f})"


def _num_table(frame: pd.DataFrame, decimals: int, index_name: str, *,
               signed: bool, invert: bool = False) -> str:
    """`signed` colours gains and losses apart; otherwise magnitude on one hue.

    `invert` is for drawdown, where a larger number is worse -- the tint follows badness,
    not size, so a reader never has to remember which way a column runs.
    """
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 0.0

    head = "".join(f"<th scope='col'>{c}</th>" for c in frame.columns)
    rows = []
    for idx in frame.index:
        cells = []
        for v in frame.loc[idx]:
            if not np.isfinite(v):
                cells.append("<td class='nil'>—</td>")
                continue
            if signed:
                rgb = _POS if v >= 0 else _NEG
                cls = " neg" if v < 0 else ""
            else:
                rgb = _NEG if invert else _NEU
                cls = ""
            cells.append(
                f"<td class='n{cls}' style='{_tint(v, vmax, rgb)}'>{v:.{decimals}f}</td>")
        rows.append(f"<tr><th scope='row'>{idx}</th>{''.join(cells)}</tr>")

    return (f"<div class='scroll'><table><thead><tr>"
            f"<th scope='col'>{index_name}</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _alloc_bars(summary: pd.DataFrame) -> str:
    """One stacked bar per strategy: what it actually held, over the whole window."""
    legend = "".join(
        f"<span class='key'><i style='background:{CATEGORY_COLORS[c]}'></i>{c}</span>"
        for c in CATEGORIES)
    rows = []
    for name in summary.index:
        segs = []
        for c in CATEGORIES:
            pct = float(summary.loc[name, c])
            if pct < 0.25:
                continue
            label = f"{c} {pct:.1f}%"
            segs.append(
                f"<span class='seg' style='width:{pct:.4f}%;"
                f"background:{CATEGORY_COLORS[c]}' title='{label}'>"
                f"<span class='segtxt'>{pct:.0f}</span></span>")
        rows.append(f"<div class='barrow'><span class='barname'>{name}</span>"
                    f"<span class='bar'>{''.join(segs)}</span></div>")
    return f"<div class='legend'>{legend}</div><div class='bars'>{''.join(rows)}</div>"


def _stat_cards(summary: pd.DataFrame) -> str:
    """The headline read, before any table: who returned most, and at what risk."""
    best_ret = summary["Annual return %"].idxmax()
    best_sharpe = summary["Sortino"].idxmax()
    risky = summary.drop(index=[i for i in summary.index if summary.loc[i, "Return std %"] <= 1e-9],
                         errors="ignore")
    safest = risky["Max drawdown %"].idxmin() if len(risky) else summary.index[0]
    cards = [
        ("Highest return", best_ret, f"{summary.loc[best_ret, 'Annual return %']:.2f}%",
         "annualized, chained across independent years"),
        ("Best risk-adjusted", best_sharpe, f"{summary.loc[best_sharpe, 'Sortino']:.2f}",
         "Sortino, rf = 0"),
        ("Shallowest drawdown", safest, f"{summary.loc[safest, 'Max drawdown %']:.2f}%",
         "worst single year, excluding all-cash"),
    ]
    return "<div class='cards'>" + "".join(
        f"<div class='card'><span class='cardlab'>{lab}</span>"
        f"<span class='cardval'>{val}</span><span class='cardwho'>{who}</span>"
        f"<span class='cardnote'>{note}</span></div>"
        for lab, who, val, note in cards) + "</div>"


CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#fbfbfc; --surface:#f2f3f6; --raised:#ffffff;
  --ink:#14181d; --ink-2:#4a535e; --ink-3:#6f7883;
  --rule:#dfe2e8; --rule-2:#eceef2;
  --accent:#0f6e78; --accent-ink:#0b525a;
  --pos:#12705a; --neg:#a3352c;
  --shadow:0 1px 2px rgba(20,24,29,.06), 0 8px 24px rgba(20,24,29,.05);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0f1216; --surface:#171b21; --raised:#1b2028;
    --ink:#e6e9ee; --ink-2:#a3acb8; --ink-3:#7d8794;
    --rule:#2a313a; --rule-2:#222831;
    --accent:#4fb3bf; --accent-ink:#7fd0d9;
    --pos:#4fb08f; --neg:#d9756a;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --ground:#0f1216; --surface:#171b21; --raised:#1b2028;
  --ink:#e6e9ee; --ink-2:#a3acb8; --ink-3:#7d8794;
  --rule:#2a313a; --rule-2:#222831;
  --accent:#4fb3bf; --accent-ink:#7fd0d9;
  --pos:#4fb08f; --neg:#d9756a;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font:400 16px/1.6 "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  -webkit-font-smoothing:antialiased;
}
.wrap{display:grid; grid-template-columns:minmax(0,1fr); gap:0;
      max-width:1180px; margin:0 auto; padding:3rem 1.25rem 5rem}
@media(min-width:1080px){
  .wrap{grid-template-columns:190px minmax(0,1fr); gap:3rem; padding-top:3.5rem}
}
nav.toc{display:none}
@media(min-width:1080px){
  nav.toc{display:block; position:sticky; top:2rem; align-self:start;
          font-size:13px; line-height:1.5; border-left:2px solid var(--rule); padding-left:1rem}
  nav.toc a{display:block; color:var(--ink-3); text-decoration:none; padding:.28rem 0}
  nav.toc a:hover,nav.toc a:focus-visible{color:var(--accent-ink)}
  nav.toc .tocgrp{font:500 11px/1.4 "IBM Plex Mono",monospace; letter-spacing:.09em;
                  text-transform:uppercase; color:var(--ink); margin:1.1rem 0 .3rem}
  nav.toc .tocgrp:first-child{margin-top:0}
}
h1{font:600 2.05rem/1.15 Spectral, Georgia, serif; margin:0 0 .5rem;
   letter-spacing:-.012em; text-wrap:balance}
h2{font:600 1.32rem/1.25 Spectral, Georgia, serif; margin:3.2rem 0 .6rem;
   padding-bottom:.4rem; border-bottom:1px solid var(--rule); text-wrap:balance}
h3{font:500 .8rem/1.4 "IBM Plex Mono", monospace; letter-spacing:.085em;
   text-transform:uppercase; color:var(--ink-2); margin:2rem 0 .55rem}
p{margin:0 0 .85rem; max-width:70ch; color:var(--ink-2)}
p strong,li strong{color:var(--ink); font-weight:600}
a{color:var(--accent-ink)}
code{font:400 .89em/1 "IBM Plex Mono", monospace;
     background:var(--surface); padding:.12em .34em; border-radius:3px}
.meta{font:400 13px/1.6 "IBM Plex Mono", monospace; color:var(--ink-3);
      margin:0 0 1.6rem; max-width:none}
.meta b{color:var(--ink-2); font-weight:500}
.callout{border-left:3px solid var(--accent); background:var(--surface);
         padding:.85rem 1.1rem; margin:1.4rem 0; border-radius:0 4px 4px 0}
.callout p{margin:0; max-width:66ch; font-size:15px}
.callout p + p{margin-top:.6rem}
dl.conv{margin:1.2rem 0 0; display:grid; gap:1.15rem}
dl.conv dt{font:500 15px/1.4 "IBM Plex Sans",sans-serif; color:var(--ink); margin-bottom:.2rem}
dl.conv dd{margin:0; color:var(--ink-2); max-width:70ch; font-size:15px}
.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
       gap:.85rem; margin:1.5rem 0 .5rem}
.card{background:var(--raised); border:1px solid var(--rule); border-radius:6px;
      padding:.9rem 1rem; box-shadow:var(--shadow); display:flex; flex-direction:column; gap:.15rem}
.cardlab{font:500 10.5px/1.4 "IBM Plex Mono",monospace; letter-spacing:.09em;
         text-transform:uppercase; color:var(--ink-3)}
.cardval{font:500 1.55rem/1.1 "IBM Plex Mono",monospace; color:var(--accent-ink);
         font-variant-numeric:tabular-nums; margin:.15rem 0}
.cardwho{font:500 14px/1.3 "IBM Plex Sans",sans-serif; color:var(--ink)}
.cardnote{font-size:12px; color:var(--ink-3); line-height:1.4}
.scroll{overflow-x:auto; margin:.5rem 0 1.4rem; border:1px solid var(--rule);
        border-radius:6px; background:var(--raised)}
table{border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums}
th,td{padding:.4rem .62rem; text-align:right; white-space:nowrap;
      border-bottom:1px solid var(--rule-2)}
thead th{position:sticky; top:0; background:var(--surface); color:var(--ink);
         font:500 11.5px/1.5 "IBM Plex Mono",monospace; letter-spacing:.04em;
         border-bottom:1px solid var(--rule); z-index:1}
tbody th{text-align:left; font:500 13px/1.5 "IBM Plex Sans",sans-serif; color:var(--ink)}
tbody td{font:400 13px/1.5 "IBM Plex Mono",monospace; color:var(--ink)}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:none}
td.neg{color:var(--neg)}
td.nil{color:var(--ink-3)}
.legend{display:flex; flex-wrap:wrap; gap:.35rem 1rem; margin:.9rem 0 .7rem}
.key{display:inline-flex; align-items:center; gap:.4rem; font-size:12.5px; color:var(--ink-2)}
.key i{width:11px; height:11px; border-radius:2px; display:inline-block; flex:none}
.bars{display:grid; gap:.42rem; margin-bottom:1rem}
.barrow{display:grid; grid-template-columns:150px minmax(0,1fr); align-items:center; gap:.75rem}
.barname{font:500 13px/1.4 "IBM Plex Sans",sans-serif; color:var(--ink);
         overflow:hidden; text-overflow:ellipsis}
.bar{display:flex; height:22px; border-radius:3px; overflow:hidden; background:var(--surface)}
.seg{display:flex; align-items:center; justify-content:center; min-width:0}
.segtxt{font:500 10.5px/1 "IBM Plex Mono",monospace; color:#fff; opacity:.92;
        font-variant-numeric:tabular-nums}
.verdict{font:500 15px/1.55 "IBM Plex Sans",sans-serif; padding:.7rem 1rem;
           border-radius:5px; border:1px solid var(--rule); background:var(--surface);
           margin:1rem 0 .5rem}
 .verdict.ok{border-left:3px solid var(--pos)}
 .verdict.bad{border-left:3px solid var(--neg)}
 .pill{font:500 11px/1 "IBM Plex Mono",monospace; padding:.28rem .5rem;
        border-radius:3px; display:inline-block; letter-spacing:.04em}
 .pill.ok{background:rgba(18,112,90,.16); color:var(--pos)}
 .pill.bad{background:rgba(163,53,44,.16); color:var(--neg)}
 td.txt{font:400 12.5px/1.45 "IBM Plex Sans",sans-serif; text-align:left;
        white-space:normal; max-width:34ch}
 tbody th .sub{font:400 11.5px/1.4 "IBM Plex Sans",sans-serif; color:var(--ink-3);
               display:block; max-width:38ch; white-space:normal}
.note{font-size:13.5px; color:var(--ink-3); margin:-.15rem 0 .9rem; max-width:72ch}
pre{background:var(--surface); border:1px solid var(--rule); border-radius:6px;
    padding:.8rem 1rem; overflow-x:auto; font:400 12.5px/1.6 "IBM Plex Mono",monospace;
    color:var(--ink)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (prefers-reduced-motion: reduce){*{animation:none!important; transition:none!important}}
</style>
"""


def _acceptance_html(acceptance) -> str:
    """Explicit pass/fail, with the blocking and non-blocking groups kept apart.

    A hard-engineering failure fails the model outright; a performance shortfall is a
    finding reported as measured. Collapsing the two into one verdict is how a broken
    safety layer gets excused as bad luck -- and, in the other direction, how a merely
    unimpressive result gets called broken.
    """
    if acceptance is None:
        return ""
    groups = {"hard_engineering": "Hard engineering &mdash; all must be exactly zero",
              "risk_reporting": "Risk reporting &mdash; all must be present",
              "performance": "Performance &mdash; reported as measured, non-blocking"}
    d = acceptance.to_dict()
    verdict = "PASS" if acceptance.passed else "FAIL"
    cls = "ok" if acceptance.passed else "bad"
    out = ["<h2 id='acceptance'>Acceptance</h2>",
           f"<p class='verdict {cls}'>Blocking criteria: <b>{verdict}</b> "
           f"&nbsp;&middot;&nbsp; {d['n_passed']} of {d['n_criteria']} criteria pass"
           "</p>"]
    for key, title in groups.items():
        rows = acceptance.group(key)
        if not rows:
            continue
        out.append(f"<h3>{title}</h3><div class='scroll'><table><thead><tr>"
                   "<th scope='col'>Criterion</th><th scope='col'>Observed</th>"
                   "<th scope='col'>Requirement</th><th scope='col'>Result</th>"
                   "</tr></thead><tbody>")
        for c in rows:
            mark = ("<span class='pill ok'>PASS</span>" if c.passed
                    else "<span class='pill bad'>FAIL</span>")
            note = f"<br><span class='sub'>{c.note}</span>" if c.note else ""
            out.append(f"<tr><th scope='row'>{c.name}{note}</th>"
                       f"<td class='txt'>{c.observed}</td>"
                       f"<td class='txt'>{c.requirement}</td><td>{mark}</td></tr>")
        out.append("</tbody></table></div>")
    return "".join(out)


def render_page(summary: pd.DataFrame, tables: dict, meta: dict,
                compliance: pd.DataFrame, acceptance=None) -> str:
    """The whole report as one self-contained fragment."""
    perf = summary[["Annual return %", "Return std %", "Sharpe", "Sortino", "Max drawdown %"]]
    dmax_pct = f"{meta['max_drawdown']:.0%}"

    year_tables = [
        ("annual-return", "Annual return", "%", tables["annual_return"], 2, True, False,
         None),
        ("volatility", "Return standard deviation", "% annualized",
         tables["volatility"], 2, False, False, None),
        ("sortino", "Sortino ratio", "rf = 0, downside deviation",
         tables["sortino"], 2, True, False,
         "The headline risk-adjusted measure. Sharpe penalises upside deviation as hard "
         "as downside; for a system built around a drawdown constraint, what matters is "
         "the dispersion of losses."),
        ("sharpe", "Sharpe ratio", "rf = 0", tables["sharpe"], 2, True, False,
         "Kept for reference, not for judging."),
        ("max-drawdown", "Maximum drawdown", "% within year", tables["max_drawdown"],
         2, False, True, "The peak resets on the first session of each year."),
    ]

    toc = ["<div class='tocgrp'>Report</div>",
           "<a href='#how'>How to read this</a>",
           "<a href='#summary'>Summary</a>",
           "<a href='#acceptance'>Acceptance</a>",
           "<a href='#compliance'>Constraint compliance</a>",
           "<div class='tocgrp'>Year by year</div>"]
    toc += [f"<a href='#{i}'>{t}</a>" for i, t, _, _, _, _, _, _ in year_tables]
    toc.append("<div class='tocgrp'>Allocation</div>")
    toc += [f"<a href='#alloc-{i}'>{c}</a>"
            for i, c in enumerate(CATEGORIES)]

    body: list[str] = []
    add = body.append

    add(f"<h1>{meta['title']}</h1>")
    add(f"<p class='meta'>Out of sample <b>{meta['first_year']}–{meta['last_year']}</b> "
        f"&nbsp;·&nbsp; N = <b>{meta['hold_days']}</b> calendar days &nbsp;·&nbsp; "
        f"D<sub>max</sub> = <b>{dmax_pct}</b> &nbsp;·&nbsp; "
        f"{meta['n_strategies']} strategies × {meta['last_year'] - meta['first_year'] + 1} "
        f"years &nbsp;·&nbsp; {meta['generated_at']}</p>")

    add("<div class='callout'><p><strong>This is the reference format.</strong> "
        "When the RL policy exists it becomes one more column in every table below, and "
        "nothing else changes. Fixing the format before training is deliberate: a report "
        "format settled after seeing the results is a report format chosen to flatter "
        "them.</p></div>")

    add(_stat_cards(summary))
    add(_acceptance_html(acceptance))

    add("<h2 id='how'>How to read this</h2>")
    add("<dl class='conv'>")
    add("<dt>Every year is an independent evaluation window.</dt><dd>Fresh capital each "
        "January, no inherited positions, no inherited locks, and the drawdown peak reset "
        "to the opening NAV. This is forced rather than stylistic: on a continuous "
        "multi-year run the ceiling is measured against a peak that never resets, and "
        "<code>spy_buy_hold</code> breached "
        f"D<sub>max</sub> = {dmax_pct} in 2009, went to 100% cash and stayed there for "
        "fifteen years — every annual cell from 2009 on would read exactly 0.00%. "
        "Independent years also match how walk-forward evaluates the policy, one trained "
        "model per test year, which is what makes the RL column comparable to these.</dd>")
    add("<dt>Sortino is the headline measure, not Sharpe.</dt><dd>Sharpe divides by "
        "total volatility, which penalises upside deviation exactly as hard as downside "
        "&mdash; a strategy is marked down for having good months. For a system whose "
        "entire purpose is a <em>drawdown</em> constraint, what matters is the dispersion "
        "of losses. The denominator is the standard downside deviation, "
        "<code>sqrt(mean(min(r, 0)^2))</code> annualised over <em>every</em> observation "
        "rather than only the losing ones &mdash; averaging over just the losers flatters "
        "a strategy that loses rarely. Sharpe is still reported, for reference.</dd>")
    add("<dt>The risk-free rate is zero, so Sharpe is return ÷ volatility.</dt><dd>CASH "
        "in this universe returns exactly 0.00% per day and is the agent's outside "
        "option, so excess return over the risk-free asset <em>is</em> the raw return. "
        "Substituting a T-bill series would make CASH a negative-carry asset the "
        "simulator does not model.</dd>")
    add("<dt>Allocation is the time average of daily weights.</dt><dd>In percentage "
        "points, summing to 100 by construction. Not a year-end snapshot — a snapshot "
        "cannot tell a portfolio that held 60% equity all year from one that held 0% for "
        "eleven months and 60% in December.</dd>")
    add("<dt>The baselines run through the full constraint layer.</dt><dd>The same "
        f"per-ETF {meta['hold_days']}-day holding lock, the same {dmax_pct} drawdown "
        "ceiling and the same risk envelope the agent faces. <code>spy_buy_hold</code> "
        "here is <em>not</em> unconstrained SPY; it is SPY as this system would have been "
        "allowed to hold it. That is what makes it a fair reference point, and it is why "
        "the returns sit below the index.</dd>")
    add("</dl>")

    add("<h2 id='summary'>Summary — whole window</h2>")
    add("<p>Return chains the independent years: what a caller redeploying each January "
        "would have compounded. Volatility pools every daily return. <strong>Max drawdown "
        "is the worst annual drawdown</strong>, not a drawdown across the window — a "
        "cross-window figure would be measured against a peak no strategy operated "
        "under.</p>")
    add(_num_table(perf, 2, "Strategy", signed=True))
    add("<h3>Average allocation · percentage points</h3>")
    add(_alloc_bars(summary))
    alloc = summary[list(CATEGORIES)].copy()
    alloc["Total"] = alloc.sum(axis=1)
    add(_num_table(alloc, 1, "Strategy", signed=False))

    add("<h2 id='compliance'>Constraint compliance</h2>")
    add("<p><code>Lock violations</code> and <code>feasibility violations</code> are "
        "<strong>hard acceptance criteria</strong>: a result failing either is not a "
        "weaker result, it is an invalid one, and the report refuses to build if any cell "
        "shows one.</p>")
    add(_num_table(compliance, 2, "Strategy", signed=False))
    add(f"<p class='note'><strong>“Years D<sub>t</sub> &gt; D<sub>max</sub>” is a count, "
        f"not a verdict.</strong> D<sub>max</sub> = {dmax_pct} constrains what the agent "
        "may <em>do</em>; it is never a guarantee about the realized path. A gap that "
        "opens overnight moves the NAV with no action available to prevent it, and the "
        "lock can hold a falling position for weeks. What would be a defect is a "
        "<em>preventable</em> violation — an action that should have been blocked and was "
        "not. Separating the two is the drawdown violation taxonomy that Stage 8 "
        "implements and T13 proves fires; until then this column is deliberately "
        "unadjudicated.</p>")

    add("<h2>Year by year</h2>")
    for anchor, title, unit, frame, dec, signed, invert, note in year_tables:
        add(f"<h3 id='{anchor}'>{title} · {unit}</h3>")
        if note:
            add(f"<p class='note'>{note}</p>")
        add(_num_table(frame, dec, "Year", signed=signed, invert=invert))

    add("<h2>Allocation by category</h2>")
    add("<p>Percentage points, time-averaged within each year. For any (year, strategy) "
        "the seven tables below sum to 100.</p>")
    for i, category in enumerate(CATEGORIES):
        swatch = (f"<i style='display:inline-block;width:10px;height:10px;border-radius:2px;"
                  f"background:{CATEGORY_COLORS[category]};margin-right:.45rem'></i>")
        add(f"<h3 id='alloc-{i}'>{swatch}{category}</h3>")
        add(_num_table(tables[f"allocation::{category}"], 1, "Year", signed=False))

    add("<h2>Reproducing this</h2>")
    add(f"<pre>python -m scripts.s12_report --config {meta['config']} \\\n"
        f"    --first-year {meta['first_year']} --last-year {meta['last_year']}</pre>")
    add(f"<p class='meta'>Run <code>{meta['run_id']}</code> &nbsp;·&nbsp; "
        f"git <code>{meta['git_sha']}</code> &nbsp;·&nbsp; seed {meta['seed']} "
        f"&nbsp;·&nbsp; every figure recomputed from "
        f"<code>trajectory.parquet</code> by <code>src/evaluation/report.py</code></p>")

    return (f"<title>{meta['title']}</title>\n{CSS}\n"
            f"<div class='wrap'><nav class='toc'>{''.join(toc)}</nav>"
            f"<main>{''.join(body)}</main></div>")
