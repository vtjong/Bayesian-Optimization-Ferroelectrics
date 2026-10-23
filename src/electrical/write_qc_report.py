"""Write the QC page: every capacitor of every sample and state overlaid, outliers flagged.

The page is where a sample mean earns trust. The operator looks at the overlaid traces and the
flagged rows before reading the per-sample table. Every value from the data is escaped.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from electrical.extract_pund_features import SWITCHING_PSW_UC_CM2  # noqa: E402
from electrical.flag_issues import OUTLIER_COLUMNS, OUTLIER_ROBUST_Z  # noqa: E402
from electrical.process_round import RoundResult  # noqa: E402

#: Columns of the per-capacitor table shown on the page, in order.
SHOWN_COLUMNS = (
    "cap",
    "switching",
    "psw_uc_cm2",
    "psw_inst_uc_cm2",
    "pr_uc_cm2",
    "pr_gap_uc_cm2",
    "kappa_summary",
    "kappa_pund",
    "j_leak_a_cm2",
    "peak_pos_coercive_v",
    "peak_neg_coercive_v",
    "split_raw_n_dip_fraction_main",
    "split_raw_n_dip_found",
    "pinching_index",
    "endurance_wake_ratio",
    "pulse_pattern",
    "issues",
    "outlier_flags",
)
#: Per-sample statistics shown as mean ± sd, with their headings.
SUMMARY_FEATURES = (
    ("psw_uc_cm2", "Psw"),
    ("psw_inst_uc_cm2", "Psw (instrument)"),
    ("pr_uc_cm2", "Pr"),
    ("kappa_summary", "κ (Q–V)"),
    ("kappa_pund", "κ (PUND)"),
    ("split_raw_n_dip_fraction_main", "Q₁ fraction"),
    ("vc_mean_v", "Vc mean"),
    ("endurance_wake_ratio", "wake-up"),
)
STYLE = (
    "body{font-family:system-ui,sans-serif;margin:24px;color:#222}"
    "table{border-collapse:collapse;font-size:12px}"
    "td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}th{background:#eee}"
    "td.l{text-align:left}.flag{background:#fde2e2}.sw{background:#e3f4e3}h2{margin-top:36px}"
)


def write_qc_report(path: Path, result: RoundResult) -> Path:
    """Write the round's QC page to ``path``, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_qc_page(result), encoding="utf-8")
    return path


def build_qc_page(result: RoundResult) -> str:
    """Return the QC page for one round as HTML."""
    capacitors = result.capacitors
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>HZO electrical QC</title>",
        f"<style>{STYLE}</style></head><body><h1>HZO electrical feature QC</h1>",
        f"<p>{len(capacitors)} capacitor-states, {capacitors['sample'].nunique()} samples. "
        f"Outlier flag: robust z &gt; {OUTLIER_ROBUST_Z} within sample and state on "
        + ", ".join(OUTLIER_COLUMNS)
        + f". Switching: whole-pulse Psw &gt; {SWITCHING_PSW_UC_CM2:g} µC/cm².</p>",
    ]

    problems = result.raw_index[result.raw_index["problems"].fillna("").astype(str) != ""]
    if len(problems) or result.issues:
        rows = [
            ("", [r["sample"], r["cap"], r["kind"], r["problems"]]) for _, r in problems.iterrows()
        ]
        rows += [
            ("flag", [i.where.get("sample_key"), i.where.get("capacitor"), "", i.describe()])
            for i in result.issues
        ]
        parts.append(
            "<h2>Issues while reading</h2>" + _table(["sample", "cap", "kind", "issue"], rows)
        )

    headings = [
        "sample",
        "state",
        "n",
        "n switching",
        *(h for _, h in SUMMARY_FEATURES),
        "inhomogeneous",
    ]
    rows = []
    for _, r in result.samples.iterrows():
        values = [r["sample"], r["state"], int(r["n_capacitors"]), int(r["n_switching"])]
        values += [_mean_sd(r, feature) for feature, _ in SUMMARY_FEATURES]
        values.append(bool(r.get("inhomogeneous", False)))
        rows.append(("", values))
    parts.append("<h2>Per-sample summary</h2>" + _table(headings, rows))

    shown = [c for c in SHOWN_COLUMNS if c in capacitors.columns]
    for (rnd, sample, state), group in capacitors.groupby(["round", "sample", "state"], sort=False):
        parts.append(
            f"<h2>{html.escape(str(sample))} — {html.escape(str(state))} (round {int(rnd)})</h2>"
        )
        traces = result.traces.get((sample, state))
        if traces:
            figure = _draw_sample_figure(f"{sample} {state}", traces)
            parts.append(f"<img src='{figure}' style='max-width:100%'>")
        rows = [
            (
                "flag" if r.get("outlier_flags") else ("sw" if r.get("switching") else ""),
                [r[c] for c in shown],
            )
            for _, r in group.iterrows()
        ]
        parts.append(_table(shown, rows))
    parts.append("</body></html>")
    return "\n".join(parts)


def _draw_sample_figure(title: str, traces: dict) -> str:
    """Overlay every capacitor's PUND current, switching current, Q-V loop and endurance curve.

    :return: The figure as a PNG data URI.
    """
    has_end = any("end" in tr for tr in traces.values())
    fig, ax = plt.subplots(1, 4 if has_end else 3, figsize=(21 if has_end else 16, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, (cap, tr) in enumerate(sorted(traces.items())):
        colour = cmap(i % 10)
        pund = tr.get("pund")
        if pund is not None:
            ax[0].plot(pund.time_s * 1e6, pund.current_a * 1e6, lw=0.6, color=colour, label=cap)
            if pund.isw_neg_a is not None:
                ax[1].plot(pund.t_rel_neg_s * 1e6, -pund.isw_neg_a * 1e6, lw=0.8, color=colour)
                ax[1].plot(
                    pund.t_rel_pos_s * 1e6,
                    pund.isw_pos_a * 1e6,
                    lw=0.8,
                    color=colour,
                    ls=":",
                    alpha=0.7,
                )
        pv = tr.get("pv")
        if pv is not None:
            ax[2].plot(
                pv.voltage_v, pv.charge_c / pv.area_cm2 * 1e6, lw=0.8, color=colour, label=cap
            )
        if has_end and "end" in tr:
            curve = tr["end"]
            ax[3].semilogx(
                curve["cycles"].clip(lower=0.5),
                curve["psw_uc_cm2"],
                marker="o",
                ms=3,
                lw=0.9,
                color=colour,
            )
    ax[0].set(title=f"{title}: PUND current", xlabel="t (µs)", ylabel="I (µA)")
    ax[1].set(
        title="switching current  N−D (solid), P−U (dotted)",
        xlabel="t from pulse start (µs)",
        ylabel="|I_sw| (µA)",
    )
    ax[2].set(title="Q–V loops", xlabel="V (V)", ylabel="P (µC/cm²)")
    ax[2].legend(fontsize=6, ncol=2, loc="upper left")
    if has_end:
        ax[3].set(
            title="endurance: Psw vs cycles",
            xlabel="cycles (0 plotted at 0.5)",
            ylabel="Psw (µC/cm²)",
        )
    for panel in ax:
        panel.grid(alpha=0.3)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _cell(value) -> str:
    """One escaped ``<td>``: numbers to four significant figures, NaN as a dash, True as yes."""
    if isinstance(value, (bool, np.bool_)):
        return f"<td class='l'>{'yes' if value else ''}</td>"
    if isinstance(value, (float, np.floating)):
        return f"<td>{value:.4g}</td>" if np.isfinite(value) else "<td>–</td>"
    if isinstance(value, (int, np.integer)):
        return f"<td>{value}</td>"
    text = "" if value is None or value != value else str(value)
    return f"<td class='l'>{html.escape(text)}</td>"


def _table(headings, rows) -> str:
    """An HTML table from headings and ``(css class, cell values)`` rows."""
    head = "<tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headings) + "</tr>"
    body = "".join(
        f"<tr class='{css}'>" + "".join(_cell(v) for v in values) + "</tr>" for css, values in rows
    )
    return f"<table>{head}{body}</table>"


def _mean_sd(row, feature: str) -> str:
    mean, sd = row.get(f"{feature}_mean", np.nan), row.get(f"{feature}_sd", np.nan)
    return f"{mean:.3f} ± {sd:.3f}" if np.isfinite(mean) else "–"
