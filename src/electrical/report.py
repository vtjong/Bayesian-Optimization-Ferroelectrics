"""QC report: every capacitor of every sample overlaid, outliers flagged, nothing averaged away."""

from __future__ import annotations

import base64
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUTLIER_COLS = ["Pr", "kappa_summary", "Psw", "kappa_PUND", "J_leak_A_cm2"]


def flag_outliers(cap: pd.DataFrame, k: float = 3.5) -> pd.DataFrame:
    """Robust z = |x - median| / (1.4826 MAD) > k within each sample, per feature."""
    cap = cap.copy()
    cap["outlier_flags"] = ""
    for (rnd, s, st), g in cap.groupby(["round", "sample", "state"]):
        for c in OUTLIER_COLS:
            if c not in g:
                continue
            x = pd.to_numeric(g[c], errors="coerce")
            med = x.median()
            mad = (x - med).abs().median() * 1.4826
            if not np.isfinite(mad) or mad == 0:
                continue
            z = (x - med).abs() / mad
            for idx in z[z > k].index:
                cap.loc[idx, "outlier_flags"] += f"{c}(z={z[idx]:.1f});"
    return cap


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def sample_figure(sample: str, traces: dict) -> str:
    """traces: {cap: {"pund": {...}, "pv": {...}, "end": DataFrame}}"""
    has_end = any("end" in tr for tr in traces.values())
    fig, ax = plt.subplots(1, 4 if has_end else 3, figsize=(21 if has_end else 16, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, (cap, tr) in enumerate(sorted(traces.items())):
        if has_end and "end" in tr:
            e = tr["end"]
            ax[3].semilogx(
                e["iteration"].clip(lower=0.5),
                e["Psw"],
                marker="o",
                ms=3,
                lw=0.9,
                color=cmap(i % 10),
                label=cap,
            )
        c = cmap(i % 10)
        p = tr.get("pund")
        if p is not None and "t" in p:
            ax[0].plot(p["t"] * 1e6, p["I"] * 1e6, lw=0.6, color=c, label=cap)
            if "Isw_neg" in p:
                ax[1].plot(
                    p["t_rel_neg"] * 1e6, -p["Isw_neg"] * 1e6, lw=0.8, color=c, label=f"{cap} N-D"
                )
                ax[1].plot(
                    p["t_rel_pos"] * 1e6, p["Isw_pos"] * 1e6, lw=0.8, color=c, ls=":", alpha=0.7
                )
        v = tr.get("pv")
        if v is not None and "V" in v:
            A = v.get("area_cm2", 9e-6)
            ax[2].plot(v["V"], v["Q"] / A * 1e6, lw=0.8, color=c, label=cap)
    ax[0].set(title=f"{sample}: PUND current (all capacitors)", xlabel="t (µs)", ylabel="I (µA)")
    ax[1].set(
        title="switching current  N−D (solid), P−U (dotted)",
        xlabel="t from pulse start (µs)",
        ylabel="|I_sw| (µA)",
    )
    ax[2].set(title="Q–V loops", xlabel="V (V)", ylabel="P (µC/cm²)")
    if has_end:
        ax[3].set(
            title="endurance: Psw (P−U, N−D mean) vs cycles",
            xlabel="cycles (0 plotted at 0.5)",
            ylabel="Psw (µC/cm²)",
        )
    ax[2].legend(fontsize=6, ncol=2, loc="upper left")
    for a in ax:
        a.grid(alpha=0.3)
    return _fig_to_b64(fig)


def write_report(
    path: str, cap: pd.DataFrame, per_sample: pd.DataFrame, traces: dict, raw_index: pd.DataFrame
):
    cap = flag_outliers(cap)
    show = [
        "cap",
        "switching",
        "Psw",
        "Psw_inst",
        "Pr",
        "Pr_summary",
        "kappa_summary",
        "kappa_PUND",
        "J_leak_A_cm2",
        "Vc_pos",
        "Vc_neg",
        "QO_frac_rawN",
        "split_rule_rawN",
        "pinching_index",
        "wake_ratio",
        "pulse_pattern",
        "problem",
        "outlier_flags",
    ]
    show = [c for c in show if c in cap.columns]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>HZO electrical QC</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#222}table{border-collapse:collapse;font-size:12px}"
        "td,th{border:1px solid #ccc;padding:3px 6px;text-align:right}th{background:#eee}"
        "td.l{text-align:left}"
        ".flag{background:#fde2e2}.sw{background:#e3f4e3}h2{margin-top:36px}</style></head><body>",
        "<h1>HZO electrical feature QC</h1>",
        f"<p>{len(cap)} capacitors, {cap['sample'].nunique()} samples. "
        "Outlier flag: robust z &gt; 3.5 within sample on "
        + ", ".join(OUTLIER_COLS)
        + ". Switching flag: Psw &gt; 2 µC/cm².</p>",
    ]
    prob = raw_index[raw_index["problems"].astype(str).str.len() > 0]
    if len(prob):
        parts.append(
            "<h2>Ingest problems</h2><table>"
            "<tr><th>sample</th><th>cap</th><th>kind</th><th>problems</th></tr>"
        )
        for _, r in prob.iterrows():
            parts.append(
                f"<tr><td class='l'>{r['sample']}</td><td>{r['cap']}</td><td>{r['kind']}</td>"
                f"<td class='l'>{html.escape(str(r['problems']))}</td></tr>"
            )
        parts.append("</table>")
    parts.append(
        "<h2>Per-sample summary</h2><table><tr>"
        + "".join(
            f"<th>{c}</th>"
            for c in [
                "sample",
                "state",
                "n",
                "n_sw",
                "Psw",
                "Psw_inst",
                "Pr",
                "κ_summary",
                "κ_PUND",
                "QO_frac_rawN",
                "Vc_mean",
                "wake_ratio",
                "inhom.",
            ]
        )
        + "</tr>"
    )
    for _, r in per_sample.iterrows():

        def ms(c):
            m, s = r.get(f"{c}_mean", np.nan), r.get(f"{c}_sd", np.nan)
            return f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "–"

        parts.append(
            f"<tr><td class='l'>{r['sample']}</td><td class='l'>{r['state']}</td>"
            f"<td>{r['n_capacitors']}</td><td>{r['n_switching']}</td>"
            f"<td>{ms('Psw')}</td><td>{ms('Psw_inst')}</td><td>{ms('Pr')}</td><td>{ms('kappa_summary')}</td>"
            f"<td>{ms('kappa_PUND')}</td><td>{ms('QO_frac_rawN')}</td><td>{ms('Vc_mean')}</td>"
            f"<td>{ms('wake_ratio')}</td><td>{'yes' if r.get('inhomogeneous') else ''}</td></tr>"
        )
    parts.append("</table>")
    for (rnd, sample, state), g in cap.groupby(["round", "sample", "state"], sort=False):
        parts.append(f"<h2>{sample} — {state} (round {rnd})</h2>")
        if (sample, state) in traces:
            parts.append(
                "<img src='data:image/png;base64,"
                f"{sample_figure(f'{sample} {state}', traces[(sample, state)])}'"
                " style='max-width:100%'>"
            )
        parts.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in show) + "</tr>")
        for _, r in g.iterrows():
            cls = "flag" if r.get("outlier_flags") else ("sw" if r.get("switching") else "")
            cells = []
            for c in show:
                v = r[c]
                if isinstance(v, (float, np.floating)):
                    cells.append(f"<td>{v:.4g}</td>" if np.isfinite(v) else "<td>–</td>")
                else:
                    cells.append(
                        "<td class='l'>"
                        f"{html.escape(str(v)) if v is not None and v == v else ''}</td>"
                    )
            parts.append(f"<tr class='{cls}'>" + "".join(cells) + "</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return cap
