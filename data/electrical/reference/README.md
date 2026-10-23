# Reference summaries from the group

| File | Source | Content |
| --- | --- | --- |
| `summary_rounds1-2.csv` | `Summary4John-2.xlsx`, sheets Round1 and Round2 | Per-sample flash settings and the group's pristine Pr, kappa and Psw (mean, sd over capacitors). The regression target for the electrical pipeline. |
| `summary_round1_extended.csv` | `ML_Summary.xlsx`, sheet Round1 | Same samples with Ec+ / Ec- and the post-10^6-cycle (`State1e6_*`) columns added. |
| `process_rounds1-2.csv` | the two sheets above | Flash voltage, pulse width, measured energy density, simulated Tmax and time above 450 C per sample; the process covariates for the correlation study. Rounds 3-6 are not yet available. |

`sample_key` is the normalised join key (`Sample 11`, `Sample2_square`, `Sample4_again` ->
`sample11`, `sample2`, `sample4`). The group's definitions of the three summary columns are
documented in `src/electrical/`.
