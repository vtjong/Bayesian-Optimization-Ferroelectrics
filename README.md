# Mapping the flash-anneal crystallization boundary in HZO

Active-learning campaign to locate the amorphous → crystalline boundary of flash-lamp-annealed
Hf₀.₅Zr₀.₅O₂ thin films in the two-knob process space (flash voltage `V`, pulse time `t`), using a
Gaussian process with a level-set acquisition to spend as few fabricated samples as possible.

The target is **total crystalline fraction**, not phase. The map is therefore monotone in thermal
severity and there is a single boundary to find.

## The question the design is built around

Peak temperature `Tmax(V, t)` is measured (`data/flash_temp_table.csv`). The temperature **trace**
`T(τ)` is not — it is asserted by a pulse shape. Activated kinetics integrate the trace, so that
assertion decides the boundary's geometry:

```
Phi(V, t) = INT exp(-Ea / kB T(tau)) dtau  =  t_eff * exp(-Ea / kB Tmax)
Tb(t2) - Tb(t1) = -theta * ln( t_eff(t2)/t_eff(t1) ),      theta = kB Tmax^2 / Ea ~ 37 K
```

If the effective dwell ignores the commanded pulse width, the tilt is exactly zero and the boundary
is a plain `Tmax` level set. If it tracks the pulse width, it is not. Four candidate cooling laws
span the range, over `t` in `[2.6, 10.1] ms`:

| shape | tail | `t_eff` | tilt |
|---|---|---|---|
| `isoT` | fitted exponential + permanent plateau | independent of `t` | **0.0 °C** |
| `ramp` | empirical two-exponential + plateau | `∝ t + 12.4` | **13.3 °C** |
| `diffusion` | **derived**, `τ^(-1/2)`, no plateau | `∝ 2t` | **49.8 °C** |
| `rect` | bounding case | `= t` | **50.0 °C** |

`diffusion` is the default: a 30 nm stack has negligible heat capacity, so its temperature is
slaved to conduction into the fused-silica substrate, and the Carslaw & Jaeger surface solution has
zero fitted parameters. It depends on `τ` only through `τ/t`, so conduction — having no intrinsic
timescale — makes the effective dwell exactly proportional to the pulse width.

**Nothing measured yet distinguishes these.** One digitized `T(τ)` at two pulse widths would, and
costs no film. The seed design is built to be informative regardless of which is true.

## Repository layout

```
src/discovery/
  constants.py    every number, grouped by the kind of claim it makes
                  (physical / prior / shape / readout / numerical)
  synthetic.py    measured Tmax table + the four pulse shapes + the thermal model
  kinetics.py     the four boundary models, pinned to one onset anchor
  picker.py       the active learner: GP, LSE/BALD/entropy acquisitions, batching
src/
  run_flash_plan.py     the seed design  -> data/flash_plan_seed.csv
  run_seed_power.py     can the seed identify the tilt?
  run_thermal_check.py  what the thermal model can and cannot be checked against
  run_calibration.py    onset and readout calibration against archival data
  run_picker_demo.py    acquisition comparison
  run_batch_demo.py     batch-size study
  run_seed_budget.py    seed/active split study
tests/            pytest suite over the models and the seed design
scripts/          shell wrappers; each emits PNGs under predictions/
```

## Quick start

```bash
./setup_venv.sh && source venv/bin/activate
./scripts/run_tests.sh          # 51 checks over the models and the design
./scripts/run_flash_plan.sh     # regenerate the seed plan
./scripts/run_seed_power.sh     # its acceptance test
```

## The seed design

14 shots of an 80-shot budget. Rather than a uniform Latin hypercube over `(V, t)` — which is not
uniform in the quantity that matters, since `Tmax` spans 82–563 °C while the transition is tens of
degrees wide — the generator stratifies the **measured** quantity: LHS over `(Tmax, log t)`,
inverted through the table to `(V, t)`.

| block | n | purpose |
|---|---|---|
| A | 4 | iso-`Tmax` ladder: one peak temperature, four pulse widths. The tilt test. |
| B | 7 | stratified core over `Tmax` ∈ [310, 490] °C, maximin against A. The GP's seed. |
| E | 1 | amorphous floor anchor. |
| D | +2 | replicates on separate specimens — the go/no-go on variables outside `(V, t)`. |

Boundary-search conditions are restricted to `t ≥ 2.6 ms`: the table has no node below that, across
which `Tmax` rises 374 °C *and* the surface maximum lies, so any `Tmax` quoted there is a spline
artifact. Every condition is scored against all four hypotheses and the CSV records the spread
rather than a single predicted label.

Acceptance test (`run_seed_power.py`, calibrated heteroscedastic readout noise):

```
power to DETECT a tilt when one exists      : 98.7 %
power to CONFIRM no tilt when there is none : 95.8 %
```

`diffusion` and `rect` separate only 50/50 — expected, they are ~1 °C apart and imply the same
experimental conclusion.

## Method

- **Surrogate** — GP on normalized `(V, t)` with Matérn 5/2 ARD and **heteroscedastic** per-point
  noise `σ_n(f) = 0.02 + 0.30 f(1−f)`, peaking at ≈0.095 mid-transition where the film is a phase
  mixture. Squared-exponential paths are analytic and ring across a steep ramp; Matérn 5/2 does not.
- **Acquisition** — level-set entropy on the latent (reducible) variance,
  `a(x) = H(Φ((μ − θ)/s))`, so it targets where new data reduces boundary uncertainty rather than
  where noise is irreducibly high. Predictive-entropy and BALD variants are retained as baselines;
  on current evidence the three are statistically comparable.
- **Batching** — Kriging-believer fantasizing spreads a batch along the contour.
- **Readout** — continuous crystalline fraction; in-loop proxy is permittivity `ε_r`.

## Standing caveats

- **The onset is a prior, not a measurement.** `T_ONSET_C = 380 °C ± 30` comes from logistic fits of
  `2P_r` vs temperature on two datasets that disagree (flash 388 °C, RTA 357 °C), fitted in a
  different voltage box and in different units, using a ferroelectric switch-on indicator rather
  than crystallinity. Reconciling them is open work; until then no result may be stated as "the
  measured onset".
- **`ε_r` is calibrated against `2P_r`,** which is o-phase specific, while the target counts any
  crystalline phase. These agree only if essentially all crystallized material in the box is
  o-phase. One structural check on the highest-`Tmax` sample audits it.
- **The cooling law is unvalidated** (above). The ensemble exists precisely so no result silently
  depends on picking one.
