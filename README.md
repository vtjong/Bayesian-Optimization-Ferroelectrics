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
costs no film. The initial design is built to be informative regardless of which is true.

## Repository layout

```
src/physics/          constants.py, thermal_model.py: the measured Tmax table and the cooling laws
src/active_learning/  surrogate.py, acquisition.py: the GP and the level-set acquisition
src/campaign/         plan.py, results.py, reporting.py: the fabrication plan and the sheet the lab fills
src/design_space.py   the settable (V, t) box
src/paths.py          where the measured data lives
src/run_*.py          entry points; scripts/ holds their shell wrappers
data/                 measured inputs and campaign records
tests/                pytest suite
```

## Quick start

```bash
./setup_venv.sh && source venv/bin/activate
./scripts/run_tests.sh              # the suite
./scripts/run_initial_design.sh     # the initial design plan and the sheet the lab fills in
python src/run_campaign.py          # read results, fit, propose the next batch
```

## The initial design

The first batch is a set of conditions chosen by the principal investigator, recorded in
`data/flash_plan_initial.csv` with the reason for each condition and the simulated peak-temperature
band attached for reading only. `run_initial_design.py` writes that plan and the blank results sheet
once; `run_initial_design_results.py` validates the filled sheet (every row carries an explicit
status, a blank is never read as zero) and reports coverage, as-fired drift and replicate contrast.
From then on `run_campaign.py` reads the measured results, fits the surrogate and proposes the next
batch in the same format.

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
