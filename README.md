# Bayesian Optimization of Pulse Annealing Parameters in Hafnium Zirconium Oxide Ferroelectric Thin Films

**Supporting Code Repository for IEEE Publication**

TODO: We should determine a publication name...

---

## Abstract

We present a Bayesian optimization framework employing Gaussian process regression for the autonomous optimization of laser pulse annealing parameters in Hf₀.₅Zr₀.₅O₂ (HZO) ferroelectric thin films. The optimization objective maximizes the ferroelectric figure of merit $2Q_{sw}/(U+|D|)$ at $10^6$ switching cycles, balancing switching charge ($Q_{sw}$), remanent polarization ($U$), and coercive field asymmetry ($D$). Using Matérn kernel Gaussian processes with automatic relevance determination (ARD), we achieve sample-efficient exploration of the two-dimensional processing space spanning pulse duration (0.5–5.0 ms) and energy density (2.7–15.4 J/cm²). The implementation utilizes Monte Carlo acquisition functions (qUCB with $\beta=5.0$) to suggest batch experiments, enabling parallel experimental validation. This methodology demonstrates the viability of machine learning-guided processing optimization for complex ferroelectric systems with expensive characterization costs.

---

## I. Introduction

Ferroelectric HZO thin films exhibit promise for neuromorphic computing and non-volatile memory applications, but their functional properties critically depend on thermal processing conditions during crystallization. Traditional parameter space exploration via design-of-experiments or grid search scales poorly with dimensionality and requires extensive experimental resources. We address this challenge through Bayesian optimization (BO), a sequential decision-making framework that constructs probabilistic surrogate models to guide experimental exploration.

### A. Problem Formulation

**Objective Function:**

```math
\text{FOM}(\mathbf{x}) = \frac{2Q_{sw}(\mathbf{x})}{U(\mathbf{x}) + |D(\mathbf{x})|} \quad \text{at } N = 10^6 \text{ cycles}
```

where $\mathbf{x} = [t_{\text{pulse}}, E_{\text{density}}]^\top$ represents the processing parameters.

**Search Space:**
- Pulse duration: $t_{\text{pulse}} \in [0.5, 5.0]$ ms (thermal diffusion timescale)
- Energy density: $E_{\text{density}} \in [2.7, 15.4]$ J/cm² (peak temperature control)
- **Note:** Search bounds are data-driven, defined by the range of initial exploratory experiments. The acquisition candidate grid extends ~4% beyond observed bounds (one grid spacing $\Delta x = (\text{max} - \text{min})/(n_{\text{grid}} - 2)$ on each side) to allow conservative extrapolation where the GP posterior remains reliable, as the optimum may lie just outside the sampled region (see `src/optimization/grid.py`).

**Constraints:**
- Sample throughput: ~2–4 experiments per batch
- Experimental uncertainty: $\sigma_n = 0.1$ (accounts for measurement noise $\approx$ 0.045, process variation, and model inadequacy; see `docs/kernel_design.md` Section V.C for detailed noise estimation)
- Total budget: $N_{\text{total}} \approx n$ experiments (TODO: FILL THIS IN LATER)

---

## II. Methodology

### A. Gaussian Process Surrogate Model

We model the unknown objective function as a Gaussian process:

```math
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
```

with constant mean prior $m(\mathbf{x}) = \mu_0$ and Matérn-$\frac{1}{2}$ covariance kernel:

```math
k_{\text{Matérn}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\!\left(-\sqrt{\sum_{d=1}^{2} \frac{(x_d - x_d')^2}{\ell_d^2}}\right)
```

The ARD lengthscales $\{\ell_d\}_{d=1}^2$ enable automatic feature relevance learning, with lengthscale constraints $\ell_d \geq 0.03$ preventing overfitting. Hyperparameters $\boldsymbol{\theta} = \{\sigma_f^2, \ell_1, \ell_2, \sigma_n^2\}$ are optimized via Type-II maximum likelihood estimation (MLE).

**Posterior Predictive Distribution:**

Given observations $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, the posterior is:

```math
f(\mathbf{x}_*) \mid \mathcal{D} \sim \mathcal{N}(\mu_*, \sigma_*^2)
```

```math
\mu_* = \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}
```

```math
\sigma_*^2 = k_{**} - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
```

### B. Acquisition Function Strategy

We employ the upper confidence bound (UCB) acquisition function for batch sequential optimization:

```math
\alpha_{\text{qUCB}}(\mathbf{X}) = \mathbb{E}\left[\max_{\mathbf{x} \in \mathbf{X}} \mu(\mathbf{x}) + \beta \sigma(\mathbf{x})\right]
```

where $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_q\}$ represents a batch of $q=4$ candidate points, and $\beta=5.0$ controls exploration-exploitation tradeoff. The batch acquisition is evaluated via Monte Carlo sampling with 1024 quasi-random Sobol sequences.

**Optimization Strategy:**

[TODO: We need to determine how many samples we are making, so we can revisit when to tune the beta value. ]
- **Phase I** ($n < 60$): $\beta = 5.0$ (exploration-dominated)
- **Phase II** ($60 \leq n < 80$): $\beta = 3.0$ (balanced)
- **Phase III** ($n \geq 80$): $\beta = 2.0$ (exploitation-focused)

### C. Implementation Details

**Software Stack:**
- GP inference: GPyTorch v1.11 (GPU-accelerated exact inference)
- Acquisition optimization: BoTorch v0.9 (MC sampling, L-BFGS-B)
- Numerical computing: PyTorch v2.0 (automatic differentiation)

**Training Configuration:**
- Optimizer: Adam with learning rate $\eta = 0.003$
- Iterations: 3000 epochs (convergence criterion: $\Delta \mathcal{L} < 10^{-4}$)
- Input normalization: MinMax scaling to $[0, 1]^2$
- Output scaling: None (preserve physical units for interpretability)

---

## III. Repository Structure

```
├── config/
│   └── training_config.yaml
├── src/
│   ├── train_clean.py
│   ├── models/
│   │   ├── factory.py
│   │   └── gp.py
│   ├── preprocessing/
│   │   ├── loaders.py
│   │   └── transforms.py
│   ├── optimization/
│   │   ├── acquisition.py
│   │   └── thompson_sampler.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── training.ipynb
├── data/
├── docs/
│   ├── kernel_design.md
│   ├── acquisition_functions.md
│   └── data_preprocessing.md
└── predictions/
```

---

## IV. Reproducibility

### Installation

**System Requirements:**
- Python 3.8+ (tested on 3.11)
- CUDA 11.8+ (optional, for GPU acceleration)
- 8 GB RAM minimum (16 GB recommended)

**Environment Setup:**

```bash
git clone https://github.com/your-org/Bayesian-Optimization-Ferroelectrics
cd Bayesian-Optimization-Ferroelectrics
./setup_venv.sh
source venv/bin/activate
```

### Running the Optimization Pipeline

**Configuration:** Edit `config/training_config.yaml` to specify hyperparameters.

**Execution:**

```bash
cd src
python train_clean.py
```

**Outputs:**
- Model checkpoint: `models/model_state.pth`
- Acquisition suggestions: `predictions/next_experiments.csv`
- Training logs: Console output with loss curves, lengthscales, noise estimates

**Expected Runtime:** ~30 seconds per iteration on CPU (Intel i7), ~5 seconds with GPU (NVIDIA RTX 3080)

### Interactive Exploration

```bash
jupyter notebook src/training.ipynb
```

Key visualizations:
1. 3D response surface with confidence intervals
2. Acquisition function landscape
3. Lengthscale evolution during training
4. Cross-validation performance (RMSE, R², Spearman $\rho$)

---

## V. Performance Metrics

**Typical Performance** (41 training observations):
- RMSE: 0.13–0.15
- R²: 0.98–0.99
- Spearman $\rho$: >0.97

---

## VI. Data Format

**Input File:** Excel spreadsheet with columns:
1. `Time (ms)`
2. `Energy density new cone (J/cm^2)`
3. `2 Qsw/(U+|D|) 1e6cycles`

**Preprocessing:**
- Remove NaN values
- Filter zero FOM entries
- MinMax normalization of inputs to $[0, 1]^2$

**Output Format:** CSV with suggested experiments.

---

## VII. Advanced Usage

### Hyperparameter Sweeps

```bash
wandb sweep config/sweep.yaml
wandb agent <sweep-id>
```

### Custom Acquisition Functions

```python
from optimization.acquisition import optimize_acquisition_function

def custom_acquisition(model, likelihood, train_y):
    pass

candidates = optimize_acquisition_function(
    acq_function=custom_acquisition,
    bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    q=4
)
```

---

## VIII. References

1. Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson, A. G. (2018). *Advances in Neural Information Processing Systems*, 31.
2. Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G., & Bakshy, E. (2020). *Advances in Neural Information Processing Systems*, 33.

---
**Last Updated:** October 2025  
**Code Version:** 1.0.0  
**Compatible with:** GPyTorch 1.11+, BoTorch 0.9+, PyTorch 2.0+
