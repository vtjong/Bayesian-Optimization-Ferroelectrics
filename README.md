# Bayesian Optimization of Pulse Annealing Parameters in Hafnium Zirconium Oxide Ferroelectric Thin Films

**Supporting Code Repository for Physical Review Publication**

---

## Abstract

We present a Bayesian optimization framework employing Gaussian process regression for the autonomous optimization of laser pulse annealing parameters in Hf₀.₅Zr₀.₅O₂ (HZO) ferroelectric thin films. The optimization objective maximizes the ferroelectric figure of merit $2Q_{sw}/(U+|D|)$ at $10^6$ switching cycles, balancing switching charge ($Q_{sw}$), remanent polarization ($U$), and coercive field asymmetry ($D$). Using Matérn kernel Gaussian processes with automatic relevance determination (ARD), we achieve sample-efficient exploration of the two-dimensional processing space spanning pulse duration (0.5–5.0 ms) and energy density (2.7–15.4 J/cm²). The implementation utilizes Monte Carlo acquisition functions (qUCB with $\beta=5.0$) to suggest batch experiments, enabling parallel experimental validation. This methodology demonstrates the viability of machine learning-guided processing optimization for complex ferroelectric systems with expensive characterization costs.

---

## I. Introduction

Ferroelectric HZO thin films exhibit promise for neuromorphic computing and non-volatile memory applications, but their functional properties critically depend on thermal processing conditions during crystallization. Traditional parameter space exploration via design-of-experiments or grid search scales poorly with dimensionality and requires extensive experimental resources. We address this challenge through Bayesian optimization (BO), a sequential decision-making framework that constructs probabilistic surrogate models to guide experimental exploration.

### A. Problem Formulation

**Objective Function:**

$$
\text{FOM}(\mathbf{x}) = \frac{2Q_{sw}(\mathbf{x})}{U(\mathbf{x}) + |D(\mathbf{x})|} \quad \text{at } N = 10^6 \text{ cycles}
$$

where $\mathbf{x} = [t_{\text{pulse}}, E_{\text{density}}]^\top$ represents the processing parameters.

**Search Space:**
- Pulse duration: $t_{\text{pulse}} \in [0.5, 5.0]$ ms (thermal diffusion timescale)
- Energy density: $E_{\text{density}} \in [2.7, 15.4]$ J/cm² (peak temperature control)

**Constraints:**
- Sample throughput: ~2–4 experiments per batch
- Measurement precision: $\sigma_{\text{noise}} \approx 0.1$ (normalized FOM units)
- Total budget: $N_{\text{total}} \approx n$ experiments (TODO: FILL THIS IN LATER)

---

## II. Methodology

### A. Gaussian Process Surrogate Model

We model the unknown objective function as a Gaussian process:

$$
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
$$

with constant mean prior $m(\mathbf{x}) = \mu_0$ and Matérn-$\frac{1}{2}$ covariance kernel:

$$
k_{\text{Matérn}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\sqrt{\sum_{d=1}^{2} \frac{(x_d - x_d')^2}{\ell_d^2}}\right)
$$

The ARD lengthscales $\{\ell_d\}_{d=1}^2$ enable automatic feature relevance learning, with lengthscale constraints $\ell_d \geq 0.03$ preventing overfitting. Hyperparameters $\boldsymbol{\theta} = \{\sigma_f^2, \ell_1, \ell_2, \sigma_n^2\}$ are optimized via Type-II maximum likelihood estimation (MLE).

**Posterior Predictive Distribution:**

Given observations $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$, the posterior is:

$$
f(\mathbf{x}_*) \mid \mathcal{D} \sim \mathcal{N}(\mu_*, \sigma_*^2)
$$

$$
\mu_* = \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}
$$

$$
\sigma_*^2 = k_{**} - \mathbf{k}_*^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
$$

### B. Acquisition Function Strategy

We employ the upper confidence bound (UCB) acquisition function for batch sequential optimization:

$$
\alpha_{\text{qUCB}}(\mathbf{X}) = \mathbb{E}\left[\max_{\mathbf{x} \in \mathbf{X}} \mu(\mathbf{x}) + \beta \sigma(\mathbf{x})\right]
$$

where $\mathbf{X} = \{\mathbf{x}_1, \ldots, \mathbf{x}_q\}$ represents a batch of $q=4$ candidate points, and $\beta=5.0$ controls exploration-exploitation tradeoff. The batch acquisition is evaluated via Monte Carlo sampling with 1024 quasi-random Sobol sequences.

**Optimization Strategy:**
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
│   └── training_config.yaml          # Hyperparameter configuration
├── src/
│   ├── train_clean.py                # Main optimization pipeline
│   ├── models/
│   │   ├── factory.py                # GP model construction
│   │   └── gp.py                     # ExactGP class definition
│   ├── preprocessing/
│   │   ├── loaders.py                # Experimental data I/O
│   │   └── transforms.py             # Feature scaling utilities
│   ├── optimization/
│   │   ├── acquisition.py            # Acquisition functions (qUCB, qEI, qPI)
│   │   └── thompson_sampler.py       # Thompson sampling implementation
│   ├── trainer.py                    # Type-II MLE training loop
│   ├── evaluator.py                  # Model performance metrics
│   └── training.ipynb                # Interactive workflow notebook
├── data/                              # Experimental observations (Excel format)
├── docs/                              # Detailed methodology documentation
│   ├── kernel_design.md              # Covariance function theory
│   ├── acquisition_functions.md      # BO strategy details
│   └── data_preprocessing.md         # Data pipeline documentation
└── predictions/                       # Acquisition function suggestions
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

**Configuration:** Edit `config/training_config.yaml` to specify hyperparameters:

```yaml
model:
  kernel: "matern"                    # Matérn-1/2 covariance
  matern_nu: 0.5                      # Smoothness parameter
  noise_prior: 0.1                    # Observational noise estimate
  lengthscale_prior: [1.0, 1.0]       # Initial ARD lengthscales

training:
  epochs: 3000                        # MLE iterations
  learning_rate: 0.003                # Adam step size

acquisition:
  mc_or_analytic: "mc"                # Monte Carlo sampling
  functions: ["qUCB"]                 # Acquisition strategy
  num_suggestions: 4                  # Batch size
  beta: 5.0                           # Exploration parameter
```

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

For visualization and exploratory analysis:

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

The surrogate model is evaluated via leave-one-out cross-validation on the training set:

- **RMSE** (root mean squared error): Prediction accuracy
- **MAE** (mean absolute error): Robust error measure
- **R²** (coefficient of determination): Explained variance
- **Spearman ρ** (rank correlation): Monotonicity preservation

**Typical Performance** (41 training observations):
- RMSE: 0.13–0.15
- R²: 0.98–0.99
- Spearman ρ: >0.97

---

## VI. Data Format

**Input File:** Excel spreadsheet with columns:
1. `Time (ms)`: Pulse duration
2. `Energy density new cone (J/cm^2)`: Laser energy density
3. `2 Qsw/(U+|D|) 1e6cycles`: Figure of merit @ $10^6$ cycles

**Preprocessing:**
- Remove NaN values (incomplete measurements)
- Filter zero FOM entries (failed crystallization)
- MinMax normalization of inputs to $[0, 1]^2$

**Output Format:** CSV with suggested experiments:
```
Acquisition_Function, Candidate_ID, Pulse_Time_ms, Energy_Density_J_cm2, Predicted_FOM
qUCB, 1, 1.37, 11.14, 2.88
qUCB, 2, 4.22, 14.06, 2.84
...
```

---

## VII. Advanced Usage

### Hyperparameter Sweeps

For systematic exploration of GP hyperparameters via Weights & Biases:

```bash
wandb sweep config/sweep.yaml
wandb agent <sweep-id>
```

This enables grid search over:
- Kernel choice (Matérn-$\frac{1}{2}$, Matérn-$\frac{5}{2}$, RBF)
- Noise prior values
- Learning rates and epoch counts

### Custom Acquisition Functions

To implement alternative strategies (e.g., knowledge gradient, entropy search):

```python
from optimization.acquisition import optimize_acquisition_function

# Define custom acquisition
def custom_acquisition(model, likelihood, train_y):
    # Implementation here
    pass

# Optimize
candidates = optimize_acquisition_function(
    acq_function=custom_acquisition,
    bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    q=4
)
```

---

## VIII. References

### Software Libraries

1. **GPyTorch:** Gardner, J. R., Pleiss, G., Bindel, D., Weinberger, K. Q., & Wilson, A. G. (2018). GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. *Advances in Neural Information Processing Systems*, 31.

2. **BoTorch:** Balandat, M., Karrer, B., Jiang, D. R., Daulton, S., Letham, B., Wilson, A. G., & Bakshy, E. (2020). BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. *Advances in Neural Information Processing Systems*, 33, 21524-21538.


---
**Last Updated:** October 2025  
**Code Version:** 1.0.0  
**Compatible with:** GPyTorch 1.11+, BoTorch 0.9+, PyTorch 2.0+
