# Bayesian Optimization for Ferroelectric Materials

This project applies Gaussian Process (GP) regression and Bayesian Optimization to optimize pulse annealing parameters for ferroelectric thin films. The goal is to maximize device performance while minimizing expensive experimental trials.

## Table of Contents
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Data Properties](#data-properties)
- [ML Design Decisions](#ml-design-decisions)
- [Usage](#usage)
- [Technical Deep Dive](#technical-deep-dive)

---

## Installation

Install the packages listed in `requirements.txt`. Source code relevant to this project is found in the `src` folder.

### Via conda
* Make a new environment with name "FEGP": 
```bash
conda create -n FEGP
```
* Activate new environment: 
```bash
conda activate FEGP
```
* Install packages into environment: 
```bash
conda install --file requirements.txt
```

### Core Dependencies
Gaussian process functionality provided by `gpytorch`, Bayesian optimization framework provided by `botorch`. Acquisition functions were written to fit with the `botorch` API.

---

## Project Overview

### Problem Statement
Ferroelectric materials require precise thermal processing (pulse annealing) to achieve optimal performance. The parameter space is large, experiments are expensive, and the objective function is noisy and non-convex. Traditional grid search is inefficient.

### Solution Approach
We use **Bayesian Optimization** with **Gaussian Process** surrogate models to:
1. Model the relationship between annealing parameters and device performance
2. Quantify uncertainty in unexplored regions
3. Intelligently select next experiments via acquisition functions
4. Maximize performance with minimal experimental budget

---

## Data Properties

### Input Parameters (Features)
Our experimental design space has 2 input dimensions:

1. **Energy Density** (`J/cm²`)
   - Controls peak temperature during pulse annealing
   - Physical range: typically 0.5-5.0 J/cm²
   - Critical for crystallization without thermal damage

2. **Pulse Width** (`ms`)
   - Controls thermal diffusion time
   - Physical range: typically 0.1-2.0 ms
   - Affects temperature profile and cooling rate

### Output Metric (Target)
**Figure of Merit**: `2 Qsw/(U+|D|) 1e6cycles`

This metric quantifies ferroelectric device quality:
- **Qsw**: Switchable polarization (desired signal)
- **U**: Up-state non-switching charge (leakage)
- **D**: Down-state non-switching charge (leakage)
- **Higher is better**: Maximize switching efficiency, minimize parasitic charge

Physical interpretation: Devices with high figure of merit have:
- Large polarization switching
- Low leakage current
- Good cyclability over 1M+ cycles

---

## ML Design Decisions

### Why Gaussian Processes?

1. **Small Data Regime**: With <100 experimental points, GPs provide:
   - Principled uncertainty quantification
   - No overfitting with proper priors
   - Smooth interpolation between sparse points

2. **Non-parametric Flexibility**: 
   - No assumption about functional form
   - Captures non-linear physics without manual feature engineering
   - Kernel learns smoothness from data

3. **Uncertainty-Aware Optimization**:
   - GP posterior variance drives exploration
   - Balances exploitation (high mean) vs exploration (high variance)
   - Critical for expensive experimental settings

### Data Preprocessing Philosophy

**Aggressive filtering is essential for GP performance.** Our preprocessing (`src/preprocessing/loaders.py`) applies strict quality filters:

#### Why Remove NaN Values?
- GPs require complete observations for training
- Missing target values destabilize covariance matrix inversions
- Cannot compute likelihood \( p(y|X) \) with undefined \( y \)

#### Why Remove Zero Measurements?
- Zero FOM indicates **experimental failure**, not a valid operating point
- Physical impossibility: Formula involves division by polarization
- Zero values are outliers that corrupt lengthscale learning
- GPs are sensitive to outliers in small datasets

#### Why This Matters for GPs:
- **Lengthscale Learning**: GP kernel lengthscales encode smoothness. Outliers bias lengthscales toward too-smooth or too-rough fits
- **Covariance Stability**: Clean data ensures well-conditioned kernel matrices
- **Acquisition Function Quality**: Noisy GP posteriors lead to poor next-experiment selection

### Exploratory Data Analysis

**Visualization Strategy** (`plot_input_output_scatter_matrix`):

We use scatter matrices to inform GP modeling decisions:

1. **Identify Input-Output Correlations**:
   - Guides lengthscale priors in ARD (Automatic Relevance Determination) kernels
   - If energy density shows strong correlation, GP will learn shorter lengthscale
   - If pulse width is less predictive, GP will learn longer lengthscale (downweight)

2. **Detect Non-linear Relationships**:
   - Justifies GP over linear regression
   - Curved relationships → use Matérn or RBF kernels
   - Discontinuities → may need specialized kernels

3. **Spot Outliers or Clusters**:
   - Informs noise model selection (homoscedastic vs heteroscedastic)
   - Identifies sub-regions where separate GPs might be needed

4. **Understand Parameter Coupling**:
   - Non-axis-aligned patterns suggest need for non-separable kernels
   - Helps decide between product kernels vs additive kernels

**Scatter Matrix Interpretation**:
- **Diagonal**: Marginal distributions of each variable (check for multimodality)
- **Off-diagonal**: Bivariate relationships (check for linearity, monotonicity)
- **Overall**: Guides choice of Matérn-5/2 vs RBF vs spectral kernels

---

## Usage

### Preprocessing
To run the data preprocessing script:
1. Update directory path in `src/preprocess.py`:
```python
dir = '/your/path/Bayesian-Optimization-Ferroelectrics/data/KHM010_'
```
2. Run preprocessing:
```bash
python src/preprocess.py
```
3. Processed data will be saved to `data/processed/`

### Training
To run the Gaussian Process training:
1. Navigate to `src/main.py`
2. Configure hyperparameter sweep in `config/sweep.yaml`
3. Initialize Weights & Biases:
```bash
wandb login
```
4. Run training with hyperparameter sweep:
```bash
wandb sweep config/sweep.yaml
```

**WandB Resources**:
- Quickstart: https://docs.wandb.ai/quickstart
- Sweep Tutorial: https://docs.wandb.ai/guides/sweeps/quickstart

### Bayesian Optimization
After training, run optimization loop:
```python
from src.optimization import bayesian_optimize
from src.models import load_gp_model

# Load trained GP
model = load_gp_model('models/model_state.pth')

# Run BO with acquisition function
next_experiments = bayesian_optimize(
    model=model,
    acquisition='expected_improvement',
    n_candidates=5
)
```

---

## Technical Deep Dive

### Gaussian Process Formulation

**Model**: \( f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')) \)

Where:
- \( m(\mathbf{x}) \): Mean function (typically zero or constant)
- \( k(\mathbf{x}, \mathbf{x}') \): Covariance kernel (encodes smoothness assumptions)

**Posterior Predictive**:

Given training data \( \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n \):

\[
f_* | \mathcal{D}, \mathbf{x}_* \sim \mathcal{N}(\mu_*, \sigma_*^2)
\]

Where:
\[
\mu_* = \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}
\]
\[
\sigma_*^2 = k_{**} - \mathbf{k}_*^T (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_*
\]

### Kernel Design

**Primary Kernel**: Matérn-5/2 with ARD

\[
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \left(1 + \sqrt{5}r + \frac{5r^2}{3}\right) \exp(-\sqrt{5}r)
\]

Where \( r = \sqrt{\sum_{d=1}^D \frac{(x_d - x_d')^2}{\ell_d^2}} \)

**Why Matérn-5/2?**
- Twice differentiable (smooth but not infinitely smooth like RBF)
- More robust to misspecification than RBF
- ARD lengthscales \( \{\ell_d\} \) automatically learn input relevance

**Alternative Kernels Considered**:
- RBF (too smooth, can overfit)
- Matérn-3/2 (less smooth, better for rough functions)
- Spectral kernels (for periodic phenomena)

### Acquisition Functions

We implement several acquisition functions in `src/optimization/`:

#### 1. Expected Improvement (EI)
\[
\alpha_{\text{EI}}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f(\mathbf{x}^+), 0)]
\]

Best for: Exploitation-focused search when near optimum

#### 2. Upper Confidence Bound (UCB)
\[
\alpha_{\text{UCB}}(\mathbf{x}) = \mu(\mathbf{x}) + \beta \sigma(\mathbf{x})
\]

Best for: Tunable exploration (via \( \beta \)) in early stages

#### 3. Thompson Sampling
Sample functions from GP posterior, optimize sampled function

Best for: Balanced exploration-exploitation with probabilistic guarantees

#### 4. Knowledge Gradient (KG)
\[
\alpha_{\text{KG}}(\mathbf{x}) = \mathbb{E}[\max_{\mathbf{x}'} \mu_{n+1}(\mathbf{x}') - \max_{\mathbf{x}'} \mu_n(\mathbf{x}')]
\]

Best for: Finite-horizon optimization when budget is limited

### Noise Modeling

**Observation Model**:
\[
y = f(\mathbf{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2)
\]

**Why Model Noise?**
- Experimental measurements have inherent variability
- Multiple measurements at same \( \mathbf{x} \) may differ
- GP noise variance \( \sigma_n^2 \) prevents overfitting to noise

**Heteroscedastic Extension** (optional):
If noise varies with \( \mathbf{x} \) (e.g., higher energy → more noise):
\[
\epsilon \sim \mathcal{N}(0, \sigma_n^2(\mathbf{x}))
\]

Implemented via separate GP for noise variance.

### Hyperparameter Optimization

**Method**: Type-II Maximum Likelihood Estimation (MLE)

Optimize marginal log-likelihood:
\[
\log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \mathbf{y} - \frac{1}{2} \log |\mathbf{K}_y| - \frac{n}{2} \log(2\pi)
\]

Where \( \mathbf{K}_y = \mathbf{K} + \sigma_n^2 \mathbf{I} \) and \( \boldsymbol{\theta} = \{\sigma_f^2, \boldsymbol{\ell}, \sigma_n^2\} \)

**Optimizer**: L-BFGS-B (constrained optimization for positive hyperparameters)

**Priors**: Gamma priors on lengthscales to prevent degenerate solutions

### Scaling and Normalization

**Critical for GP performance** - implemented in `src/preprocessing/transforms.py`

#### Why Feature Scaling is Essential for GPs

1. **Distance-Based Kernels**: GPs use kernels (RBF, Matérn) that compute distances between points. Without scaling:
   - Features with large ranges dominate distance calculations
   - Features with small ranges become irrelevant
   - Example: Energy (0-5 J/cm²) vs Time (0.1-2 ms) - energy would dominate

2. **Numerical Stability**: MinMax scaling to [0,1] ensures:
   - Well-conditioned covariance matrices (no near-singular inversions)
   - Stable gradient computations during MLE optimization
   - Prevents overflow/underflow in exponential kernel calculations

3. **Equal Initial Importance**: Scaling gives all features equal weight initially
   - ARD (Automatic Relevance Determination) lengthscales learn relative importance
   - GP discovers which features matter through data, not scale artifacts

#### Input Scaling Strategy

**MinMax to [0,1]**:
\[
\tilde{x}_d = \frac{x_d - \min(x_d)}{\max(x_d) - \min(x_d)}
\]

Implemented via `TorchMinMaxScaler` in `transforms.py`

**Input Ordering**: `[Time, Energy]`
- Time affects thermal diffusion (physical lengthscale ~1-5 ms)
- Energy affects peak temperature (physical lengthscale ~3-15 J/cm²)
- ARD kernel learns if one dimension is more predictive

#### Output Scaling Strategy

**Figure of Merit is NOT scaled**:
- Preserves interpretability of predictions
- Domain experts understand raw FOM values
- GP learns output scale through kernel `outputscale` parameter
- Noise model operates in original output units

If output scaling is needed (e.g., for numerical stability with extreme values):
\[
\tilde{y} = \frac{y - \mu_y}{\sigma_y}
\]

**Inverse Transform for Predictions**:
\[
y = \sigma_y \cdot \tilde{y} + \mu_y
\]

#### Practical Implementation

From `prepare_gp_training_tensors()`:
```python
# Inputs: scaled to [0,1] for numerical stability
train_x = scaler.fit_transform(train_x)

# Outputs: unscaled for interpretability
train_y = torch.Tensor(fe_data["FOM"].values)
```

**Important for Bayesian Optimization**:
- In production BO: only fit scaler on **observed** data
- When suggesting new experiments, transform using fitted scaler
- Never leak information from "future" experiments into scaler fitting

### Model Selection and Validation

**Cross-Validation Strategy**:
- K-fold CV with spatial awareness (prevent data leakage in parameter space)
- Metrics: RMSE, negative log-predictive density (NLPD), calibration error

**Avoiding Overfitting**:
- Marginal likelihood automatically trades off fit and complexity
- ARD kernels zero-out irrelevant dimensions
- Early stopping based on validation NLPD

---

## File Structure

```
src/
├── preprocessing/
│   ├── loaders.py           # Data loading and EDA
│   │   ├── load_experimental_data()
│   │   └── plot_input_output_scatter_matrix()
│   ├── transforms.py        # Tensor conversion and scaling
│   │   ├── TorchMinMaxScaler (class)
│   │   └── prepare_gp_training_tensors()
│   └── preprocess.py        # Full preprocessing pipeline
├── models/
│   └── gp_models.py         # GP model definitions
├── optimization/
│   ├── acquisition.py       # Acquisition functions
│   └── bayesian_opt.py      # BO loop
├── visualization/
│   └── plotting.py          # Result visualization
└── training.ipynb           # Interactive training notebook
```

**Design Note**: `transforms.py` contains both `TorchMinMaxScaler` and `prepare_gp_training_tensors()` in a single file because:
- They are tightly coupled (the function uses the scaler)
- Both handle tensor transformations for the same purpose (GP training)
- Splitting would create unnecessary file overhead for ~200 lines
- Follows the cohesion principle: related functionality stays together

---

## References

- **GPyTorch**: https://gpytorch.ai/
- **BoTorch**: https://botorch.org/
- **Gaussian Processes for ML**: Rasmussen & Williams (2006)
- **Practical Bayesian Optimization**: Shahriari et al. (2016)

---

## Contributing

This project is under active development. For questions or collaboration, please open an issue.

## License

[Add license information]   
