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

### Configuration

All training hyperparameters are managed through YAML configuration files for reproducibility and version control.

**Main Config File**: `config/training_config.yaml`

```yaml
# Model Architecture
model:
  matern_nu: 0.5
  lengthscale_prior: [1.0, 1.0]
  train_lengthscale: true
  min_lengthscale: 0.03

# Training Parameters
training:
  epochs: 3000
  learning_rate: 0.003
  log_interval: 500
  
# Acquisition Function
acquisition:
  mc_or_analytic: "mc"
  functions: ["qUCB", "thompson"]
  num_suggestions: 4
```

**Loading Configuration in Code**:

```python
from src.config_loader import load_config, config_to_args

# Option 1: Use nested config (recommended)
config = load_config('config/training_config.yaml')
epochs = config.training.epochs
matern_nu = config.model.matern_nu

# Option 2: Flat args for backward compatibility
args = config_to_args(config)
epochs = args.epochs
```

**Quick Experiments with Overrides**:

```python
from src.config_loader import load_config_with_overrides

# Override specific parameters without modifying config file
config = load_config_with_overrides(
    'config/training_config.yaml',
    epochs=5000,
    learning_rate=0.01
)
```

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

**Clean Training Script** (Recommended):

```bash
cd src
python train_clean.py
```

This follows a clean 7-step workflow:
1. Load configuration → 2. Load data → 3. Create model → 4. Train → 5. Evaluate → 6. Suggest next experiments → 7. Export

**Module Organization**:
```python
# Training & evaluation
from trainer import train_gp_model, save_model_checkpoint
from evaluator import evaluate_model

# Bayesian Optimization
from optimization.acquisition import suggest_next_experiments_mc
from optimization.thompson_sampler import ThompsonSampler
```

**Interactive Training (Jupyter Notebook)**:

1. Open `src/training.ipynb`
2. Modify `config/training_config.yaml` to set hyperparameters
3. Run cells sequentially - config is automatically loaded

**Hyperparameter Sweeps with WandB**:

1. Configure sweep parameters in `config/sweep.yaml`
2. Initialize Weights & Biases:
```bash
wandb login
```
3. Run training with hyperparameter sweep:
```bash
wandb sweep config/sweep.yaml
wandb agent <sweep-id>
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

**Kernel Choice** (`src/models/factory.py`):

We support two kernel types, configured via `create_kernel()`:

#### 1. RBF (Radial Basis Function)

\[
k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{2\ell^2}\right)
\]

**Properties**:
- **Infinitely differentiable**: Produces very smooth predictions
- **Good for**: Well-behaved, smooth functions
- **Risk**: Can oversmooth and miss rapid changes
- **Best when**: You know the underlying function is smooth

#### 2. Matérn Kernel (Recommended)

\[
k_{\text{Matérn}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu}r\right)^\nu K_\nu\left(\sqrt{2\nu}r\right)
\]

Where \( r = \sqrt{\sum_{d=1}^D \frac{(x_d - x_d')^2}{\ell_d^2}} \)

**Matérn Smoothness Parameter (\( \nu \))**:

| \( \nu \) | Differentiability | Use Case |
|-----------|------------------|----------|
| 0.5 | Once differentiable (exponential kernel) | Very rough functions |
| 1.5 | Twice differentiable | Most physical systems |
| 2.5 | Five times differentiable | Smoother processes |

**Why Matérn?**
- **Flexible smoothness**: \( \nu \) controls differentiability
- **Robust**: More robust to misspecification than RBF
- **Physical**: Better matches real-world phenomena (not infinitely smooth)
- **Recommended**: Matérn-5/2 (\( \nu = 2.5 \)) balances smoothness and flexibility

**Alternative Kernels Considered**:
- RBF: Too smooth, assumes infinite differentiability
- Matérn-3/2: Good for rougher functions
- Spectral kernels: For periodic phenomena (not applicable here)

### ARD (Automatic Relevance Determination)

Both kernels use **ARD lengthscales** \( \{\ell_d\}_{d=1}^D \) - one per input dimension.

**How ARD Works**:
- **Small lengthscale** (\( \ell_d \approx 0 \)): GP is sensitive to that dimension
  - Function varies rapidly with \( x_d \)
  - **Important feature**
- **Large lengthscale** (\( \ell_d \to \infty \)): GP is insensitive to that dimension
  - Function barely changes with \( x_d \)
  - **Irrelevant feature** - can be dropped

**Learning Feature Importance**:
During training (MLE), lengthscales are optimized. The GP automatically:
1. Shortens lengthscales for predictive features
2. Lengthens lengthscales for irrelevant features
3. Effectively performs feature selection

**Example**: If energy density is critical but pulse time doesn't matter:
- \( \ell_{\text{energy}} \approx 0.1 \) (short, sensitive)
- \( \ell_{\text{time}} \approx 10.0 \) (long, insensitive)

### Lengthscale Constraints

**Minimum Lengthscale** (`min_lengthscale: float`):

We constrain \( \ell_d \geq \ell_{\text{min}} \) (typically 0.03) to prevent:

**Too Small (Overfitting)**:
- Model fits every noise wiggle
- High variance, poor generalization
- GP interpolates training points exactly
- Posterior mean passes through every observation

**Too Large (Underfitting)**:
- Model ignores local structure
- High bias, oversimplified
- GP predicts near-constant values
- Loses predictive power

**Proper Range**:
- After input scaling to [0,1], typical lengthscales: 0.1-1.0
- Min constraint (0.03) allows sensitivity without overfitting
- Max constraint usually not needed (MLE penalizes too-large values)

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

#### Fixed vs Learned Noise

Our implementation (`create_gp_model()`) uses **FixedNoiseGaussianLikelihood**:

**Fixed Noise (Our Approach)**:
```python
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    noise=noise_level * torch.ones(n_samples)
)
```

**Advantages**:
- **Faster optimization**: One fewer parameter to optimize
- **More stable**: Avoids noise collapsing to zero or exploding
- **Small datasets**: Better when you have <100 points
- **Known noise**: Use when experimental noise is measured/estimated

**Disadvantages**:
- **Fixed assumption**: If noise estimate is wrong, calibration suffers
- **Homoscedastic**: Assumes constant noise across parameter space

**When to Use**:
- Ferroelectric experiments: Noise is approximately known from repeated measurements
- Small dataset regime: Learning noise can be unstable
- Fast iteration: Fixed noise speeds up hyperparameter optimization

**Alternative: Learned Noise**:
```python
likelihood = gpytorch.likelihoods.GaussianLikelihood()
# Noise learned via MLE along with kernel parameters
```

Use when:
- Large dataset (>1000 points)
- Unknown noise level
- Willing to trade speed for flexibility

**Heteroscedastic Extension** (advanced):
If noise varies with \( \mathbf{x} \) (e.g., higher energy → more noise):
\[
\epsilon \sim \mathcal{N}(0, \sigma_n^2(\mathbf{x}))
\]

Implemented via separate GP for noise variance (see `gpytorch.mlls.FixedNoiseGaussianLikelihood` with input-dependent noise).

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
├── config/
│   ├── training_config.yaml  # Main training configuration
│   └── sweep.yaml            # WandB hyperparameter sweep config
│
├── src/
│   ├── config_loader.py      # Configuration loading utilities
│   │   ├── Config (class)
│   │   ├── load_config()
│   │   ├── load_config_with_overrides()
│   │   └── config_to_args()
│   │
│   ├── preprocessing/
│   │   ├── loaders.py        # Data loading and EDA
│   │   │   ├── load_experimental_data()
│   │   │   └── plot_input_output_scatter_matrix()
│   │   ├── transforms.py     # Tensor conversion and scaling
│   │   │   ├── TorchMinMaxScaler (class)
│   │   │   └── prepare_gp_training_tensors()
│   │   └── preprocess.py     # Full preprocessing pipeline
│   │
│   ├── models/
│   │   ├── factory.py        # GP model and kernel factory functions
│   │   │   ├── create_kernel()
│   │   │   └── create_gp_model()
│   │   ├── gp.py            # ExactGPModel class definition
│   │   └── __init__.py      # Model module exports
│   │
│   ├── optimization/
│   │   ├── acquisition.py    # Acquisition functions
│   │   └── bayesian_opt.py   # BO loop
│   │
│   ├── visualization/
│   │   └── plotting.py       # Result visualization
│   │
│   └── training.ipynb        # Interactive training notebook
│
├── data/                     # Experimental data
├── models/                   # Saved model checkpoints
└── predictions/              # BO predictions and suggestions
```

**Design Notes**:

1. **Configuration Management**: `config_loader.py` provides a clean interface for YAML configs with dot-notation access and backward compatibility with argparse-style code.

2. **Preprocessing Organization**: `transforms.py` contains both `TorchMinMaxScaler` and `prepare_gp_training_tensors()` because:
   - They are tightly coupled (function uses the scaler)
   - Both handle tensor transformations for GP training
   - Follows cohesion principle: related functionality stays together

3. **Model Factory Pattern**: `models/factory.py` provides factory functions for GP creation:
   - `create_kernel()`: Constructs RBF or Matérn kernels with ARD
   - `create_gp_model()`: Creates complete GP with likelihood and kernel
   - Centralizes model construction logic for consistency
   - See README Technical Deep Dive for detailed kernel design reasoning

4. **Config Files**: Separate YAML files for different purposes:
   - `training_config.yaml`: Single experiment settings
   - `sweep.yaml`: WandB hyperparameter search space

---

## References

- **GPyTorch**: https://gpytorch.ai/
- **BoTorch**: https://botorch.org/
- **Gaussian Processes for ML**: Rasmussen & Williams (2006)
- **Practical Bayesian Optimization**: Shahriari et al. (2016)
