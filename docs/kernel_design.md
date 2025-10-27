# Kernel Design for Gaussian Processes

This document explains kernel selection, ARD lengthscales, and constraints.

## Gaussian Process Formulation

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

## Kernel Types

### RBF (Radial Basis Function)

\[
k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp\left(-\frac{||\mathbf{x} - \mathbf{x}'||^2}{2\ell^2}\right)
\]

**Properties**:
- Infinitely differentiable → very smooth predictions
- Good for well-behaved, smooth functions
- Risk: Can oversmooth and miss rapid changes

### Matérn Kernel (Recommended)

\[
k_{\text{Matérn}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu}r\right)^\nu K_\nu\left(\sqrt{2\nu}r\right)
\]

Where \( r = \sqrt{\sum_{d=1}^D \frac{(x_d - x_d')^2}{\ell_d^2}} \)

**Smoothness Parameter (\( \nu \))**:

| ν   | Differentiability           | Use Case             |
|-----|-----------------------------|----------------------|
| 0.5 | Once differentiable         | Very rough functions |
| 1.5 | Twice differentiable        | Most physical systems|
| 2.5 | Five times differentiable   | Smoother processes   |

**Why Matérn?**
- Flexible smoothness via ν parameter
- More robust to misspecification than RBF
- Better matches real-world phenomena (not infinitely smooth)
- **Recommended**: Matérn-5/2 (ν=2.5) balances smoothness and flexibility

## ARD (Automatic Relevance Determination)

Both kernels use **ARD lengthscales** \( \{\ell_d\}_{d=1}^D \) - one per input dimension.

### How ARD Works

- **Small lengthscale** (\( \ell_d \approx 0 \)): GP is sensitive to that dimension
  - Function varies rapidly with \( x_d \)
  - **Important feature**
- **Large lengthscale** (\( \ell_d \to \infty \)): GP is insensitive to that dimension
  - Function barely changes with \( x_d \)
  - **Irrelevant feature** - can be dropped

### Learning Feature Importance

During MLE training, lengthscales are optimized. The GP automatically:
1. Shortens lengthscales for predictive features
2. Lengthens lengthscales for irrelevant features
3. Effectively performs feature selection

**Example**: If energy density is critical but pulse time doesn't matter:
- \( \ell_{\text{energy}} \approx 0.1 \) (short, sensitive)
- \( \ell_{\text{time}} \approx 10.0 \) (long, insensitive)

## Lengthscale Constraints

**Minimum Lengthscale** (`min_lengthscale: float`):

We constrain \( \ell_d \geq \ell_{\text{min}} \) (typically 0.03) to prevent:

### Too Small (Overfitting)
- Model fits every noise wiggle
- High variance, poor generalization
- GP interpolates training points exactly

### Too Large (Underfitting)
- Model ignores local structure
- High bias, oversimplified
- GP predicts near-constant values

### Proper Range
- After input scaling to [0,1], typical lengthscales: 0.1-1.0
- Min constraint (0.03) allows sensitivity without overfitting
- Max constraint usually not needed (MLE penalizes too-large values)

## Noise Modeling

### Fixed vs Learned Noise

Our implementation uses **FixedNoiseGaussianLikelihood**:

**Advantages**:
- Faster optimization (one fewer parameter)
- More stable (avoids noise collapse)
- Better for small datasets (<100 points)

**When to Use Fixed Noise**:
- Ferroelectric experiments: noise approximately known from repeated measurements
- Small dataset regime: learning noise can be unstable
- Fast iteration needed

**Alternative: Learned Noise**:
```python
likelihood = gpytorch.likelihoods.GaussianLikelihood()
```
Use when you have >1000 points and unknown noise level.

## Hyperparameter Optimization

**Method**: Type-II Maximum Likelihood Estimation (MLE)

Optimize marginal log-likelihood:
\[
\log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}) = -\frac{1}{2} \mathbf{y}^T \mathbf{K}_y^{-1} \mathbf{y} - \frac{1}{2} \log |\mathbf{K}_y| - \frac{n}{2} \log(2\pi)
\]

Where \( \mathbf{K}_y = \mathbf{K} + \sigma_n^2 \mathbf{I} \) and \( \boldsymbol{\theta} = \{\sigma_f^2, \boldsymbol{\ell}, \sigma_n^2\} \)

**Optimizer**: L-BFGS-B (constrained optimization for positive hyperparameters)

**Priors**: Gamma priors on lengthscales to prevent degenerate solutions

## References

- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*
- GPyTorch Documentation: https://gpytorch.ai/

