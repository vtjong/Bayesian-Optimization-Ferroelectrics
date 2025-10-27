# Supplementary Information: Kernel Design and Hyperparameter Optimization

**Technical Documentation for Gaussian Process Surrogate Model**

---

## I. Gaussian Process Formulation

### A. Probabilistic Model

The unknown objective function $f: \mathbb{R}^d \rightarrow \mathbb{R}$ is modeled as a realization from a Gaussian process:

```math
f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))
```

where:
- $m(\mathbf{x})$: Mean function (constant prior $m(\mathbf{x}) = \mu_0$)
- $k(\mathbf{x}, \mathbf{x}')$: Covariance kernel encoding smoothness assumptions

### B. Posterior Predictive Distribution

Given experimental observations $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$:

```math
f_{*} \mid \mathcal{D}, \mathbf{x}_{*} \sim \mathcal{N}(\mu_{*}, \sigma_{*}^2)
```

where the predictive mean and variance are:

```math
\mu_{*} = \mathbf{k}_{*}^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{y}
```

```math
\sigma_{*}^2 = k_{**} - \mathbf{k}_{*}^\top (\mathbf{K} + \sigma_n^2 \mathbf{I})^{-1} \mathbf{k}_{*}
```

Here, $\mathbf{K}$ is the $n \times n$ Gram matrix with entries $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$, $\mathbf{k}_{*}$ is the $n$-vector of covariances between test point $\mathbf{x}_*$ and training points, and $k_{**} = k(\mathbf{x}_*, \mathbf{x}_*)$.

---

## II. Covariance Kernel Selection

### A. Radial Basis Function (RBF) Kernel

The squared exponential or RBF kernel is defined as:

```math
k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \sigma_f^2 
\exp\!\left(-\frac{\lVert \mathbf{x} - \mathbf{x}' \rVert^2}{2\ell^2}\right)
```

**Properties:**
- Infinitely differentiable → extremely smooth sample paths
- Universal approximator for continuous functions
- Risk of over-smoothing rapid transitions

**Use Case:** Smooth, well-behaved objective functions without sharp discontinuities.

---

### B. Matérn Kernel (Recommended)

The Matérn kernel provides tunable smoothness via parameter $\nu$:

```math
k_{\text{Matérn}}(\mathbf{x}, \mathbf{x}') = 
\sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)} 
\left(\sqrt{2\nu}\, r\right)^{\nu} 
K_{\nu}\!\left(\sqrt{2\nu}\, r\right)
```

where  
$r = \sqrt{\sum_{d=1}^{D} \frac{(x_d - x_d')^2}{\ell_d^2}}$ (ARD distance metric)

and $K_{\nu}$ denotes the modified Bessel function of the second kind.

**Smoothness Parameter ($\nu$)**:

| $\nu$ | Differentiability | Sample Path Regularity | Application Domain |
|-------|-------------------|------------------------|-------------------|
| 0.5   | Once differentiable | Non-smooth, rough | Noisy physical systems |
| 1.5   | Twice differentiable | Moderately smooth | Most engineering processes |
| 2.5   | Five times differentiable | Very smooth | Continuous manufacturing |

**Simplified Forms:**

For specific $\nu$ values, the Matérn kernel simplifies:

- **$\nu = \frac{1}{2}$ (Exponential):**
```math
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \exp(-r)
```

- **$\nu = \frac{3}{2}$:**
```math
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 (1 + \sqrt{3}r) \exp(-\sqrt{3}r)
```

- **$\nu = \frac{5}{2}$:**
```math
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 \left(1 + \sqrt{5}r + \frac{5r^2}{3}\right) \exp(-\sqrt{5}r)
```

**Advantages:**
- Flexible smoothness control via $\nu$
- More robust to model misspecification than RBF
- Better match for physical processes (finite smoothness)
- Computational efficiency for $\nu \in \{0.5, 1.5, 2.5\}$

**Recommendation:** Matérn-$\frac{5}{2}$ ($\nu = 2.5$) provides excellent balance between smoothness and flexibility for most materials optimization problems.

---

## III. Automatic Relevance Determination (ARD)

### A. Anisotropic Lengthscales

Both kernels employ dimension-specific lengthscales $\{\ell_d\}_{d=1}^D$ (one per input dimension):

```math
r(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{d=1}^{D} \frac{(x_d - x_d')^2}{\ell_d^2}}
```

This generalizes isotropic kernels by allowing different characteristic length scales along each axis.

### B. Feature Relevance Interpretation

The lengthscale $\ell_d$ quantifies the relevance of dimension $d$:

- **Small lengthscale** ($\ell_d \approx 0$):  
  GP is highly sensitive to changes in $x_d$  
  Function varies rapidly along dimension $d$  
  **Interpretation:** Important feature for prediction

- **Large lengthscale** ($\ell_d \to \infty$):  
  GP is insensitive to $x_d$  
  Function barely changes with $x_d$  
  **Interpretation:** Irrelevant feature (can be pruned)

### C. Automatic Feature Selection

During Type-II MLE training, lengthscales are optimized to maximize marginal likelihood. The GP automatically:

1. Shortens $\ell_d$ for predictive features
2. Lengthens $\ell_d$ for noise dimensions
3. Performs implicit feature selection via ARD

**Example:** In ferroelectric processing optimization:
- Pulse energy density: $\ell_{\text{energy}} \approx 0.1$ (short, critical)
- Pulse duration: $\ell_{\text{time}} \approx 0.3$ (moderate, important)
- Ambient temperature: $\ell_{\text{ambient}} \approx 10.0$ (long, irrelevant)

---

## IV. Lengthscale Constraints

### A. Minimum Lengthscale Constraint

We enforce $\ell_d \geq \ell_{\min}$ (typically $\ell_{\min} = 0.03$) to prevent pathological behavior:

**Problem: Too Small ($\ell_d \to 0$)**
- Model fits noise rather than signal (overfitting)
- High variance, poor generalization
- GP interpolates training points exactly: $\mu(\mathbf{x}_i) = y_i$ for all $i$
- Posterior uncertainty collapses to zero at observations

**Problem: Too Large ($\ell_d \to \infty$)**
- Model ignores local structure (underfitting)
- High bias, oversimplified predictions
- GP predicts nearly constant values across input space
- Fails to capture functional dependencies

**Optimal Range:**
After input scaling to $[0,1]^d$, empirical lengthscales typically fall within:
- $\ell_d \in [0.1, 1.0]$ for relevant features
- $\ell_d > 2.0$ for irrelevant features

The minimum constraint ($\ell_{\min} = 0.03$) allows sufficient sensitivity without overfitting.

### B. Implementation

In GPyTorch, lengthscale constraints are specified via:

```python
gpytorch.constraints.GreaterThan(min_lengthscale)
```

This applies a soft constraint through reparameterization:
```math
\ell_d = \ell_{\min} + \text{softplus}(\tilde{\ell}_d)
```

where $\tilde{\ell}_d \in \mathbb{R}$ is the unconstrained parameter optimized during training.

---

## V. Noise Modeling

### A. Fixed Noise Likelihood

We employ `FixedNoiseGaussianLikelihood` with predetermined noise level $\sigma_n^2$:

```math
y_i = f(\mathbf{x}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_n^2)
```

**Advantages:**
- Faster optimization (one fewer hyperparameter)
- More stable training (avoids noise collapse $\sigma_n^2 \to 0$)
- Better suited for small datasets ($n < 100$)

**When to Use:**
- Noise level approximately known from experimental repeatability
- Limited data regime where learning noise is unstable
- Rapid iteration required (fewer parameters to optimize)

### B. Learned Noise Likelihood

Alternative: `GaussianLikelihood` with trainable noise:

```python
likelihood = gpytorch.likelihoods.GaussianLikelihood()
```

**When to Use:**
- Large datasets ($n > 1000$)
- Unknown or heteroskedastic noise
- Sufficient data to reliably estimate $\sigma_n^2$

### C. Noise Level Estimation and Selection

#### 1. Noise Sources in Ferroelectric Measurements

Total experimental uncertainty comprises multiple independent sources:

**Measurement Noise ($\sigma_{\text{meas}}$):**
- Device-to-device variability under identical conditions
- Equipment precision limitations
- Testing protocol repeatability

**Process Variation ($\sigma_{\text{proc}}$):**
- Film quality heterogeneity
- Pulse annealing spatial non-uniformity
- Sample preparation variations
- Environmental fluctuations (temperature, humidity)

**Model Inadequacy ($\sigma_{\text{model}}$):**
- Unmodeled physics (grain structure, defects)
- Discretization of continuous processes
- Interaction terms not captured by GP

**Total Noise:**
```math
\sigma_n^2 = \sigma_{\text{meas}}^2 + \sigma_{\text{proc}}^2 + \sigma_{\text{model}}^2
```

#### 2. Empirical Estimation Methodology

**Lower Bound (Measurement Only):**

Device-to-device variation in 2$P_r$ measurements on 10μm × 10μm capacitors:
- $N = 20$ devices at fixed processing conditions
- Standard deviation: $\sigma_{\text{meas}} \approx 0.045$
- **Limitation:** Underestimates total experimental uncertainty

**Conservative Estimate (Recommended):**

Accounting for process and model uncertainty:
- Typical ferroelectric variability: 5-15% relative error
- For FOM $\in [0.5, 4.5]$, absolute noise: $\sigma_n \approx 0.1$
- **Rationale:** Prevents overfitting, matches RMSE $\approx 0.13$ on held-out data

**Empirical Validation (Gold Standard):**

Direct measurement via replicate experiments:
1. Select 2-3 representative processing conditions
2. Fabricate 3-5 replicate samples per condition
3. Measure FOM for all replicates
4. Compute pooled standard deviation across conditions
5. Use as $\sigma_n$ prior

Expected range: $\sigma_n \in [0.08, 0.15]$ for HZO thin films

#### 3. Selection Guidelines

| Dataset Size | Noise Known? | Recommendation |
|--------------|--------------|----------------|
| $n < 50$ | No | Fixed $\sigma_n = 0.1$ (conservative) |
| $n < 50$ | Yes (replicates) | Fixed $\sigma_n = \sigma_{\text{empirical}}$ |
| $50 \leq n < 200$ | No | Fixed $\sigma_n = 0.1$ or learned with strong prior |
| $n \geq 200$ | Any | Learned via `GaussianLikelihood()` |

**Current Study:** Fixed $\sigma_n = 0.1$ based on:
- Limited replicate data (measurement-only estimate = 0.045)
- Conservative accounting for process variation (factor of ~2×)
- Cross-validation RMSE validation (0.13 suggests true noise ~0.1)
- Standard practice for materials Bayesian optimization

#### 4. Sensitivity Analysis

**Impact of noise underestimation** ($\sigma_n = 0.045$ vs. truth $\approx 0.1$):
- GP overfits measurement noise
- Posterior variance too small (overconfident)
- Acquisition function under-explores
- Risk of premature convergence to local optimum

**Impact of noise overestimation** ($\sigma_n = 0.2$ vs. truth $\approx 0.1$):
- GP overly smooth (underfitting)
- Posterior variance too large (underconfident)
- Acquisition function over-explores
- Slower convergence, but safer

**Recommendation:** Err on side of overestimation (0.1-0.15) rather than underestimation.

---

## VI. Hyperparameter Optimization

### A. Type-II Maximum Likelihood Estimation

Hyperparameters $\boldsymbol{\theta} = \{\sigma_f^2, \boldsymbol{\ell}, \sigma_n^2\}$ are optimized by maximizing the marginal log-likelihood:

```math
\log p(\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}) =
-\frac{1}{2} \mathbf{y}^\top \mathbf{K}_y^{-1} \mathbf{y}
-\frac{1}{2} \log \lvert \mathbf{K}_y \rvert
-\frac{n}{2} \log(2\pi)
```

where $\mathbf{K}_y = \mathbf{K} + \sigma_n^2 \mathbf{I}$ is the noisy covariance matrix.

### B. Optimization Algorithm

**Optimizer:** Adam (adaptive moment estimation)
- Learning rate: $\eta = 0.003$
- Iterations: 3000 epochs
- Convergence criterion: $|\Delta \mathcal{L}| < 10^{-4}$

Adam combines advantages of:
- AdaGrad: Adaptive learning rates per parameter
- RMSProp: Exponential moving average of squared gradients
- Momentum: Accelerated convergence

### C. Hyperparameter Priors

To regularize optimization and prevent degenerate solutions, we place priors on hyperparameters:

**Lengthscale Prior:**
```math
\ell_d \sim \text{Gamma}(\alpha_\ell, \beta_\ell)
```
Typical values: $\alpha_\ell = 3$, $\beta_\ell = 6$ (mode at 0.33, mean at 0.5)

**Outputscale Prior:**
```math
\sigma_f^2 \sim \text{Gamma}(\alpha_\sigma, \beta_\sigma)
```
Typical values: $\alpha_\sigma = 2$, $\beta_\sigma = 0.15$ (weakly informative)

These priors prevent:
- Lengthscales collapsing to zero
- Output variance exploding to infinity
- Overfitting in small-data regime

### D. Training Monitoring

Key metrics logged during optimization:
1. **Marginal log-likelihood:** Should increase monotonically
2. **Lengthscales:** Should stabilize after initial transient
3. **Noise level:** Should remain bounded ($\sigma_n \in [0.01, 0.5]$)
4. **Outputscale:** Should match output variance order of magnitude

---

## VII. Computational Considerations

### A. Complexity Analysis

**Training (per iteration):**
- Covariance matrix construction: $O(n^2 d)$
- Cholesky decomposition: $O(n^3)$
- Likelihood evaluation: $O(n^2)$
- Total: $O(n^3)$ dominated by matrix inversion

**Prediction:**
- Posterior mean: $O(n^2 d + n^2)$
- Posterior variance: $O(n^2)$

**Scalability:** Exact GP inference becomes prohibitive for $n > 10{,}000$. For larger datasets, consider:
- Sparse variational GPs (Titsias, 2009)
- Structured kernel interpolation (Wilson & Nickisch, 2015)
- GPU acceleration via GPyTorch

### B. Numerical Stability

To ensure robust inversion of $\mathbf{K}_y$:
1. Add jitter: $\mathbf{K}_y \leftarrow \mathbf{K}_y + 10^{-6} \mathbf{I}$
2. Use Cholesky decomposition instead of direct inversion
3. Monitor condition number: $\kappa(\mathbf{K}_y) < 10^{10}$

---

## VIII. References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

2. Gardner, J. R., Pleiss, G., Weinberger, K. Q., Bindel, D., & Wilson, A. G. (2018). GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. *Advances in Neural Information Processing Systems*, 31.

3. MacKay, D. J. C. (1998). Introduction to Gaussian processes. *Neural Networks and Machine Learning*, 168, 133-165.

4. Titsias, M. (2009). Variational learning of inducing variables in sparse Gaussian processes. *Artificial Intelligence and Statistics*, 567-574.

5. Wilson, A. G., & Nickisch, H. (2015). Kernel interpolation for scalable structured Gaussian processes (KISS-GP). *International Conference on Machine Learning*, 1775-1784.

---

**Last Updated:** October 2025  
**Compatible with:** GPyTorch 1.11+, PyTorch 2.0+
