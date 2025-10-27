# Supplementary Information: Acquisition Function Theory and Implementation

**Technical Documentation for Bayesian Optimization Strategy**

---

## I. Acquisition Function Framework

Acquisition functions $\alpha: \mathcal{X} \rightarrow \mathbb{R}$ guide sequential experimental design by balancing:
- **Exploitation:** Sample where predicted FOM is high ($\mu(\mathbf{x})$ large)
- **Exploration:** Sample where uncertainty is high ($\sigma(\mathbf{x})$ large)

The next experiment is selected by maximizing the acquisition function:

```math
\mathbf{x}_{n+1} = \arg\max_{\mathbf{x} \in \mathcal{X}} \alpha(\mathbf{x} \mid \mathcal{D}_n)
```

---

## II. Analytic Acquisition Functions

### A. Expected Improvement (EI)

The expected improvement over current best observation $f^+ = \max_{i=1}^n y_i$ is:

```math
\alpha_{\text{EI}}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f^+, 0)]
```

**Closed-Form Solution:**

```math
\alpha_{\text{EI}}(\mathbf{x}) = 
\begin{cases}
(\mu(\mathbf{x}) - f^+) \Phi(Z) + \sigma(\mathbf{x}) \phi(Z) & \text{if } \sigma(\mathbf{x}) > 0 \\
0 & \text{if } \sigma(\mathbf{x}) = 0
\end{cases}
```

where $Z = \frac{\mu(\mathbf{x}) - f^+}{\sigma(\mathbf{x})}$, and $\Phi(\cdot)$, $\phi(\cdot)$ denote standard normal CDF and PDF.

**Properties:**
- Exploitative: Strongly favors regions near current optimum
- Zero at observed points: $\alpha_{\text{EI}}(\mathbf{x}_i) = 0$
- Scales with uncertainty: Large $\sigma$ increases EI

**Best For:** Late-stage optimization when approximate optimum location is known.

---

### B. Probability of Improvement (PI)

The probability of improving over $f^+$:

```math
\alpha_{\text{PI}}(\mathbf{x}) = P(f(\mathbf{x}) > f^+) = \Phi\left(\frac{\mu(\mathbf{x}) - f^+}{\sigma(\mathbf{x})}\right)
```

**Properties:**
- Most conservative: Only considers improvement probability, not magnitude
- Focuses on high-confidence improvements
- Can be too exploitative

**Best For:** Risk-averse optimization when false positives are costly.

---

### C. Upper Confidence Bound (UCB)

The UCB acquisition function directly balances mean and uncertainty:

```math
\alpha_{\text{UCB}}(\mathbf{x}) = \mu(\mathbf{x}) + \beta \sigma(\mathbf{x})
```

**Hyperparameter:** $\beta > 0$ controls exploration-exploitation tradeoff
- $\beta \to 0$: Pure exploitation (greedy)
- $\beta \to \infty$: Pure exploration (uniform sampling)
- $\beta \in [2, 5]$: Practical range for most problems

**Theoretical Foundation:** GP-UCB algorithm (Srinivas et al., 2010) provides regret bounds:

```math
\beta_t = 2 \log\left(\frac{t^2 \pi^2}{6\delta}\right)
```

for sub-linear cumulative regret $R_T = O(\sqrt{T \gamma_T \log T})$ with probability $1-\delta$.

**Properties:**
- Simple and interpretable
- Tunable exploration via $\beta$
- No dependence on current best (unlike EI, PI)
- Amenable to theoretical analysis

**Best For:** General-purpose optimization with tunable exploration, especially in early stages.

---

## III. Monte Carlo Acquisition Functions

### A. Batch Optimization Motivation

Analytic acquisition functions optimize one point at a time. For parallel experimentation, we require batch acquisition:

```math
\mathbf{X}_{n+1} = \{\mathbf{x}_{n+1}^{(1)}, \ldots, \mathbf{x}_{n+1}^{(q)}\} = \arg\max_{\mathbf{X} \subset \mathcal{X}} \alpha(\mathbf{X} \mid \mathcal{D}_n)
```

Computing the joint acquisition exactly is intractable. Monte Carlo (MC) methods approximate the expectation via sampling.

---

### B. q-Expected Improvement (qEI)

Batch extension of EI:

```math
\alpha_{\text{qEI}}(\mathbf{X}) = \mathbb{E}\left[\max\left(\max_{i=1}^q f(\mathbf{x}^{(i)}) - f^+, 0\right)\right]
```

**MC Approximation:**
```math
\alpha_{\text{qEI}}(\mathbf{X}) \approx \frac{1}{N} \sum_{j=1}^N \max\left(\max_{i=1}^q f^{(j)}(\mathbf{x}^{(i)}) - f^+, 0\right)
```

where $f^{(j)} \sim \mathcal{GP}(\mu, k)$ are posterior sample paths.

**Implementation:** Reparameterization trick + quasi-Monte Carlo (Sobol sequences) for variance reduction.

---

### C. q-Probability of Improvement (qPI)

Batch probability that at least one point improves:

```math
\alpha_{\text{qPI}}(\mathbf{X}) = P\left(\max_{i=1}^q f(\mathbf{x}^{(i)}) > f^+\right)
```

**Properties:**
- Conservative batch selection
- Diversity through joint probability
- Less prone to duplicate suggestions than sequential PI

---

### D. q-Upper Confidence Bound (qUCB)

Batch UCB maximizes joint upper confidence:

```math
\alpha_{\text{qUCB}}(\mathbf{X}) = \mathbb{E}\left[\max_{i=1}^q \mu(\mathbf{x}^{(i)}) + \beta \sigma(\mathbf{x}^{(i)})\right]
```

**Recommended Configuration:**
- Batch size: $q = 4$ (matches experimental throughput)
- Exploration parameter: $\beta = 5.0$ (early stage), $\beta = 2.0$ (late stage)
- MC samples: $N = 1024$ (Sobol sequence)

**Properties:**
- Naturally encourages diversity (high $\sigma$ at unexplored regions)
- Tunable via $\beta$ (same interpretation as analytic UCB)
- Computationally efficient via quasi-MC

**Best For:** Parallel Bayesian optimization with tunable exploration, especially for expensive experiments.

---

## IV. Thompson Sampling

### A. Conceptual Framework

Thompson sampling (posterior sampling) selects experiments by:
1. Sample function from GP posterior: $f^{(s)} \sim \mathcal{GP}(\mu, k \mid \mathcal{D}_n)$
2. Optimize sampled function: $\mathbf{x}_{n+1} = \arg\max_{\mathbf{x}} f^{(s)}(\mathbf{x})$

**Interpretation:** Randomized acquisition matching posterior belief about optimum location.

### B. Theoretical Properties

**Regret Bound:** For Gaussian processes, Thompson sampling achieves Bayesian regret:

```math
\text{BayesRegret}(T) = O(\sqrt{T \gamma_T})
```

matching information-theoretic lower bounds (Russo & Van Roy, 2014).

**Exploration-Exploitation:** Naturally balances via posterior uncertainty—no tuning parameter needed.

### C. Implementation

**Efficient Sampling:** Via Matheron's rule (Wilson et al., 2020):
```math
f(\mathbf{x}) = \mu(\mathbf{x}) + \mathbf{k}_*^\top \mathbf{L}^{-\top} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
```

where $\mathbf{L} \mathbf{L}^\top = \mathbf{K}_y$ (Cholesky factorization).

**Batch Thompson Sampling:** Generate $q$ samples sequentially with fantasized observations (Kandasamy et al., 2018).

**Best For:** High-dimensional optimization ($d > 5$), complex posterior geometries, or when theoretical guarantees are required.

---

## V. Knowledge Gradient (KG)

### A. One-Step Lookahead

The knowledge gradient quantifies expected improvement in posterior mean maximum after one observation:

```math
\alpha_{\text{KG}}(\mathbf{x}) = \mathbb{E}\left[\max_{\mathbf{x}'} \mu_{n+1}(\mathbf{x}') - \max_{\mathbf{x}'} \mu_n(\mathbf{x}')\right]
```

where $\mu_{n+1}$ is the posterior mean after observing $f(\mathbf{x})$.

**Properties:**
- Myopically optimal for one-step lookahead
- Accounts for value of information
- Computationally expensive (nested optimization)

**Best For:** Finite-horizon optimization with known budget.

---

## VI. Acquisition Function Selection Guide

| Scenario | Recommended | Alternative | Avoid |
|----------|-------------|-------------|-------|
| Early exploration ($n < 50$) | qUCB ($\beta=5$) | Thompson | qEI, qPI |
| Mid-stage ($50 \leq n < 80$) | qUCB ($\beta=3$) | qEI | qPI |
| Late refinement ($n \geq 80$) | qEI or qUCB ($\beta=2$) | KG | Thompson |
| Parallel experiments | qUCB, qEI | Thompson | Analytic |
| High-dimensional ($d > 5$) | Thompson | qUCB | qEI |
| Unknown noise | qUCB (robust) | Thompson | qEI |
| Risk-averse | qPI | qUCB ($\beta=2$) | qEI |

---

## VII. Implementation Example

```python
from optimization.acquisition import (
    suggest_next_experiments_analytic,
    suggest_next_experiments_mc,
)

# Monte Carlo acquisition (recommended)
suggestions = suggest_next_experiments_mc(
    model=model,
    likelihood=likelihood,
    train_y=train_y,
    bounds=bounds,
    q=4,                    # Batch size
    acq_functions=["qUCB"], # Function type
    beta=5.0,               # Exploration parameter
    seed=42,                # Reproducibility
)

# Analytic acquisition (single suggestion)
suggestions = suggest_next_experiments_analytic(
    model=model,
    likelihood=likelihood,
    train_y=train_y,
    bounds=bounds,
    beta=5.0,
)
```

---

## VIII. Optimization Algorithm

Acquisition function maximization employs multi-start L-BFGS-B:

**Algorithm:**
1. Sample $N_{\text{raw}}$ initial points via Sobol sequence
2. Evaluate acquisition at all points
3. Select top $N_{\text{restart}}$ points as starting locations
4. Run L-BFGS-B from each starting point
5. Return best solution across all restarts

**Typical Configuration:**
- Raw samples: $N_{\text{raw}} = 900$ (30² grid)
- Restarts: $N_{\text{restart}} = 20$
- Convergence tolerance: $10^{-6}$

This ensures global exploration of acquisition landscape.

---

## IX. References

1. Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian process optimization in the bandit setting: No regret and experimental design. *ICML*, 1015-1022.

2. Russo, D., & Van Roy, B. (2014). Learning to optimize via posterior sampling. *Mathematics of Operations Research*, 39(4), 1221-1243.

3. Wilson, J., Hutter, F., & Deisenroth, M. (2018). Maximizing acquisition functions for Bayesian optimization. *NeurIPS*, 9884-9895.

4. Kandasamy, K., Krishnamurthy, A., Schneider, J., & Póczos, B. (2018). Parallelised Bayesian optimisation via Thompson sampling. *AISTATS*, 133-142.

5. Balandat, M., et al. (2020). BoTorch: A framework for efficient Monte-Carlo Bayesian optimization. *NeurIPS*, 33, 21524-21538.

---

**Last Updated:** October 2025  
**Compatible with:** BoTorch 0.9+, GPyTorch 1.11+
