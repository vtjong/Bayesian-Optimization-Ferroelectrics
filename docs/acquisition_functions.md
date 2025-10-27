# Acquisition Functions for Bayesian Optimization

Acquisition functions guide the selection of next experiments by balancing exploration vs exploitation.

## Analytic Acquisition Functions

### Expected Improvement (EI)

$$
\alpha_{\text{EI}}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f(\mathbf{x}^+), 0)]
$$

**Best for**: Exploitation-focused search when near optimum

### Probability of Improvement (PI)

$$
\alpha_{\text{PI}}(\mathbf{x}) = P(f(\mathbf{x}) > f(\mathbf{x}^+))
$$

**Best for**: Conservative optimization with high confidence needs

### Upper Confidence Bound (UCB)

$$
\alpha_{\text{UCB}}(\mathbf{x}) = \mu(\mathbf{x}) + \beta \sigma(\mathbf{x})
$$

**Best for**: Tunable exploration (via $\beta$) in early stages

## Monte Carlo Acquisition Functions

### qExpectedImprovement (qEI)
Batch version of EI using Monte Carlo sampling to suggest multiple experiments jointly.

**When to use**: Need to run multiple experiments in parallel

### qProbabilityOfImprovement (qPI)
Batch version of PI for conservative batch selection.

### qUpperConfidenceBound (qUCB)
Batch UCB with tunable exploration parameter.

**Recommended**: Use $\beta = 2-5$ for good exploration-exploitation balance

## Thompson Sampling

Sample functions from GP posterior, optimize sampled function.

**Best for**: Balanced exploration-exploitation with probabilistic guarantees

Implementation: See `src/optimization/thompson_sampler.py`

## Knowledge Gradient (KG)

$$
\alpha_{\text{KG}}(\mathbf{x}) = \mathbb{E}[\max_{\mathbf{x}'} \mu_{n+1}(\mathbf{x}') - \max_{\mathbf{x}'} \mu_n(\mathbf{x}')]
$$

**Best for**: Finite-horizon optimization when budget is limited

## Usage Example

```python
from optimization.acquisition import (
    suggest_next_experiments_analytic,
    suggest_next_experiments_mc,
)

# Single suggestion (analytic)
suggestions = suggest_next_experiments_analytic(
    model=model,
    likelihood=likelihood,
    train_y=train_y,
    bounds=bounds,
    beta=5.0,
)

# Batch suggestions (MC)
suggestions = suggest_next_experiments_mc(
    model=model,
    likelihood=likelihood,
    train_y=train_y,
    bounds=bounds,
    q=4,  # Number of parallel experiments
    acq_functions=["qEI", "qUCB"],
)
```

## References

- Shahriari et al. (2016). *Practical Bayesian Optimization of Machine Learning Algorithms*
- BoTorch Documentation: https://botorch.org/

