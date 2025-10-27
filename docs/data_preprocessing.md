# Data Preprocessing for Gaussian Processes

## Why Preprocessing Matters

GPs use distance-based kernels that are scale-sensitive. Proper preprocessing is critical for:
- Numerical stability
- Fair feature comparison
- Accurate lengthscale learning

## Data Cleaning

### Removing NaN Values
- GPs require complete observations
- Missing targets destabilize covariance matrix inversions
- Cannot compute likelihood \( p(y|X) \) with undefined \( y \)

### Removing Zero Measurements
- Zero FOM indicates experimental failure
- Not a valid operating point
- Outliers corrupt lengthscale learning
- GPs are sensitive to outliers in small datasets

## Feature Scaling

### Why Scale?

Without scaling, features with large ranges dominate distance calculations:

**Example**: Energy (0-5 J/cm²) vs Time (0.1-2 ms)
- Energy range: 5
- Time range: 1.9
- Energy dominates distance metric unfairly

### MinMax Scaling to [0,1]

$$
\tilde{x}_d = \frac{x_d - \min(x_d)}{\max(x_d) - \min(x_d)}
$$

**Benefits**:
- All features start with equal importance
- Well-conditioned covariance matrices
- ARD lengthscales learn true relevance from data

**Implementation**:
```python
from preprocessing.transforms import TorchMinMaxScaler

scaler = TorchMinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)
```

### Input Ordering

We use `[Time, Energy]` ordering to match physical intuition:
- Time affects thermal diffusion (physical lengthscale ~1-5 ms)
- Energy affects peak temperature (physical lengthscale ~3-15 J/cm²)
- ARD will learn actual importance regardless

## Output Handling

### Why NOT Scale Output?

We keep figure of merit **unscaled** for:
- **Interpretability**: Domain experts understand raw FOM values
- **Physical meaning**: Predictions in original units
- **GP flexibility**: GP learns output scale via `outputscale` parameter

**Exception**: If output has extreme values (>1000x range), standardize:

$$
\tilde{y} = \frac{y - \mu_y}{\sigma_y}
$$

## For Bayesian Optimization

**Critical**: In production BO, only fit scaler on **observed** data

```python
# CORRECT: Only use observed data
scaler.fit(observed_x)
new_candidate_scaled = scaler.transform(new_candidate)

# WRONG: Don't leak information from future experiments
scaler.fit(all_x_including_future)  
```

This prevents data leakage where future experimental outcomes influence current decisions.

## Exploratory Data Analysis

Use `plot_input_output_scatter_matrix()` to:
1. Identify input-output correlations
2. Detect non-linear relationships
3. Spot outliers or clusters
4. Understand parameter coupling

This guides kernel selection (Matérn vs RBF, lengthscale priors, etc.)

