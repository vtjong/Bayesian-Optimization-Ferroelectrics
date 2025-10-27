# Supplementary Information: Data Preprocessing and Feature Engineering

**Technical Documentation for GP Input Preparation**

---

## I. Preprocessing Motivation

Gaussian processes employ distance-based covariance kernels that are inherently scale-sensitive. Proper preprocessing is critical for:

1. **Numerical stability:** Condition number $\kappa(\mathbf{K}) < 10^{10}$
2. **Fair feature comparison:** Equal prior importance across dimensions
3. **Accurate ARD learning:** Unbiased lengthscale estimation
4. **Optimization convergence:** Well-conditioned gradients

---

## II. Data Cleaning

### A. Missing Value Removal

**Rationale:** Gaussian processes require complete observations $(\mathbf{x}_i, y_i)$.

**Issues with Missing Data:**
- Cannot compute $k(\mathbf{x}_i, \mathbf{x}_j)$ if $\mathbf{x}_i$ or $\mathbf{x}_j$ incomplete
- Marginal likelihood undefined: $p(\mathbf{y} \mid \mathbf{X})$ requires full $\mathbf{y}$ vector
- Covariance matrix inversion destabilized

**Implementation:**
```python
# Remove rows with any NaN values
data_clean = data.dropna()
```

**Alternative Approaches:**
- Imputation (mean, median, k-NN) — introduces bias
- Multiple imputation — computationally expensive
- GP with missing data handling — advanced topic (Tresp, 2000)

**Recommendation:** Simple removal preferred for small datasets where missing data fraction $< 10\%$.

---

### B. Outlier Filtering

**Physical Outliers:** Zero figure of merit indicates:
- Failed crystallization (no ferroelectric phase formation)
- Measurement equipment malfunction
- Sample preparation error

These are not valid observations of the objective function $f(\mathbf{x})$.

**Statistical Outliers:** Data points beyond $\mu \pm 3\sigma$ may indicate:
- Experimental anomalies
- Measurement noise
- Genuine extreme behavior (verify before removal)

**Implementation:**
```python
# Remove zero FOM (failed experiments)
data_valid = data[data['FOM'] > 0]

# Optional: Z-score filtering
z_scores = (data['FOM'] - data['FOM'].mean()) / data['FOM'].std()
data_filtered = data[abs(z_scores) < 3]
```

**Caution:** Conservative filtering recommended — do not remove potentially valuable observations without physical justification.

---

## III. Feature Scaling

### A. Scale Dependence Problem

**Unscaled Distance Metric:**

Without normalization, Euclidean distance is dominated by large-scale features:

```math
\lVert \mathbf{x} - \mathbf{x}' \rVert^2 = \sum_{d=1}^D (x_d - x_d')^2
```

**Example:** Ferroelectric processing parameters
- Pulse time: $t \in [0.5, 5.0]$ ms → range = 4.5
- Energy density: $E \in [2.73, 15.44]$ J/cm² → range = 12.71

Without scaling, energy density contributes $\approx 8\times$ more to distance than pulse time, biasing ARD lengthscale learning.

**Consequence:** Lengthscales $\{\ell_d\}$ become confounded with input scales, preventing accurate feature importance assessment.

---

### B. Min-Max Normalization

We apply min-max scaling to map each feature to $[0, 1]$:

```math
\tilde{x}_d = \frac{x_d - \min(x_d)}{\max(x_d) - \min(x_d)}
```

**Properties:**
- Preserves relationships: $x_d^{(i)} < x_d^{(j)} \Leftrightarrow \tilde{x}_d^{(i)} < \tilde{x}_d^{(j)}$
- Bounded range: $\tilde{x}_d \in [0, 1]$
- Outlier sensitive: Single extreme value affects all points

**Advantages for GP:**
- Uniform prior belief across dimensions
- ARD lengthscales directly interpretable as relevance
- Improves conditioning of $\mathbf{K}$ (all entries $\in [0, \sigma_f^2]$)
- Acquisition function optimization better behaved

**Implementation:**
```python
from preprocessing.transforms import TorchMinMaxScaler

scaler = TorchMinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)

# Inverse transform for interpretation
train_x_original = scaler.inverse_transform(train_x_scaled)
```

---

### C. Alternative Scaling Methods

**Standardization (Z-score):**

```math
\tilde{x}_d = \frac{x_d - \mu_d}{\sigma_d}
```

**Advantages:**
- Robust to outliers (mean/std less sensitive than min/max)
- Unbounded range (can handle extrapolation)

**Disadvantages:**
- Output not bounded → harder to set kernel hyperparameter priors
- Requires assumption of approximately Gaussian marginals

**Recommendation:** Min-max scaling preferred for small datasets ($n < 200$) with known input bounds.

---

## IV. Output Handling

### A. Output Scaling Decision

We preserve the figure of merit in **original physical units** (unscaled).

**Rationale:**

1. **Interpretability:** Domain experts reason in FOM units
2. **Physical meaning:** Optimization target has direct scientific interpretation  
3. **GP flexibility:** Outputscale parameter $\sigma_f^2$ automatically adapts to output magnitude
4. **Acquisition functions:** EI, PI, UCB have meaningful units (expected FOM improvement)

**GP Adaptation:**

The GP learns output scale via $\sigma_f^2$ in:

```math
k(\mathbf{x}, \mathbf{x}') = \sigma_f^2 k_{\text{base}}(\mathbf{x}, \mathbf{x}')
```

where $\sigma_f^2 \approx \text{Var}(y)$ is optimized during Type-II MLE.

---

### B. When to Scale Output

**Exception:** If output exhibits extreme dynamic range ($\max(y) / \min(y) > 1000$), standardize:

```math
\tilde{y} = \frac{y - \mu_y}{\sigma_y}
```

**Inversion:** Predictions must be rescaled:

```math
y_* = \mu_y + \sigma_y \tilde{y}_*
```

```math
\sigma_*^2(y) = \sigma_y^2 \sigma_*^2(\tilde{y})
```

**Not Required:** For FOM $\in [0, 5]$, scaling unnecessary.

---

## V. Bayesian Optimization Considerations

### A. Sequential Scaler Fitting

**Critical:** In production BO, scaler is fit **only on observed data** at each iteration:

```python
# CORRECT: Fit scaler on current observations
scaler.fit(observed_x)
candidate_scaled = scaler.transform(candidate_x)

# WRONG: Do not leak future information
scaler.fit(all_x_including_future)  # Violates sequential decision-making
```

**Information Leakage:** Fitting scaler on future experiments provides Oracle knowledge about parameter range, invalidating BO theoretical guarantees.

---

### B. Scale Parameter Stability

**Problem:** Adding single observation can shift min/max bounds, changing scale for all previous points.

**Solution:** After $n \geq 30$ observations, consider:

1. **Fixed scaling:** Use initial bounds, do not refit
2. **Padded bounds:** $[\min(x) - \epsilon, \max(x) + \epsilon]$ to allow slight extrapolation
3. **Quantile-based:** Use 5th and 95th percentiles instead of min/max

**Trade-off:** Stability vs. adaptation to expanded search region.

---

## VI. Exploratory Data Analysis

### A. Correlation Analysis

**Tool:** `plot_input_output_scatter_matrix()`

**Objectives:**
1. **Identify relationships:** Linear, nonlinear, interactions
2. **Detect outliers:** Visual inspection more effective than statistical tests
3. **Assess parameter coupling:** Off-diagonal structure in scatter matrix
4. **Guide kernel selection:** Sharp transitions → Matérn-$\frac{1}{2}$, smooth → Matérn-$\frac{5}{2}$/RBF

**Example Insights:**
- Strong time-FOM correlation → $\ell_{\text{time}}$ will be short
- Weak energy-FOM correlation → $\ell_{\text{energy}}$ may be long
- Nonlinear relationship → GP preferred over linear regression

---

### B. Input Distribution Analysis

**Check for:**
- **Clustering:** Oversampling in specific regions biases GP
- **Boundary effects:** Observations concentrated at edges prevent interior exploration
- **Grid artifacts:** Regular spacing may indicate non-adaptive sampling (suboptimal for BO)

**Ideal:** Space-filling design (Latin hypercube, Sobol) for initial samples, then BO-driven adaptive sampling.

---

## VII. Data Format Specifications

### A. Input Format

**Required Columns:**
1. `Time (ms)`: Pulse duration, continuous $\in [0.5, 5.0]$
2. `Energy density new cone (J/cm^2)`: Energy flux, continuous $\in [2.7, 15.4]$
3. `2 Qsw/(U+|D|) 1e6cycles`: Figure of merit, continuous $> 0$

**File Type:** Excel (`.xlsx`) or CSV (`.csv`)

**Missing Values:** Encoded as `NaN` or blank cells

---

### B. Output Format

**Predictions:** CSV with structure:
```
Acquisition_Function, Candidate_ID, Pulse_Time_ms, Energy_Density_J_cm2, Predicted_FOM
qUCB, 1, 1.37, 11.14, 2.88
qUCB, 2, 4.22, 14.06, 2.84
```

**Columns:**
- `Acquisition_Function`: String identifier (e.g., "qUCB", "qEI")
- `Candidate_ID`: Integer index $\in \{1, \ldots, q\}$
- `Pulse_Time_ms`: Unscaled time (physical units)
- `Energy_Density_J_cm2`: Unscaled energy (physical units)
- `Predicted_FOM`: GP posterior mean $\mu_*(\mathbf{x})$

---

## VIII. Preprocessing Pipeline Summary

**Standard Workflow:**

1. **Load raw data** from Excel
2. **Remove NaN** values (missing observations)
3. **Filter outliers** (zero FOM, statistical outliers)
4. **Extract features** ($\mathbf{X}$) and targets ($\mathbf{y}$)
5. **Fit scaler** on $\mathbf{X}$: `scaler.fit(X)`
6. **Transform inputs**: $\tilde{\mathbf{X}} = \text{scaler.transform}(\mathbf{X})$
7. **Train GP** on $(\tilde{\mathbf{X}}, \mathbf{y})$
8. **Inverse transform** predictions for interpretation

**PyTorch Integration:**

All preprocessing returns `torch.Tensor` objects for seamless GPU acceleration:

```python
train_x: torch.Tensor  # Shape: (n, d), dtype: float32
train_y: torch.Tensor  # Shape: (n,), dtype: float32
```

---

## IX. References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. Chapter 5: Model Selection and Adaptation.

2. Tresp, V. (2000). A Bayesian committee machine. *Neural Computation*, 12(11), 2719-2741.

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

4. McKay, M. D., Beckman, R. J., & Conover, W. J. (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. *Technometrics*, 21(2), 239-245.

---

**Last Updated:** October 2025  
**Compatible with:** PyTorch 2.0+, NumPy 1.24+, Pandas 2.0+
