# Bayesian Optimization for Ferroelectric Materials

Gaussian Process (GP) regression with Bayesian Optimization for optimizing pulse annealing parameters in ferroelectric thin films.

---

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd Bayesian-Optimization-Ferroelectrics
./setup_venv.sh

# Activate environment
source venv/bin/activate

# Run training
cd src
python train_clean.py
```

### Requirements

- Python 3.8+
- PyTorch, GPyTorch, BoTorch
- See `requirements.txt` for full list

---

## Project Structure

```
├── config/
│   ├── training_config.yaml      # Main config (model, training, acquisition)
│   └── sweep.yaml                # WandB hyperparameter sweep config
├── src/
│   ├── train_clean.py            # Main training script (7-step workflow)
│   ├── trainer.py                # GP training via MLE
│   ├── evaluator.py              # Model evaluation metrics
│   ├── config_loader.py          # YAML config utilities
│   ├── models/
│   │   ├── factory.py            # GP model & kernel creation
│   │   └── gp.py                 # ExactGPModel definition
│   ├── preprocessing/
│   │   ├── loaders.py            # Data loading & EDA
│   │   └── transforms.py         # Scaling & tensor conversion
│   └── optimization/
│       ├── acquisition.py        # EI, PI, UCB, qEI, qPI, qUCB
│       └── thompson_sampler.py   # Thompson sampling implementation
├── data/                         # Experimental data (Excel files)
├── docs/                         # Technical documentation
└── predictions/                  # BO suggestions output
```

---

## Usage

### Configuration

Edit `config/training_config.yaml`:

```yaml
model:
  kernel: "matern"        # 'matern' or 'rbf'
  matern_nu: 0.5          # 0.5, 1.5, or 2.5
  noise_prior: 0.1

training:
  epochs: 3000
  learning_rate: 0.003

acquisition:
  mc_or_analytic: "mc"    # 'mc' or 'analytic'
  functions: ["qUCB"]
  num_suggestions: 4
```

### Training

```bash
cd src
python train_clean.py
```

**Workflow**:
1. Load config & data
2. Create GP model (Matérn/RBF kernel with ARD)
3. Train via MLE (optimize lengthscales, outputscale, mean)
4. Evaluate performance (RMSE, R², Spearman)
5. Suggest next experiments (acquisition functions)
6. Export suggestions to CSV

### Interactive Development

Use `src/training.ipynb` for experimentation and visualization.

### Hyperparameter Sweeps (WandB)

```bash
wandb sweep config/sweep.yaml
wandb agent <sweep-id>
```
---

## Data

**Inputs** (2D):
- Pulse time (ms): thermal diffusion control
- Energy density (J/cm²): peak temperature control

**Output**:
- Figure of merit: `2*Qsw/(U+|D|)` @ 1M cycles
- Higher = better switching efficiency

---

## Technical Documentation

For detailed technical information, see:

- **[Kernel Design](docs/kernel_design.md)**: GP formulation, RBF vs Matérn, ARD lengthscales, constraints
- **[Acquisition Functions](docs/acquisition_functions.md)**: EI, PI, UCB, qEI, Thompson Sampling
- **[Data Preprocessing](docs/data_preprocessing.md)**: Scaling, normalization, data cleaning

---

## Module Usage Examples

### Training

```python
from trainer import train_gp_model, save_model_checkpoint

model, likelihood, loss_history = train_gp_model(
    model=model,
    likelihood=likelihood,
    train_x=train_x,
    train_y=train_y,
    n_epochs=3000,
)
```

### Evaluation

```python
from evaluator import evaluate_model

y_pred, y_std, metrics = evaluate_model(
    model=model,
    likelihood=likelihood,
    test_x=test_x,
    test_y=test_y,
)
```

### Bayesian Optimization

```python
from optimization.acquisition import suggest_next_experiments_mc

suggestions = suggest_next_experiments_mc(
    model=model,
    likelihood=likelihood,
    train_y=train_y,
    bounds=bounds,
    q=4,  # Batch size
    acq_functions=["qEI", "qUCB"],
)
```

---

## References

- **GPyTorch**: https://gpytorch.ai/
- **BoTorch**: https://botorch.org/
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*
- Shahriari et al. (2016). *Practical Bayesian Optimization*
