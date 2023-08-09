# Thompson sampler acquisition 
# Adapted from: https://botorch.org/tutorials/thompson_sampling
from contextlib import ExitStack
import time
import numpy as np
import torch
import gpytorch.settings as gpts
from torch.quasirandom import SobolEngine
from botorch.generation import MaxPosteriorSampling

import matplotlib
import matplotlib.pyplot as plt

class ThompsonSampling():
    def __init__(self, model, likelihood, scaler, seed, dim=2):
        self.model = model
        self.likelihood = likelihood
        self.scaler = scaler
        self.seed = seed
        self.dim = dim

    def get_x_pred(self, acq_pt): 
        return np.round(self.scaler.inverse_transform(acq_pt),2)

    def get_y_pred_mean(self, acq_pt, q):
        y_pred_mean = self.likelihood(self.model(acq_pt)).mean.detach().numpy()
        return np.round(y_pred_mean.item(),2) if q==1 else np.round(y_pred_mean,2)

    def get_x_cands(self, n_pts):
        sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        X_init = sobol.draw(n=n_pts)
        return X_init

    # Thompson sampling (sampler: "cholesky", "ciq", "rff")
    def generate_batch(self, n_cands, batch_size, sampler):

        # Draw samples on a Sobol sequence
        X_cand = self.get_x_cands(n_cands)

        with ExitStack() as es:
            if sampler == "cholesky":
                es.enter_context(gpts.max_cholesky_size(float("inf")))
            elif sampler == "ciq":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(True))
                es.enter_context(
                    gpts.minres_tolerance(2e-3)
                )  # Controls accuracy and runtime
                es.enter_context(gpts.num_contour_quadrature(15))
            elif sampler == "lanczos":
                es.enter_context(
                    gpts.fast_computations(
                        covar_root_decomposition=True, log_prob=True, solves=True
                    )
                )
                es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(False))
            elif self.sampler == "rff":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

        with torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

        return X_next

    def run_optimization(self, n_cands, n_init, max_evals, batch_size, sampler="ciq"):
        X = self.get_x_cands(n_init)
        Y = torch.tensor([self.get_y_pred_mean(x.unsqueeze(0), 1) for x in X])
        print(f"{len(X)}) Best value: {Y.max().item():.2e}")

        while len(X) < max_evals:
            start = time.monotonic()
            X_next = self.generate_batch(n_cands, batch_size, sampler)
            end = time.monotonic()
            print(f"Generated batch in {end - start:.3f} seconds")
            Y_next = torch.tensor(
                [self.get_y_pred_mean(x.unsqueeze(0), 1) for x in X_next])

            # Append data
            X = torch.cat((X, X_next), dim=0)
            Y = torch.cat((Y, Y_next), dim=0)

            print(f"{len(X)}) Best value: {Y.max().item():.2e}")
        return X, Y
    
    def vis_thompson(self, optimum, n_cand, max_evals, Y_ciq, Y_lanczos):
        fig = plt.figure(figsize=(10, 8))
        matplotlib.rcParams.update({"font.size": 20})

        results = [
            (Y_ciq, f"CIQ-{n_cand}", "g", "*", 12, "-"),
            (Y_lanczos, f"Lanczos-{n_cand}", "m", "^", 9, "-"),
        ]

        # optimum = train_y.max()

        ax = fig.add_subplot(1, 1, 1)
        names = []
        for res, name, c, m, ms, ls in results:
            names.append(name)
            fx = res.cummax(dim=0)[0]
            t = 1 + np.arange(len(fx))
            plt.plot(t[0::2], fx[0::2], c=c, marker=m, linestyle=ls, markersize=ms)

        plt.plot([0, max_evals], [optimum, optimum], "k--", lw=3)
        plt.xlabel("Fig of merit value", fontsize=18)
        plt.xlabel("Number of evaluations", fontsize=18)
        plt.title("Thompson", fontsize=24)
        plt.xlim([0, max_evals])
        plt.ylim([0, 5])

        plt.grid(True)
        plt.tight_layout()
        plt.legend(
            names + ["Global optimal value"],
            loc="lower right",
            ncol=1,
            fontsize=18,
        )
        plt.show()
