import sys, os
import this
from tkinter import Y
import numpy as np
import torch
import gpytorch
import wandb
from model import GridGP
from datasetmaker import dataset
from acq_funcs import EI, PI, cust_acq, thompson
from plotter import vis_pred, vis_acq

###### SWEEPS ########
config_defaults = {
    "epochs": 3000,
    "kernel": "rbf",
    "lr": 0.005,
    "lscale_1": 1.0,
    "lscale_2": 1.0,
    "lscale_3": None,
    "lscale_4": None,
    "dim": 2,
    "noise": 0.2
}

wandb.init(config=config_defaults)
config = wandb.config

def kernel_func(config_kernel, num_params):
    """
    kernel_func(config_kernel, num_params) returns kernel function with 
    dimensions specified by [num_params]. 
    """
    if config_kernel == "rbf":
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_params))

def make_model(train_x, train_y, num_params, config):
    """
    make_model(train_x, train_y, num_params, config) returns likelihood and model
    with lengthscale, noise, kernel function specified by sweeps. 
    """
    kernel = kernel_func(config.kernel, num_params)
    noise = config.noise * torch.ones(len(train_x))
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise)
    model = GridGP(train_x, train_y, likelihood, kernel)
    
    if config.dim == 2:
        lscale = [config.lscale_1, config.lscale_2]
    elif config.dim == 4:
        lscale = [config.lscale_1, config.lscale_2, config.lscale_3, config.lscale_4]
    model.covar_module.base_kernel.lengthscale = torch.tensor(lscale)
    return likelihood, model

def train(train_x, train_y, num_params, config):
    likelihood, model = make_model(train_x, train_y, num_params, config)
    training_iter = config.epochs

    # Place both the model and likelihood in training mode
    model.train(), likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)

        # backpropagate error
        wandb.define_metric("train loss", summary="min")
        loss = -mll(output, train_y)
        wandb.log({"train loss": loss.item()})
        loss.backward()

        if i % 100 == 0: 
            print('Iter %d/%d - Loss: %.3f  lengthscale1: %s   noise: %s' % (
                    i+1, training_iter, loss.item(), 
                    model.covar_module.base_kernel.lengthscale.detach().numpy(),
                    model.likelihood.noise.detach().numpy()
                    )) 
        optimizer.step()
    return likelihood, model

def eval_mod(likelihood, model, test_x):
    """ 
    eval_mod(likelihood, model, test_x) evaluates GP model. 
    """
    model.eval(), likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        obs = likelihood(model(test_x), noise=(torch.ones(len(test_x))*5))
    return obs

def get_bounds(n): return [n for i in range(config.dim)]

def unravel_acq(acq_func, obs, bounds, train_y, nshape):
    """ 
    unravel_acq(acq_func, obs, bounds, train_y, nshape) is a helper function 
    for acq. 
    """
    acq = acq_func(obs, bounds, train_y).detach().numpy().reshape(nshape).T
    return np.unravel_index(acq.argmax(), acq.shape)

def acq(obs, train_y, bounds):
    """ 
    acq(obs, train_y, bounds) evaluates acquisition functions on current 
    predictions (observations) and outputs suggested points for exploration on manifold. 
    """
    transpose = lambda tensor: tensor.detach().numpy().reshape(nshape).T
    nshape = tuple(bounds)

    pi = unravel_acq(PI, obs, bounds, train_y, nshape) # prob of improvement
    ei = unravel_acq(EI, obs, bounds, train_y, nshape) # expected improvement
    ca = unravel_acq(cust_acq, obs, bounds, train_y, nshape) # custom acq
    th = unravel_acq(thompson, obs, bounds, train_y, nshape) # thompson acq

    lower, upper = obs.confidence_region()
    upper_surf, lower_surf = transpose(upper), transpose(lower)
    ucb = np.unravel_index(upper_surf.argmax(), upper_surf.shape)

    pred_var = obs.variance.view(nshape).detach().numpy().T
    pred_labels = obs.mean.view(nshape)
    max_var = np.unravel_index(pred_var.argmax(), pred_var.shape)
    acqs = {"PI":pi, "EI":ei, "CA":ca, "TH":th, "UCB":ucb, "Max_var":max_var}

    return pred_labels, upper_surf, lower_surf, acqs

def pred_to_csv(acqs, pred_labels, col_mean, col_sd, test_grid):
    """
    pred_to_csv(acqs, pred_labels, col_mean, col_sd, test_grid) outputs suggested
    inputs and their respective predicted outputs to csv. 
    """    
    x_raw = lambda acq: test_grid[acq[1]]*col_sd + col_mean # undo standardization
    dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/predictions/"
    file = open(dir + "preds.csv", "w", encoding="utf-8")
    
    file.write("Energy density \t Time (ms)\n")
    for lab, pred in acqs.items():
        file.write(lab + ": " + str(x_raw(pred).tolist()[0]) + "\t" + str(x_raw(pred).tolist()[1]) + "\n")
    
    file.write("\nFigure of merit\n")
    for lab, pred in acqs.items():
        file.write(lab + ": " + str(pred_labels[pred].item()) + "\n")

def main():
    column_mean, column_sd, train_x, train_y, num_params, test_grid, test_x = dataset()
    likelihood, model = train(train_x, train_y, num_params, config)
    obs = eval_mod(likelihood, model, test_x)
    bounds = get_bounds(n=30)
    vis_pred(config.noise, column_mean, column_sd, train_x, train_y, test_grid, obs, tuple(bounds))
    pred_labels, upper_surf, lower_surf, acqs = acq(obs, train_y, bounds)
    pred_to_csv(acqs, pred_labels, column_mean, column_sd, test_grid)
    vis_acq(config.noise, column_mean, column_sd, train_x, train_y, test_grid, pred_labels, upper_surf, lower_surf, acqs)
    
if __name__ == "__main__":
    main()
