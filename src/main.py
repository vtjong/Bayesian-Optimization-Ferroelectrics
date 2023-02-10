import torch
import gpytorch
import wandb
from model import GridGP
from datasetmaker import dataset

###### SWEEPS ########
config_defaults = {
    "epochs": 9000,
    "kernel": "rbf",
    "lr": 0.1,
    "lscale_1": 5,
    "lscale_2": 10,
    "lscale_3": 0.5,
    "lscale_4": 1,
    "noise": 3.0
}
wandb.init(config=config_defaults)
config = wandb.config

# Switch kernel
def kernel_func(config_kernel, num_params):
    if config_kernel == "rbf":
        return gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=num_params))

# Init GP model
def make_model(train_x, train_y, num_params, config):
    kernel = kernel_func(config.kernel, num_params)
    noises = config.noise * torch.ones(len(train_x))
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises)
    model = GridGP(train_x, train_y, likelihood, kernel)
    lscale = [config.lscale_1, config.lscale_2, config.lscale_3, config.lscale_4]
    model.covar_module.base_kernel.lengthscale = torch.tensor(lscale)
    return likelihood, model

# Training loop (long form, for inspection of results during training)
def train(train_x, train_y, num_params):
    likelihood, model = make_model(train_x, train_y, num_params, config)
    training_iter = config.epochs

    # Place both the model and likelihood in training mode
    model.train(), likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)

        # backpropogate error
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

def eval(likelihood, model, test_x):
    # make predictions (whether by long or short form)
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # also use mll from short form 
        obs = likelihood(model(test_x), noise=(torch.ones(len(test_x))*5))
    return obs

def main():
    train_x, train_y, num_params, test_x = dataset()
    likelihood, model = train(train_x, train_y, num_params)
    obs = eval(likelihood, model, test_x)
    

if __name__ == "__main__":
    main()
