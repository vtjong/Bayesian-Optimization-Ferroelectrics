"""
Bayesian Optimization acquisition functions. 

Assumes gpytorch-style observations, with .mean and .variance methods
All return tensors, though may rely on numpy-based methods

bounds mimics botorch, looking for bounds for some acq functions

"""
import scipy
import numpy as np
from scipy.integrate import trapz
import torch
from scipy import stats


def like_imp(obs, bounds, train_y):
    mli = (obs.mean - obs.mean.max()) / np.sqrt(obs.variance)
    mli_cdf = stats.norm.cdf(mli.detach().numpy())
    return torch.Tensor(mli_cdf)

def EI(obs, bounds, train_y, eps=0.01):
    # variance needs to be detached to do deal with numpy arrays
    Z = ( obs.mean - train_y.max() - eps) / obs.variance.sqrt()
    Zn = Z.detach().numpy()
    ei = ( (obs.mean - train_y.max() - eps) * stats.norm.cdf(Zn) + 
            obs.variance.sqrt().detach().numpy() * stats.norm.pdf(Zn)
        ) 
    mask = obs.variance > 0 # zero if variance is "zero" (min variance defined in model is 0.1)
    ei = ei * mask
    return ei

def EI2(obs, bounds, train_y, eps=0.01):
    # old EI where variance was not sqrt'd
    # variance needs to be detached to do deal with numpy arrays
    Z = ( obs.mean - train_y.max() - eps) / obs.variance
    Zn = Z.detach().numpy()
    ei = ( (obs.mean - train_y.max() - eps) * stats.norm.cdf(Zn) + 
            obs.variance.detach().numpy() * stats.norm.pdf(Zn)
        ) 
    mask = obs.variance > 0 # zero if variance is "zero" (min variance defined in model is 0.1)
    ei = ei * mask
    return ei

def PI(obs, bounds, train_y, eps=0.01):
    Z = torch.div( obs.mean - train_y.max() - eps, torch.abs(obs.variance.sqrt()))
    Phi = stats.norm.cdf(Z.detach().numpy())
    return torch.Tensor(Phi)


def NU(obs, bounds, train_y):
    max_i = np.argmax(obs.mean).item()
    pm = obs.mean[max_i]
    sm = obs.variance[max_i]
    
    dc = np.where(obs.variance == sm)
    nu = (obs.mean - pm) / (obs.variance - sm)
    nu[dc] = 0
    return 1 / nu

def UCB(obs, bounds, train_y):
    var = obs.variance.detach().clone().numpy()
    var[var<0] = 0

    upper = obs.mean + np.sqrt(var) * 1.96
    return upper

def variance(obs, bounds, train_y):
    return obs.variance.clone()

def thompson(obs, bounds, train_y):
    return obs.sample()

def cust_acq(obs, bounds, train_y, base_fn=EI2):
    base_profile = base_fn(obs, bounds, train_y) # outputs tensor

    # scale to 1
    base_profile = base_profile / base_profile.max()

    # If everything is zero, return random with length of acq func? 

    # 0 out signal below 1% (things aren't always strictly 0)
    base_profile[np.where(base_profile < 0.01)] = 0

    # segment with 0 regions as dividers
    def find_nonzero_runs(a):
        # Credit to: https://stackoverflow.com/questions/31544129/extract-separate-non-zero-blocks-from-array
        # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
        isnonzero = np.concatenate(([0], (np.asarray(a) != 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(isnonzero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    ranges = find_nonzero_runs(base_profile)

    # calculate area under each curve and scale
    # expand range to ensure at least 3 points are available to integrate
    for start, stop in ranges:
        run = base_profile[start-1:stop+1]
        area = trapz(run) # assumes dx=1, which is likely wrong but we don't 
                          # care, since everything should be relative.          
        base_profile[start-1:stop+1] = base_profile[start-1:stop+1] * area

    return base_profile.clone()