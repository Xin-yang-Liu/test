#define input parameters here
import numpy as np

prior_mean = 22000
prior_sigma = 2000
obs_values = np.array([12.84, 13.12, 12.13, 12.19, 12.67])/1000
N_prior_samples = 1000000
N_resample = 500000

class data:
    def __init__(self, mean, sigma, size):
        self.mean = mean
        self.sigma = sigma
        self.size = size

def initial_value():
    prior = data(prior_mean, prior_sigma, N_prior_samples)
    return prior, obs_values, N_resample
