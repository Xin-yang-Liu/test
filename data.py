#define input parameters here
import numpy as np
from scipy.stats import multivariate_normal as multigauss

prior_dim = 1
prior_mean = 22000
prior_sigma = 2000
obs_values = np.array([12.84, 13.12, 12.13, 12.19, 12.67])/1000
N_prior_samples = 30000
N_resample = 10000

class data:
    '''
    mean, sigma, number of samples, array of sample values
    '''
    def __init__(self, dim, mean, sigma, size):
        self.dim = dim
        self.mean = mean
        self.sigma = sigma
        self.size = size
        self.sample = np.random.normal(loc=self.mean, scale=self.sigma, size=self.size)
    
    def pdf(self):
        '''
        probability density function
        '''
        return multigauss.pdf(self.sample, mean=self.mean, cov=self.sigma)

def initprior():
    prior = data(1, prior_mean, prior_sigma, N_prior_samples)
    return prior, obs_values, N_resample
