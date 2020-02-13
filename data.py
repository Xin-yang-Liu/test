#define input parameters here
import numpy as np
from scipy.stats import multivariate_normal as multigauss


class data:
    '''
    number of samples, array of sample values
    '''
    def __init__(self, dim, size):
        self.dim = dim
        self.size = size
        self.sample = np.zeros(size)
        self.pdfvalue = np.zeros(size)
    

class gaussiandata(data):
    '''
    for Gaussian Distribution

    '''
    def __init__(self, dim, size, mean, sigma):
        super().__init__(dim, size)
        self.mean = mean
        self.sigma = sigma

    def pdf(self):
        '''
        probability density function
        '''
        self.pdfvalue = multigauss.pdf(self.sample, mean=self.mean, cov=self.sigma)
        return self.pdfvalue

    def logpdf(self):
        return multigauss.logpdf(self.sample, mean=self.mean, cov=self.sigma)


