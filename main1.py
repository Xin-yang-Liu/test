import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as multigauss

import data
from forward import forward
from resample import resample

'''
regardless of the dimension of parameter of forward(parameter), the pdf of them
shall always be 1 dimensional array
'''

#================================= input ======================================#

dim_parameter = 2
dim_data = 1
prior_mean = np.array([0.02, 0.013])
prior_sigma = np.array([[0.002, 0],[0, 0.0015]])
obs_values = np.array([12.84, 13.12, 12.13, 12.19, 12.67])/1000
N_prior_samples = 80000
N_resample = 80000

#============================= initialization =================================#

prior = data.gaussiandata(dim_parameter, N_prior_samples, prior_mean, prior_sigma)
prior.sample = prior.gen_sample()
likelihood, sim = data.data(dim_data, prior.size), data.data(dim_data, prior.size)
posterior = data.data(dim_parameter, N_resample)


sim.sample = forward(prior.sample, sim.sample)


for i in range(prior.size):
    likelihood.pdfvalue[i] = multigauss.pdf(obs_values, 
                                mean=sim.sample[i], cov=1e-6).prod()


#normalization = likelihood.pdfvalue.sum()/prior.size
posterior.pdfvalue = likelihood.pdfvalue*prior.pdf()#/normalization


posterior.sample = resample(dim_parameter, prior.sample, posterior.pdfvalue, N_resample)
sim.sample = forward(posterior.sample, sim.sample)
mean_predict = np.mean(sim.sample)
var_predict = prior.size/(prior.size -1) * np.var(sim.sample)
print(posterior.sample[:,0].mean(), posterior.sample[:,1].mean(), mean_predict)


fig, ax = plt.subplots(1, 3)
ax[0].hist(prior.sample[:,0],20)
ax[2].hist(sim.sample[:,0],20)
#ax[1].plot(obs_values)
ax[1].hist(posterior.sample[:,0],20)
fig.tight_layout()
plt.show()
