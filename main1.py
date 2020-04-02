import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as multigauss

import data
from forward import forward
from resample import resample_f

'''
regardless of the dimension of parameter of forward(parameter), the pdf of them
shall always be 1 dimensional array
'''
np.random.seed(2000)
#================================= input ======================================#

dim_parameter = 1
dim_data = 1
prior_mean = np.array([18])
prior_sigma = np.array([1.8])
obs_values = np.array([12.59])
N_prior_samples = 30000
N_resample = 30000

#============================= initialization =================================#

prior = data.gaussiandata(dim_parameter, N_prior_samples, prior_mean, prior_sigma)
prior.sample = prior.gen_unif_sample()
likelihood, sim = data.data(dim_data, prior.size), data.data(dim_data, prior.size)
posterior = data.data(dim_parameter, N_resample)


sim.sample = forward(prior.sample)


for i in range(prior.size):
    likelihood.pdfvalue[i] = multigauss.pdf(obs_values, 
                                mean=sim.sample[i], cov=1.259).prod()


#normalization = likelihood.pdfvalue.sum()/prior.size
posterior.pdfvalue = likelihood.pdfvalue*prior.pdf()#/normalization


posterior.sample = resample_f(dim_parameter, prior.sample, posterior.pdfvalue, N_resample)
sim.sample = forward(posterior.sample)
mean_predict = np.mean(sim.sample)
var_predict = prior.size/(prior.size -1) * np.var(sim.sample)
#print(posterior.sample[:,0].mean(), posterior.sample[:,1].mean(), mean_predict, var_predict)

x1 = np.linspace(16,24,50)
x2 = np.linspace(12,18,50)
fig, ax = plt.subplots(1, 2)
ax[0].hist(posterior.sample,50,density=1)
ax[0].set_title("E")
ax[0].plot([np.mean(posterior.sample), np.mean(posterior.sample)], [0,0.4], 'k--')
ax[0].plot(x1, multigauss.pdf(x1,mean=posterior.sample.mean(),cov=posterior.sample.var()),'y--')

ax[1].hist(sim.sample,50,density=1)
ax[1].set_title("Displacement")
ax[1].plot(12.59, 0.4, 'r.')
ax[1].plot([mean_predict, mean_predict], [0,0.4], 'k--')
ax[1].plot(x2, multigauss.pdf(x2,mean=mean_predict,cov=var_predict),'y--')
fig.tight_layout()
plt.show()
