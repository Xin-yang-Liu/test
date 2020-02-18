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
#np.random.seed(1)
#================================= input ======================================#

dim_parameter = 2
dim_data = 1
prior_mean = np.array([0.032, 0.013])
prior_sigma = np.array([[0.001, 0],[0, 0.00015]])
obs_values = np.array([9.84, 10.12, 9.13, 9.19, 9.67])/1000
N_prior_samples = 50000
N_resample = 50000

#============================= initialization =================================#

prior = data.gaussiandata(dim_parameter, N_prior_samples, prior_mean, prior_sigma)
prior.sample = prior.gen_sample()
likelihood, sim = data.data(dim_data, prior.size), data.data(dim_data, prior.size)
posterior = data.data(dim_parameter, N_resample)


sim.sample = forward(prior.sample)


for i in range(prior.size):
    likelihood.pdfvalue[i] = multigauss.pdf(obs_values, 
                                mean=sim.sample[i], cov=1e-6).prod()


#normalization = likelihood.pdfvalue.sum()/prior.size
posterior.pdfvalue = likelihood.pdfvalue*prior.pdf()#/normalization


posterior.sample = resample(dim_parameter, prior.sample, posterior.pdfvalue, N_resample)
sim.sample = forward(posterior.sample)
mean_predict = np.mean(sim.sample)
var_predict = prior.size/(prior.size -1) * np.var(sim.sample)
print(posterior.sample[:,0].mean(), posterior.sample[:,1].mean(), mean_predict, var_predict)


fig, ax = plt.subplots(1, 3)
ax[0].hist(prior.sample[:,0],20)
ax[0].set_title("Prior")
ax[1].hist(posterior.sample[:,0],20)
ax[1].set_title("Posterior")
ax[2].hist(sim.sample,50)
ax[2].set_title("Predict")
ax[2].plot(obs_values, N_resample/15*np.ones(5), 'k.')
ax[2].plot(0.00959, N_resample/15*np.ones(1), 'r.')
ax[2].plot([mean_predict, mean_predict], [N_resample/15,0], 'g--')
fig.tight_layout()
plt.show()
