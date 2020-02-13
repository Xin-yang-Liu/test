import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as multigauss
import data
from forward import forward
from resample import resample
################################################################################


np.random.seed(1)
prior_dim = 1
prior_mean = 22000
prior_sigma = 2000
obs_values = np.array([13.1])/1000
N_prior_samples = 30000
N_resample = 10000


prior = data.gaussiandata(prior_dim, N_prior_samples, prior_mean, prior_sigma)
prior.sample = np.random.normal(loc=prior.mean, scale=prior.sigma,size=prior.size)
likelihood, sim = data.data(1,prior.size), data.data(1,prior.size)
posterior = data.data(1,N_resample)

sim.sample = forward(prior.sample)


for i in range(prior.size):
    likelihood.pdfvalue[i] = multigauss.logpdf(obs_values, 
                                mean=sim.sample[i], cov=1e-6).sum()


#normalization = likelihood.pdfvalue.sum()/prior.size
posterior.pdfvalue = likelihood.pdfvalue + prior.logpdf()#/normalization


posterior.sample = resample(prior.sample, posterior.pdfvalue, size = N_resample)
sim.sample = forward(posterior.sample)
mean_predict = np.mean(sim.sample)
var_predict = prior.size/(prior.size -1) * np.var(sim.sample)
print(posterior.sample.mean(), prior.sample.mean())


fig, ax = plt.subplots(1, 3)
ax[0].hist(prior.sample,20)
ax[2].hist(sim.sample,20)
#ax[1].plot(obs_values)
ax[1].hist(posterior.sample,20)
fig.tight_layout()
plt.show()
