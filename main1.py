import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as multigauss
import data
from forward import forward
from resample import resample
################################################################################

dim = 2
prior_mean = np.array([0.022,0.015])
prior_sigma = np.array([[0.002, 0],[0, 0.0015]])
obs_values = np.array([12.84, 13.12, 12.13, 12.19, 12.67])/1000
N_prior_samples = 3000
N_resample = 1000


prior = data.gaussiandata(dim, N_prior_samples, prior_mean, prior_sigma)
prior.sample = prior.gen_sample()
likelihood, sim = data.data(prior.size), data.data(prior.size)
posterior = data.data(N_resample)

sim.sample = forward(prior.sample)


for i in range(prior.size):
    likelihood.pdfvalue[i] = multigauss.pdf(obs_values, 
                                mean=sim.sample[i], cov=1e-6).sum()


#normalization = likelihood.pdfvalue.sum()/prior.size
posterior.pdfvalue = likelihood.pdfvalue*prior.pdf()#/normalization


posterior.sample = resample(dim, prior.sample, posterior.pdfvalue, N_resample)
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
