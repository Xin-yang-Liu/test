import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as multigauss
import data
from forward import forward
from resample import resample


prior, obs_values, N_resample = data.initprior()

likelihood = np.ones(prior.size)
sim_sample = forward(prior.sample)

#for observation at same point more than once
for j in range(prior.size):
    for i in range(len(obs_values)):
        likelihood[j] *= multigauss.pdf(obs_values[i],
                                    mean=sim_sample[j], cov=1e-6)


normalization = likelihood.sum()/prior.size
posterior = likelihood*multigauss.pdf(prior.sample, mean=prior.mean, cov=prior.sigma)/normalization


posterior_sample = resample(prior.sample, posterior, size = N_resample)
sim_resample = forward(posterior_sample)
mean_predict = np.mean(sim_resample)
var_predict = prior.size/(prior.size -1) * np.var(sim_resample)
print(posterior_sample.mean(), prior.sample.mean())


fig, ax = plt.subplots(1, 3)
ax[0].hist(prior.sample,20)
ax[1].hist(sim_resample,50)
#ax[1].plot(obs_values)
ax[2].hist(posterior_sample,20)
fig.tight_layout()
plt.show()
