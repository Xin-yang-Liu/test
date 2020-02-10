import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left as find
import resample
from input import initial_value as input

'''
prior_mean = 22000
prior_sigma = 2000
obs_values = np.array([12.84, 13.12, 12.13, 12.19, 12.67])/1000
N_prior_samples = 1000000
N_resample = 500000
'''
prior, obs_values, N_resample = input()

def forward(E):
    return 5/32*0.012*5**4/E/0.15/0.3**3


def gaussin(x, sigma, mean):
    return np.exp(-0.5*(x-mean)*(x-mean)/sigma**2)/sigma/np.sqrt(2*np.pi)


likelihood = np.ones(prior.size)

E_sample = np.random.normal(loc=prior.mean, scale=prior.sigma, size=prior.size)

a = forward(E_sample)

for j in range(prior.size):
    for i in range(len(obs_values)):
        likelihood[j] *= gaussin(obs_values[i],
                                 sigma=1e-3, mean=a[j])


normalization = likelihood.sum()/prior.size
posterior = likelihood*gaussin(E_sample, sigma=prior.sigma, mean=prior.mean)/normalization


E_resample = resample(E_sample,posterior,size = N_resample)
V_resample = forward(E_resample)
np.mean(V_resample)

fig, ax = plt.subplots(1, 3)
ax[0].hist(E_sample,20)
ax[1].hist(V_resample,50)
#ax[1].plot(obs_values)
ax[2].hist(E_resample,20)
fig.tight_layout()
plt.show()
