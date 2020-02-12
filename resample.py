from bisect import bisect_left as find
from numpy import zeros
from numpy.random import uniform 

def resample(sample, p, size):
    '''
    return an array of values of a variable with the PDF of p
    
    sample: array of original values

    p: the probability density array at each point

    size: Number of resamples
    '''    
    orignal_samples = len(sample)
    interval = zeros(orignal_samples)
    interval[0] = p[0]

    #generate array of the sum of probability 
    for i in range(1,orignal_samples):
        interval[i] = interval[i-1] + p[i]

    random_resample = uniform(high=interval[orignal_samples-1], size = size)
    resample = zeros(size)

    for i in range(size):
        resample[i] = sample[find(interval, random_resample[i])]

    return resample