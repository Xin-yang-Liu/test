import numpy as np

p = 0.012
L = 5
b = 0.15
h = 0.3

def forward(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 5/32*p*L**4/x[i,0]/b/x[i,1]**3*1e-6*1e-2
    return y