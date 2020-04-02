import numpy as np

p = 0.012
L = 5
b = 0.15
h = 0.3

def forward(x):
    y = np.zeros(len(x))
    y = 5/32*p*L**4/x/b/h**3
    return y