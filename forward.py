import numpy as np

def forward(x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = 0.00000434/x[i,0]/x[i,1]
    return y