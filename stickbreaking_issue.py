# coding=utf-8


import numpy as np
import pymc3
from pymc3.distributions.transforms import StickBreaking

stick = StickBreaking()
print(pymc3.__version__)
print(np.__version__)

def forward_py(x):
    x = x.T
    n = x.shape[0]
    lx = np.log(x)
    shift = np.sum(lx, 0, keepdims=True) / n
    y = lx[:-1] - shift
    return y.T

x = np.array([[0.31604495, 0.10538385, 0.00798379, 0.37937219, 0.19121521],
 [0.67614711, 0.14670955, 0.08485597, 0.07346115, 0.01882622]])
print(forward_py(x))
print(stick.forward_val(x))