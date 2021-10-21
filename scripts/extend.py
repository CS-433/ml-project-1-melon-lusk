# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    n = x.shape[0]
    result = np.ones((n,1))
    x = x[:,:30]
    if degree == 0:
        return result
    else:
        for i in range(degree + 1):
            if i == 0:
                continue
            extension = np.power(x,i)
            result = np.c_[result,extension]
        return result

    
def extend(x,fs):
    result = x
    for function in fs:
        extension = function(x[:, :30])
        result = np.c_[result,extension]
    return result