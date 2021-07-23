"""
Setup a model and data
"""

import numpy as np

# set the true values of the model parameters for creating the data
beta1d = -3 # gradient of the line
beta0d = 550 # y-intercept of the line

# set the "predictor variable"/abscissa
M = 30
xmin = 20.
xmax = 80.
stepsize = (xmax-xmin)/M
x = np.arange(xmin, xmax, stepsize)

# define the model function
def straight_line(x, m, c):
    """
    A straight line model: y = m*x + c
    
    Args:
        x (list): a set of abscissa points at which the model is defined
        m (float): the gradient of the line
        c (float): the y-intercept of the line
    """
    
    return m * x + c

# create the data - the model plus Gaussian noise
sig = 50.0  # standard deviation of the noise
data = straight_line(x, beta1d, beta0d) + np.random.normal(scale=sig, size=M)

print(data)