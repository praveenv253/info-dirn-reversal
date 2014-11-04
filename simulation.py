#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

class AutoRegressor(object):
    r"""
    This class defines an autoregressor of the form
        $$ y_t = \sum_{i=1}^{p}{\alpha_i y_{t-i}}
                 + \sum_{k=1}^{n}{ \sum_{j=0}^{q}{\beta^{(k)}_j x_{t-j}} }
                 + \epsilon_t $$
    where \epsilon is zero-mean gaussian noise of a given noise variance.

    Parameters
    ----------
    coeffs : np.ndarray
        A one-dimensional numpy array of the coefficients \alpha_i.
    deps : np.ndarray
        A two-dimensional numpy array of dependency coefficients. Each row is a
        different dependency. Column i contains the dependency value at time
        t - i.
    noise_var : float
        Variance of \epsilon.
    init : float
        Initial value of the regressor, y_0.
    """

    def __init__(self, coeffs=np.empty(0), deps=np.empty((0, 0)), noise_var=1,
                 init=0):
        self.y = init
        self.coeffs = coeffs
        self.y_memory = np.zeros(coeffs.shape)
        self.deps = deps
        self.deps_memory = np.zeros(deps.shape)
        self.noise_std = np.sqrt(noise_var)

    def step(self, dep_values=np.empty(0)):
        # Find new value of y_t from y alone
        y_t = (np.dot(self.coeffs, self.y_memory)
               + np.random.normal(scale=self.noise_std))
        # Add in the effects of the dependencies
        if self.deps_memory.size:
            # First update the dependency memory with the new dependency values
            self.deps_memory[:, 1:] = self.deps_memory[:, :-1]
            self.deps_memory[:, 0] = dep_values
        y_t += np.sum(self.deps * self.deps_memory)
        # Update y's memory
        if self.y_memory.size:
            self.y_memory[1:] = self.y_memory[:-1]
            self.y_memory[0] = y_t
        self.y = y_t

def create_data(num_time_steps):
    tx = AutoRegressor(coeffs=np.array([0.5, 0.5]))
    rx = AutoRegressor(coeffs=np.array([0.5, ]), deps=np.array([[1, ]]))
    txdata = np.zeros(num_time_steps)
    rxdata = np.zeros(num_time_steps)
    for t in xrange(num_time_steps):
        txdata[t] = tx.y
        tx.step()
        rxdata[t] = rx.y
        rx.step(np.array([tx.y]))

    return txdata, rxdata

if __name__ == '__main__':
    txdata, rxdata = create_data(1000)
    plt.plot(txdata)
    plt.plot(rxdata)
    plt.legend(('Tx', 'Rx'))
    plt.show()

