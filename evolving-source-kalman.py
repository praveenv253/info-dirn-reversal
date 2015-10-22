#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

import scipy.io as io

def kalman_est(mu_prev, phi_prev, var_epsilon, var_N, alpha_1, z):
    mu_minus = alpha_1 * mu_prev
    phi_minus = alpha_1**2 * phi_prev + var_epsilon
    k = phi_minus / (phi_minus + var_N)
    mu = mu_minus + k * (z - mu_minus)
    phi = (1 - k) * phi_minus

    return [mu, phi]

def generate_data(var_N, var_epsilon, alpha_1, num_iter=100):
    r"""
    Generates new data for the S&K scheme with a source autoregression model
        $$\Theta_i = \alpha_1 \Theta_{i-1} + \epsilon_i$$
    Transmissions are
        $$X_i = \Theta_i - \widehat{\Theta_{i-1}}$$
    The received signal is
        $$Y_i = X_i + N_i$$
    with iid additive Gaussian noise, and the estimate is
        $$\widehat{\Theta_i} = \sum_{j=0}^{p-1} \beta_j Z_{i-j}$$
    where $Z_i$ is a statistic
        $$Z_i = Y_i + \widehat{\Theta_{i-1}}$$.
    """

    # Stationarity is an underlying assumption of the model, so var_theta is
    # completely defined by the autoregression parameters
    var_theta = var_epsilon / (1 - alpha_1**2)

    theta = np.empty(num_iter)
    x = np.empty(num_iter)
    y = np.empty(num_iter)
    z = np.empty(num_iter)
    theta_hat = np.zeros(num_iter)
    phi = 0
    for i in range(num_iter):
        if i == 0:
            # Initialize theta_1 as Normal(0, sigma_theta^2)
            #theta[i] = np.random.normal(scale=np.sqrt(var_theta))
            # Initialize theta_1 as sigma_theta
            theta[i] = np.sqrt(var_theta)
            # The first transmission is simply x = theta_1
            x[i] = theta[i]
        else:
            # Generate the new theta: theta_i = alpha_1 * theta_{i-1} + epsilon
            epsilon = np.random.normal(scale=np.sqrt(var_epsilon))
            theta[i] = alpha_1 * theta[i-1] + epsilon
            # Create the transmission: x = theta_i - theta_hat_{i-1}
            x[i] = theta[i] - theta_hat[i-1]

        # Add noise to get the received signal
        noise = np.random.normal(scale=np.sqrt(var_N))
        y[i] = x[i] + noise

        # Correct for theta_hat
        if i == 0:
            z[i] = y[i]
        else:
            z[i] = x[i] + theta_hat[i-1]

        # Compute the estimate using the kalman filter
        [theta_hat[i], phi] = kalman_est(theta_hat[i-1], phi, var_epsilon,
                                         var_N, alpha_1, z[i])

    return theta, x, y, theta_hat

def main():
    # Model parameters
    # There are a number of effects at play here. We require var_theta < var_N,
    # so that there is something to estimate. But we also require
    # var_epsilon < var_theta, so that theta is evolving slowly. The alpha_1
    # should automatically adjust to reflect this and should not be an issue.
    var_N = 1           # Reference noise level
    var_epsilon = 0.005  # Variance of the noise added to the autoregressor
    var_theta = 0.25     # Variance of theta
    alpha_1 = np.sqrt(1 - var_epsilon / var_theta)
    #alpha_1 = 0.9       # Memory term / "fading" of the autoregressor

    theta, x, y, theta_hat = generate_data(var_N, var_epsilon, alpha_1)
    plt.plot(theta)
    plt.plot(theta_hat)
    plt.show()

    num_trials = 1000
    num_iter = 1000
    xthetas = np.empty((2, num_iter, num_trials))
    for i in range(num_trials):
        if i % 100 == 99:
            print(i+1)
        params = generate_data(var_N, var_epsilon, alpha_1, num_iter)
        xthetas[0, :, i] = params[1] # X
        xthetas[1, :, i] = params[3] # Theta_hat
    io.savemat('xthetas', {'X': xthetas})

if __name__ == '__main__':
    main()
