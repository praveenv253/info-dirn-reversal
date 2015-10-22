#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

import scipy.io as io

def compute_beta(var_N, var_epsilon, alpha_1, p):
    """
    Under an autoressive source model and an S&K feedback scheme for
    communicating observations (and corrections of the estimate):
    Compute the (first `p') regression coefficients of the i-th estimate,
    Theta_hat_i, in terms of the shifted observations (Y_i + Theta_hat_{i-1}).
    """

    # Stationarity is an underlying assumption of the model, so var_theta is
    # completely defined by the autoregression parameters
    var_theta = var_epsilon / (1 - alpha_1**2)

    # Simulation parameters
    N = 50000                    # Ensemble size
    n = 100                      # Number of iterations

    # Set up matrices
    x = np.empty(N)
    y = np.empty((p, N))         # Ensemble index is the column index
    theta_hat = np.zeros(N)

    betas = []
    #theta1s = []
    #theta1hats = []
    for i in range(1, n):
        if i == 1:
            # Initialize theta_1 as Normal(0, sigma_theta^2)
            theta = np.random.normal(scale=np.sqrt(var_theta), size=(N,))
        else:
            # Generate the new theta: theta_i = alpha_1 * theta_{i-1} + epsilon
            epsilon = np.random.normal(scale=np.sqrt(var_epsilon), size=(N,))
            theta = alpha_1 * theta + epsilon

        # Create the transmission: x = theta_i - theta_hat_{i-1}
        x = theta - theta_hat
        # Add noise
        noise = np.random.normal(scale=np.sqrt(var_N), size=(N,))
        y[1:, :] = y[:-1, :]             # Shift y down
        y[0, :] = x + noise + theta_hat  # Correct for theta_hat
                                         # Note: This is theta_hat_{i-1}

        # Create the estimate: theta_hat
        # The estimate is a linear function of the observations `y'. So we fit
        # coefficients beta over the ensemble.
        num_coeffs = min(i, p)
        beta = lstsq(y[:num_coeffs, :].T, theta)[0]

        # Construct the new theta_hat
        theta_hat = np.dot(y[:num_coeffs, :].T, beta)

        # Collect theta and theta_hat from one trial out of the ensemble
        # for plotting
        #theta1s.append(theta[1])
        #theta1hats.append(theta_hat[1])

        # Append the new coefficient to the set of betas
        betas.append(beta)

    # Plot the betas to see whether or not they change with `i'
    #plt.plot(betas[15])
    #plt.plot(betas[24])
    #plt.plot(betas[33])
    #plt.plot(betas[42])
    #plt.plot(betas[91])
    #plt.show()

    # The betas don't vary with `i', so we can average over time to get a
    # better beta.
    beta = np.mean(np.array(betas[p:]), axis=0)
    # Now that we have a fully defined model for Theta_hat, we can try to see
    # how much the error is for new data.
    # We can also do the directed info analysis to compute the directed info
    # in the forward and backward directions.

    # Plot theta and theta_hat from one trial out of the ensemble to visually
    # check that tracking is working.
    #plt.plot(theta1s)
    #plt.plot(theta1hats)
    #plt.show()

    return beta

def generate_data(var_N, var_epsilon, alpha_1, beta, num_iter=100):
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

    p = beta.size

    theta = np.empty(num_iter)
    x = np.empty(num_iter)
    y = np.empty(num_iter)
    z = np.empty(num_iter)
    theta_hat = np.zeros(num_iter)
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

        # Compute the estimate using beta
        theta_hat[i] = np.sum(beta[:min(i+1, p)] * z[max(i+1-p, 0):i+1][::-1])

    return theta, x, y, theta_hat

def main():
    # Model parameters
    # There are a number of effects at play here. We require var_theta < var_N,
    # so that there is something to estimate. But we also require
    # var_epsilon < var_theta, so that theta is evolving slowly. The alpha_1
    # should automatically adjust to reflect this and should not be an issue.
    var_N = 1           # Reference noise level
    var_epsilon = 0.10  # Variance of the noise added to the autoregressor
    var_theta = 0.5     # Variance of theta
    alpha_1 = np.sqrt(1 - var_epsilon / var_theta)
    #alpha_1 = 0.9       # Memory term / "fading" of the autoregressor

    p = 20              # Number of regression coefficients

    beta = compute_beta(var_N, var_epsilon, alpha_1, p)
    theta, x, y, theta_hat = generate_data(var_N, var_epsilon, alpha_1, beta)

    num_trials = 10
    num_iter = 1000
    xthetas = np.empty((2, num_iter, num_trials))
    for i in range(num_trials):
        params = generate_data(var_N, var_epsilon, alpha_1, beta, num_iter)
        xthetas[0, :, i] = params[1] # X
        xthetas[1, :, i] = params[3] # Theta_hat
    io.savemat('xthetas', {'X': xthetas})

    plt.semilogy(beta)
    plt.show()

    plt.plot(theta)
    plt.plot(theta_hat)
    plt.show()

if __name__ == '__main__':
    main()
