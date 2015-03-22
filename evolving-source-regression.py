#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

def main():
    # Model parameters
    var_N = 1                    # Reference
    var_epsilon = 0.1
    alpha_1 = 0.9
    var_theta = var_epsilon / (1 - alpha_1**2)
    p = 10

    # Simulation parameters
    N = 50000                    # Ensemble size
    n = 100                      # Number of iterations

    # Set up matrices
    x = np.empty(N)
    y = np.empty((p, N))         # Ensemble index is the column index
    theta_hat = np.zeros(N)

    betas = []
    for i in range(1, n):
        if i == 1:
            # Initialize theta_1 as Normal(0, sigma_theta^2)
            theta = np.random.normal(scale=np.sqrt(var_theta), size=(N,))
        else:
            # Generate the new theta: theta^+ = alpha_1 * theta + epsilon
            epsilon = np.random.normal(scale=np.sqrt(var_epsilon), size=(N,))
            theta = alpha_1 * theta + epsilon

        # Create the transmission: x
        x = theta - theta_hat
        # Add noise
        noise = np.random.normal(scale=np.sqrt(var_N), size=(N,))
        y[1:, :] = y[:-1, :]             # Shift y down
        y[0, :] = x + noise + theta_hat  # Correct for theta_hat

        # Create the estimate: theta_hat
        # The estimate is a linear function of the observations `y'. So we fit
        # coefficients beta over the ensemble.
        num_coeffs = min(i, p)
        beta = lstsq(y[:num_coeffs, :].T, theta)[0]

        # Construct the new theta_hat
        theta_hat = np.dot(y[:num_coeffs, :].T, beta)

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
    beta = np.mean(betas[p:])
    # Now that we have a fully defined model for Theta_hat, we can try to see
    # how much the error is for new data.
    # We can also do the directed info analysis to compute the directed info
    # in the forward and backward directions.

if __name__ == '__main__':
    main()
