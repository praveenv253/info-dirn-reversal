#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def simulate_sk(num_iter=100, snr=1.0, rev_noise_ratio=0, theta0=None):
    sigma_n = 1
    sigma_theta = snr
    sigma_r = rev_noise_ratio * sigma_n

    # Choose a starting theta
    if theta0 is None:
        theta = np.random.normal(scale=snr)
    else:
        theta = theta0

    # Arrays in which to return the time series
    xs = np.empty(num_iter)
    theta_hats = np.empty(num_iter)
    errors = np.empty(num_iter)

    # Start the simulation

    # Do the first step outside the loop
    x = theta
    theta_hat = x + np.random.normal(scale=sigma_n)
    xs[0] = x
    theta_hats[0] = theta_hat
    errors[0] = sigma_n ** 2

    # Loop for the remaining steps
    for i in range(1, num_iter):
        # Compute noise in the reverse direction
        if sigma_r == 0:
            r = 0
        else:
            r = np.random.normal(scale=sigma_r)
        # Compute X, to be transmitted
        x = theta - (theta_hats[i - 1] + r)
        # Add noise
        y = x + np.random.normal(scale=sigma_n)
        # Compute theta_hat
        theta_hat = theta_hats[i - 1] + y / (i + 1)
        # Populate the arrays to be returned
        xs[i] = x
        theta_hats[i] = theta_hat
        # Theoretical estimate of error after the i'th iteration (2-sigma)
        errors[i] = 2 * np.sqrt(sigma_n**2 / (i + 1) + sigma_r**2 / i)

    # Return
    return xs, theta_hats, errors

if __name__ == '__main__':
    num_iter = 500
    xs, theta_hats, errors = simulate_sk(num_iter=500, snr=0.5,
                                         rev_noise_ratio=0.1, theta0=1)

    t = np.arange(num_iter)
    plt.plot(t, xs)
    plt.plot(t, xs[0] * np.ones(t.size), 'k--', linewidth=2)
    plt.errorbar(t, theta_hats, errors)
    plt.title('Evolution of $X_i$ and $\hat{\Theta}_i$\nError is 2-$\sigma$')
    plt.legend(('$X_i$', '$\Theta$', '$\hat{\Theta}_i$'))
    plt.xlabel('$i$')
    plt.ylabel('$X$ or $\hat{\Theta}$')
    plt.show()
