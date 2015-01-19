#!/usr/bin/env python

import numpy as np


def dir_info_fwd(var_theta, var_n, var_r, n):
    dir_info_1 = 0.5 * np.log2(var_theta / var_n + 1)
    dir_info_rest = 0.5 * np.log2( (1 + var_r / var_n)
                                   * (n*var_theta + var_n + var_r)
                                   / var_n / (var_theta + var_n + var_r) )
    return dir_info_1 + dir_info_rest


def dir_info_rev(var_theta, var_n, var_r, n):
    dir_info = 0

    for i in range(n+1):
        # We are leaving out the 2 * pi * e in the entropy calculations,
        # because it will get cancelled at the end anyway.
        if i == 0:
            hx_x = 0.5 * np.log2(var_theta)
            hx_x_theta = 0.5 * np.log2(var_theta)
        elif i == 1:
            hx_x = 0.5 * np.log2(var_n + var_r)
            hx_x_theta = 0.5 * np.log2(var_r)
        else:
            var_bar = var_n / (i - 1) + (i - 2) * var_r / (i - 1)**2
            hx_x = 0.5 * np.log2( (var_r**2 + 2*var_r*var_bar
                                   + var_n*var_r/i**2 + var_n*var_bar/i**2)
                                  / (var_bar + var_r) )
            # H(X_{i+1} | X^{i}, \hat\Theta^{i}):
            var_bar_i = var_n / i + (i - 1) * var_r / i**2
            det_sigma_22 = ( (var_r + var_bar) * (var_bar_i + var_theta)
                             - (var_r / i - (i-1) * var_bar / i) ** 2 )
            sigma_11 = 2 * var_r #+ var_n / i**2
            sigma_12 = [-var_r, -var_r/i] #var_n/i - var_r/i]
            sigma_21 = sigma_12
            sigma_22 = [ [var_bar_i + var_theta, (i-1)*var_bar/i - var_r/i],
                         [(i-1)*var_bar/i - var_r/i, var_r + var_bar] ]
            conditional_var = ( sigma_11
                                - ( sigma_12[0] * sigma_22[0][0] * sigma_21[0]
                                  + sigma_12[0] * sigma_22[0][1] * sigma_21[1]
                                  + sigma_12[1] * sigma_22[1][0] * sigma_21[0]
                                  + sigma_12[1] * sigma_22[1][1] * sigma_21[1])
                                / det_sigma_22
                              )
            hx_x_theta = 0.5 * np.log2(conditional_var)

        # Add mutual information for each `i' to get net directed information
        dir_info += hx_x - hx_x_theta

    return dir_info
