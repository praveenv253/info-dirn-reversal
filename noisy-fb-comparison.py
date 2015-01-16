#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

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
            sigma_11 = 2 * var_r + var_n / i**2
            sigma_12 = [-var_r, var_n/i - var_r/i]
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


if __name__ == '__main__':
    snr = np.logspace(-10, 6, num=20, base=2.0)
    sigma_n = np.ones(snr.size)
    sigma_r = sigma_n
    sigma_theta = snr
    n_vals = [10, 100, 1000]

    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    cnums = np.logspace(1.8, 2.5, len(n_vals)).astype(int)

    for j in range(len(n_vals)):
        n = n_vals[j]
        cnum = cnums[j]

        fwd_directed_info = dir_info_fwd(sigma_theta**2, sigma_n**2,
                                         sigma_r**2, n)
        plt.semilogx(snr, fwd_directed_info, color=blues(cnum))

        rev_directed_info = dir_info_rev(sigma_theta**2, sigma_n**2,
                                         sigma_r**2, n)
        plt.semilogx(snr, rev_directed_info, color=reds(cnum))

    #plt.legend((r'$I(X^n \rightarrow \hat{\Theta}^n)$',
    #            r'$I(0*\hat{\Theta}^{n-1} \rightarrow X^n)$'))

    plt.xlabel(r'$\sigma_\theta^2 / \sigma_n^2$')
    plt.ylabel(r'Directed information in bits')
    plt.title('Comparison of directed information flows in the\n'
              'Schalkwijk and Kailath scheme with noisy feedback')
    plt.tight_layout()

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(('fwd, $n=10$', 'rev, $n=10$',
               'fwd, $n=100$', 'rev, $n=100$',
               'fwd, $n=1000$', 'rev, $n=1000$'),
               loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
