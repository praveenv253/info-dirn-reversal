#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from noisy_fb_dir_info import dir_info_fwd, dir_info_rev

if __name__ == '__main__':
    n_lim = 100 # Maximum number of iterations
    snrs = np.array([0.25, 0.5, 1])
    sigma_n = np.ones(snrs.size)
    sigma_theta = snrs
    rev_noise_ratio = 0.35
    sigma_r = rev_noise_ratio * sigma_n

    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    greens = plt.get_cmap('Greens')
    cnums = np.logspace(2, 2.5, len(snrs)).astype(int)

    n_vals = np.logspace(np.log2(1), np.log2(n_lim), 10, base=2).astype(int)
    fwd_directed_info = np.empty((len(n_vals), len(snrs)))
    rev_directed_info = np.empty((len(n_vals), len(snrs)))

    for j in range(len(snrs)):
        snr = snrs[j]
        var_n = sigma_n[j] ** 2
        var_theta = sigma_theta[j] ** 2
        var_r = sigma_r[j] ** 2

        for i in range(len(n_vals)):
            n = n_vals[i]
            fwd_directed_info[i, j] = dir_info_fwd(var_theta, var_n, var_r, n)
            rev_directed_info[i, j] = dir_info_rev(var_theta, var_n, var_r, n)

    plt.figure()
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)
    for i in range(len(snrs)):
        cnum = cnums[i]
        plt.semilogx(n_vals, fwd_directed_info[:, i], color=blues(cnum),
                     linewidth=2)
    plt.semilogx(n_vals, rev_directed_info[:, -2], color=reds(cnums[-2]),
                 linewidth=2)

    plt.xlabel(r'Iterations ($n$)', fontsize=18)
    plt.ylabel(r'Directed information in bits', fontsize=18)
    ax.text(0.05, 0.95, '$\sigma_R^2/\sigma_N^2=%.2f$' % rev_noise_ratio,
            verticalalignment='top', horizontalalignment='left',
            transform=ax.transAxes, fontsize=20)
    #plt.legend((r'$\sigma_\theta^2/\sigma_N^2=0.25$, Tx to Rx',
    #            r'$\sigma_\theta^2/\sigma_N^2=0.5$, Tx to Rx',
    #            r'$\sigma_\theta^2/\sigma_N^2=1$, Tx to Rx',
    #            r'Dir. info. from Rx to Tx for all $\sigma_\theta^2$'),
    #           loc='upper left')
    plt.tight_layout()

    plt.show()
