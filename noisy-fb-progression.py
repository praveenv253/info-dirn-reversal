#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from noisy_fb_dir_info import dir_info_fwd, dir_info_rev

if __name__ == '__main__':
    n_lim = 1000 # Maximum number of iterations
    snrs = np.array([0.25, 0.5, 1])
    sigma_n = np.ones(snrs.size)
    sigma_r = sigma_n
    sigma_theta = snrs

    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    greens = plt.get_cmap('Greens')
    cnums = np.logspace(1.8, 2.5, len(snrs)).astype(int)

    n_vals = np.logspace(np.log2(1), np.log2(n_lim), 50, base=2).astype(int)
    fwd_directed_info = np.empty((len(n_vals), len(snrs)))
    rev_directed_info = np.empty((len(n_vals), len(snrs)))

    for i in range(len(n_vals)):
        n = n_vals[i]
        fwd_directed_info[i, :] = dir_info_fwd(sigma_theta**2, sigma_n**2,
                                               sigma_r**2, n)
        rev_directed_info[i, :] = dir_info_rev(sigma_theta**2, sigma_n**2,
                                               sigma_r**2, n)

    for i in range(len(snrs)):
        cnum = cnums[i]
        plt.semilogx(n_vals, fwd_directed_info[:, i], color=blues(cnum))
        plt.semilogx(n_vals, rev_directed_info[:, i], color=reds(cnum))

    for i in range(len(snrs)):
        snr = snrs[i]
        cnum = cnums[i]
        # TODO Need to adjust the convergence criterion
        plt.semilogx([100./snr, 100./snr], [0, 3], color=greens(cnum),
                     linestyle='--')

    plt.xlabel(r'Iterations ($n$)')
    plt.ylabel(r'Directed information in bits')
    plt.title('Comparison of directed information flows in the\nSchalkwijk and'
              ' Kailath scheme')
    plt.legend(('$SNR=0.25$, fwd', '$SNR=0.25$, rev',
                '$SNR=0.5$, fwd', '$SNR=0.5$, rev',
                '$SNR=1$, fwd', '$SNR=1$, rev'), loc='upper left')
    plt.tight_layout()

    plt.show()
