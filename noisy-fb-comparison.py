#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from noisy_fb_dir_info import dir_info_fwd, dir_info_rev

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
