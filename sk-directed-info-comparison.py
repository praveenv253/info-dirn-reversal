#!/usr/bin/env python

import numpy as np
from scipy import special as spl
import matplotlib.pyplot as plt

def gammalog(x):
    return spl.gammaln(x) / np.log(2)

if __name__ == '__main__':
    snr = np.logspace(-10, 10, num=20, base=2.0)
    n_vals = [10, 100, 1000]

    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    cnums = np.logspace(1.8, 2.5, len(n_vals)).astype(int)

    for i in range(len(n_vals)):
        n = n_vals[i]
        cnum = cnums[i]

        fwd_directed_info = 0.5 * np.log(1 + n * snr)
        plt.semilogx(snr, fwd_directed_info, color=blues(cnum))

        term_1 = (1 + 2 * snr - np.sqrt(1 + 4 * snr)) / (2 * snr)
        term_2 = (1 + 2 * snr + np.sqrt(1 + 4 * snr)) / (2 * snr)
        term_3 = (1 + 2 * snr * n - np.sqrt(1 + 4 * snr)) / (2 * snr)
        term_4 = (1 + 2 * snr * n + np.sqrt(1 + 4 * snr)) / (2 * snr)
        rev_directed_info = 0.5 * (gammalog(n) + gammalog(term_1)
                                   + gammalog(term_2) + gammalog(1/snr + n)
                                   - gammalog(1/snr + 1) - gammalog(term_3)
                                   - gammalog(term_4))
        plt.semilogx(snr, rev_directed_info, color=reds(cnum))

    #plt.legend((r'$I(X^n \rightarrow \hat{\Theta}^n)$',
    #            r'$I(0*\hat{\Theta}^{n-1} \rightarrow X^n)$'))

    plt.xlabel(r'$\sigma_\theta^2 / \sigma_n^2$')
    plt.ylabel(r'Directed information in bits')
    plt.title('Comparison of directed information flows in the\n'
              'Schalkwijk and Kailath scheme')
    plt.tight_layout()

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(('fwd, $n=10$', 'rev, $n=10$',
               'fwd, $n=100$', 'rev, $n=100$',
               'fwd, $n=1000$', 'rev, $n=1000$'),
               loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
