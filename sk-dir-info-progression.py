#!/usr/bin/env python

import numpy as np
from scipy import special as spl
import matplotlib.pyplot as plt

def gammalog(x):
    return spl.gammaln(x) / np.log(2)

if __name__ == '__main__':
    n_lim = 500 # Maximum number of iterations
    #snrs = np.logspace(-2, -1, num=3, base=2.0)
    snrs = [0.25, 0.5, 0.75]

    blues = plt.get_cmap('Blues')
    reds = plt.get_cmap('Reds')
    greens = plt.get_cmap('Greens')
    cnums = np.logspace(1.8, 2.5, len(snrs)).astype(int)

    for i in range(len(snrs)):
        snr = snrs[i]
        cnum = cnums[i]
        n = np.logspace(np.log2(1), np.log2(n_lim), 50, base=2).astype(int)

        fwd_directed_info = 0.5 * np.log(1 + n * snr)
        plt.semilogx(n, fwd_directed_info, color=blues(cnum))

        term_1 = (1 + 2 * snr - np.sqrt(1 + 4 * snr)) / (2 * snr)
        term_2 = (1 + 2 * snr + np.sqrt(1 + 4 * snr)) / (2 * snr)
        term_3 = (1 + 2 * snr * n - np.sqrt(1 + 4 * snr)) / (2 * snr)
        term_4 = (1 + 2 * snr * n + np.sqrt(1 + 4 * snr)) / (2 * snr)
        rev_directed_info = 0.5 * (gammalog(n) + gammalog(term_1)
                                   + gammalog(term_2) + gammalog(1/snr + n)
                                   - gammalog(1/snr + 1) - gammalog(term_3)
                                   - gammalog(term_4))
        plt.semilogx(n, rev_directed_info, color=reds(cnum))

    for i in range(len(snrs)):
        snr = snrs[i]
        cnum = cnums[i]
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
