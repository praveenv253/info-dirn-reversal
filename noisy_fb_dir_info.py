#!/usr/bin/env python

import numpy as np


def dir_info_fwd(var_theta, var_n, var_r, n):
    dir_info = 0

    for i in range(1, n+1):
        if i == 1:
            dir_info += 0.5 * np.log2(1 + var_theta / var_n)
        if i == 2:
            var_theta_theta = (var_r + var_n
                               + var_theta * var_n / (var_theta + var_n))
            htheta_theta = 0.5 * np.log2(var_theta_theta)
            htheta_x_theta = 0.5 * np.log2(var_n)
            dir_info += htheta_theta - htheta_x_theta
        else:
            # Construct a matrix for htheta_theta
            var_uu = var_theta + var_r + var_n
            var_uv = var_theta * np.ones((i-1, 1))
            var_vv = np.empty((i-1, i-1))
            for _p in range(i-1):
                p = _p + 1
                for _q in range(i-1):
                    q = _q + 1
                    var_vv[_p, _q] = (var_theta + min(p, q) * var_n / (p * q)
                                      + min(p-1, q-1) * var_r / (p * q))
            var_vv_inv = np.linalg.inv(var_vv)
            var_theta_theta = var_uu - np.dot(var_uv.T,
                                              np.dot(var_vv_inv, var_uv))
            htheta_theta = 0.5 * np.log2(var_theta_theta)
            htheta_x_theta = 0.5 * np.log2(var_n)
            dir_info += htheta_theta - htheta_x_theta

    return dir_info


def dir_info_rev(var_theta, var_n, var_r, n):
    dir_info = 0

    for i in range(n):
        # We are leaving out the 2 * pi * e in the entropy calculations,
        # because it will get cancelled at the end anyway.
        if i == 0:
            hx_x = 0.5 * np.log2(var_theta)
            hx_x_theta = 0.5 * np.log2(var_theta)
        elif i == 1:
            hx_x = 0.5 * np.log2(var_n + var_r)
            hx_x_theta = 0.5 * np.log2(var_r)
        else:
            # Construct matrix for hx_x
            var_uu = 2 * var_r + var_n / i**2
            var_uv = np.zeros((i, 1))
            var_uv[-1] = - var_r
            var_vv = np.zeros((i, i))
            var_vv[0, 0] = var_theta
            for _p in range(1, i):
                p = _p + 1
                for _q in range(1, i):
                    q = _q + 1
                    var_vv[_p, _q] = (min(p-2, q-2)*var_r / ((p-1) * (q-1))
                                      + min(p-1, q-1)*var_n / ((p-1) * (q-1)))
                    if p == q:
                        var_vv[_p, _q] += var_r
                    elif p > q:
                        var_vv[_p, _q] += var_r / (p-1)
                    else:
                        var_vv[_p, _q] += var_r / (q-1)
            var_vv_inv = np.linalg.inv(var_vv)
            var_x_x = var_uu - np.dot(var_uv.T, np.dot(var_vv_inv, var_uv))
            hx_x = 0.5 * np.log2(var_x_x)
            hx_x_theta = 0.5 * np.log2(var_r)

        # Add mutual information for each `i' to get net directed information
        dir_info += hx_x - hx_x_theta

    return dir_info
