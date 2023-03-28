import os
import math
import sys

import numpy as np
from rdp_accountant import compute_rdp, get_privacy_spent

def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma
    
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps
    

def epsilon(N, batch_size, noise_multiplier, epochs, delta=1e-5):
    """Calculates epsilon for stochastic gradient descent.

    Args:
        N (int): Total numbers of examples
        batch_size (int): Batch size
        noise_multiplier (float): Noise multiplier for DP-SGD
        epochs (float): number of epochs (may be fractional)
        delta (float): Target delta

    Returns:
        float: epsilon

    Example::
        >>> epsilon(10000, 256, 0.3, 100, 1e-5)
    """
    q = batch_size / N
    steps = int(math.ceil(epochs * N / batch_size))
    optimal_order = _ternary_search(lambda order: _apply_dp_sgd_analysis(q, noise_multiplier, steps, [order], delta), 1, 512, 0.1)
    return _apply_dp_sgd_analysis(q, noise_multiplier, steps, [optimal_order], delta)


def _apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Calculates epsilon for stochastic gradient descent.

    Args:
        q (float): Sampling probability, generally batch_size / number_of_samples
        sigma (float): Noise multiplier
        steps (float): Number of steps mechanism is applied
        orders (list(float)): Orders to try for finding optimal epsilon
        delta (float): Target delta

    Returns:
        float: epsilon

    Example::
        >>> epsilon(10000, 256, 0.3, 100, 1e-5)
    """
    rdp = compute_rdp(q, sigma, steps, orders)
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    return eps

def _ternary_search(f, left, right, precision):
    """Performs a search over a closed domain [left, right] for the value which minimizes f."""
    while True:
        if abs(right - left) < precision:
            return (left + right) / 2

        left_third = left + (right - left) / 3
        right_third = right - (right - left) / 3

        if f(left_third) < f(right_third):
            right = right_third
        else:
            left = left_third
