#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: esraaj
"""

import numpy as np
from GaussianAdam import myGaussianAdam

_universal_alpha = 0.02

class SGD:
    def __init__(self, params, alpha: float = _universal_alpha, **kwargs):
        self.alpha = alpha
        self.params = params.astype(float)

    def step(self, grad):
        self.params -= self.alpha * grad
        return self.params


class Nesterov:
    def __init__(self, params, alpha: float = _universal_alpha, momentum: float = 0.9, **kwargs):
        self.alpha = alpha
        self.momentum = momentum

        self.params = params.astype(float)
        self.v = np.zeros_like(self.params)

    def step(self, grad):
        self.v = self.momentum * self.v + grad

        step = grad + self.momentum * self.v

        self.params -= self.alpha * step
        return self.params


class Adagrad:
    def __init__(self, params, alpha: float = 0.5, eps: float = 1e-8, **kwargs):
        self.alpha = alpha
        self.eps = eps
        self.params = params.astype(float)
        self.G = np.zeros_like(self.params)

    def step(self, grad):
        self.G += grad * grad
        self.params -= self.alpha * grad / (np.sqrt(self.G) + self.eps)
        return self.params


class RMSProp:
    def __init__(self, params, alpha: float =_universal_alpha, rho: float = 0.999, eps: float = 1e-8, **kwargs):
        self.alpha = alpha
        self.rho = rho
        self.eps = eps

        self.params = params.astype(float)
        self.v = np.zeros_like(self.params)

    def step(self, grad):
        self.v = self.rho * self.v + (1.0 - self.rho) * (grad * grad)
        self.params -= self.alpha * grad / (np.sqrt(self.v) + self.eps)
        return self.params


class Adam:
    def __init__(self, params, alpha: float = 0.02, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, **kwargs):
        self.alpha = alpha 
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.params = params.astype(float)

        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)
        self.t = 0

    def step(self, grad):
        self.t += 1

        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)

        # Bias Correction
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        self.params -= self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
        return self.params

# Initialise optimizer wrapper
def make_optimizer(name, params, **kwargs):
    name = name.lower()

    if name == 'sgd':
        return SGD(params, **kwargs)
    elif name in ('nesterov', 'nag'):
        return Nesterov(params, **kwargs)
    elif name == 'adagrad':
        return Adagrad(params, **kwargs)
    elif name == 'rmsprop':
        return RMSProp(params, **kwargs)
    elif name == 'adam':
        return Adam(params, **kwargs)
    elif name == 'mygaussianadam':
        return myGaussianAdam(params, **kwargs)
    else:
        raise NameError(f"Unknown Optimizer '{name}'")
