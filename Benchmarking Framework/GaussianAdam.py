#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: esraaj
"""

import numpy as np
from scipy.stats import beta

class myGaussianAdam:
    """
    Impliments a Gaussian noise term added to the learning parameter.
    """
    def __init__(
            self,
            params : np.ndarray,
            beta1 : float = 0.9,
            beta2 : float = 0.999,
            gamma : float = 0.0001, # default is 0.001
            alpha : float = 0.02,
            eps : float = 1e-8,
            gaussian_mean : float = 0,
            gaussian_std : float = 15,
            **kwargs
            ):
        
        # Store Variables
        self.params = params.astype(float)
        self.hyperparams = beta1, beta2, gamma
        self.gamma = gamma # Made to allow this code to fit with the rest of the test-bed framework
        self.alpha = alpha
        self.eps = eps

        self.normal = list([
            gaussian_mean,
            gaussian_std
            ])
        
        self.m = np.zeros_like(self.params)
        self.v =np.zeros_like(self.params)
        self.t = 0
        
        self.noise_log = list([])
    
    def step(self, grad: np.ndarray) -> np.ndarray:
            
        """
        One Gaussian Adam update.
        """
        beta1, beta2, gamma = self.hyperparams
        self.t += 1
        
        noise = abs(np.random.normal(self.normal[0], self.normal[1], self.params.shape[0]))
        
        self.noise_log.append(noise)

        # Update biased moments
        self.m = (beta1 * self.m + (1.0 - beta1) * grad)
        self.v = (beta2 * self.v + (1.0 - beta2) * (grad * grad))

        self.normal[1] *= (1.0 - gamma)

        # Bias correction
        m_hat = self.m / (1.0 - beta1 ** self.t)
        v_hat = self.v / (1.0 - beta2 ** self.t)
        

        # Adam parameter update
        self.params -= (self.alpha * (1.0 + noise)) * m_hat / (np.sqrt(v_hat) + self.eps)

        return self.params

