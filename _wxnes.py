from __future__ import annotations

import math

import numpy as np
from typing import Optional
from logging import getLogger

from base_nes import BaseNES

class WXNES(BaseNES):
    """wNES
    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        seed:
            A seed number (optional).

        population_size:
            A population size (optional).

    """

    # Paper: https://dl.acm.org/doi/10.1145/1830483.1830557

    def __init__(
        self,
        mean: np.ndarray,
        sigma: float,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        eta:float=None,
        update_mean: bool=True,
        update_sigma:bool=True,
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        n_dim = len(mean)
        assert n_dim > 1, "The dimension of mean must be larger than 1"

        if population_size is None:
            population_size = 4 + math.floor(3 * math.log(n_dim))
        assert population_size > 0, "popsize must be non-zero positive value."

        w_hat = np.log(population_size / 2 + 1) - np.log(
            np.arange(1, population_size + 1)
        )
        w_hat[np.where(w_hat < 0)] = 0
        weights = w_hat / sum(w_hat) - (1.0 / population_size)

        ## instance variables
        #dim
        self._n_dim = n_dim
        #pop_size
        self._popsize = population_size
        #weights
        self._weights = weights
        # learning rate
        if eta is None:
            eta=(3 / 5) * (3 + math.log(n_dim)) / (n_dim * math.sqrt(n_dim))
        self._eta_mean = 1
        self._eta_sigma = eta
        # distribution parameter
        self._mean = mean
        self._A = (sigma)*np.eye(n_dim)

        self._g = 0
        self._rng = np.random.RandomState(seed)

        self.update_mean = update_mean
        self.update_sigma = update_sigma


    @property
    def dim(self) -> int:
        """A number of dimensions"""
        return self._n_dim

    @property
    def population_size(self) -> int:
        """A population size"""
        return self._popsize

    @property
    def generation(self) -> int:
        """Generation number which is monotonically incremented
        when multi-variate gaussian distribution is updated."""
        return self._g

    def reseed_rng(self, seed: int) -> None:
        self._rng.seed(seed)

    def _sample_solution(self) -> np.ndarray:
        # z = self._rng.randn(self._n_dim)  # ~ N(0, I)
        # x = self._mean + scipy.linalg.sqrtm(self._C).dot(z)  # ~ N(m, σ^2 B B^T)
        x = self._rng.multivariate_normal(mean=self._mean,cov= self._A.dot(self._A.T), size=self._popsize, check_valid="warn")
        # print(x)
        return x

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        x = self._sample_solution()
        return x

    def tell(self,solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""
        logger = getLogger(__name__)

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        
        self._g += 1
        solutions.sort(key=lambda s: s[1])

        # natural gradient estimation in local coordinate
        G_mu = np.sum(
            [self._weights[i] * np.linalg.inv(self._A.dot(self._A.T)).dot((s[0] - self._mean)) for i,s in enumerate(solutions)], axis=0
        )
        g_M = np.sum(
            [
                self._weights[i]
                * (np.linalg.inv(self._A).dot(np.outer(s[0] - self._mean, s[0] - self._mean)).dot(np.linalg.inv(self._A.T)).dot(self._A).dot(self._A.T)) + ((self._A).dot(self._A.T).dot(np.linalg.inv(self._A)).dot(np.outer(s[0] - self._mean, s[0] - self._mean)).dot(np.linalg.inv(self._A.T))) -2*self._A.dot(self._A.T)
                for i,s in enumerate(solutions)
            ],
            axis=0,
        )
        #normarize g_M 
        g_M /= np.linalg.norm(g_M)


        # parameter update
        if self.update_mean:
            self._mean = self._mean + self._eta_mean * G_mu
        if self.update_sigma:
            self._A = self._A.dot(_expm(((self._eta_sigma)/2.0)*g_M))

        return self._mean,G_mu,g_M

def _expm(mat: np.ndarray) -> np.ndarray:
    D, U = np.linalg.eigh(mat)
    expD = np.exp(D)
    return U @ np.diag(expD) @ U.T