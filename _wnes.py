from __future__ import annotations

import math

import numpy as np
from typing import Optional
from logging import getLogger

from base_nes import BaseNES

class WNES(BaseNES):
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
        eta_update_rate:float,
        seed: Optional[int] = None,
        population_size: Optional[int] = None,
        eta:float=None,
        update_mean=True,
        update_sigma=True,
        eig_calc="mean"
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
        self._eta_mean = 1.0
        self._eta_sigma = eta
        # distribution parameter
        self._mean = mean
        self._C = (sigma**2)*np.eye(n_dim)

        self._g = 0
        self._rng = np.random.RandomState(seed)
        # eta uptdate
        self.eta_update_rate= eta_update_rate
        self.pre_eta_sigma = -1
        self.eta_history = [] #each data: (generation, (pre_eta, updated_eta))
        self.is_positive_definite = True
        self.eig_calc = eig_calc


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
        x = self._rng.multivariate_normal(mean=self._mean,cov= self._C, size=self._popsize, check_valid="warn")
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

        z_k = np.array(
            [
                np.linalg.inv(self._C).dot(s[0] - self._mean)
                for s in solutions
            ]
        )

        # natural gradient estimation in local coordinate
        G_mu = np.sum(
            [self._weights[i] * np.linalg.inv(self._C).dot((s[0] - self._mean)) for i,s in enumerate(solutions)], axis=0
        )
        G_C = np.sum(
            [
                self._weights[i]
                * ((np.outer(s[0] - self._mean, s[0] - self._mean) - self._C).dot(np.linalg.inv(self._C)) + (np.linalg.inv(self._C)).dot(np.outer(s[0] - self._mean, s[0] - self._mean) - self._C))
                for i,s in enumerate(solutions)
            ],
            axis=0,
        )
        g_C = np.sum(
            [
                self._weights[i]
                * (np.linalg.inv(self._C).dot(np.outer(s[0] - self._mean, s[0] - self._mean)- self._C).dot(np.linalg.inv(self._C)))
                for i,s in enumerate(solutions)
            ],
            axis=0,
        )

        # if not (np.allclose(self._C, self._C.T, atol=1e-8)):
        #     print("yes!!!!!!!!!!!")
        #     # print(self._C)
        #     # print(self._C.T)
        # if not (np.allclose(np.linalg.inv(self._C), (np.linalg.inv(self._C)).T, atol=1e-10)):
        #     print("no!!!!!!!")

        # for s in solutions:
        #     aaa = np.outer(s[0] - self._mean, s[0] - self._mean)- self._C
        #     if not (np.allclose(aaa, aaa.T, atol=1e-10)):
        #         print("hi!!!!")
            
        # while True:
        #     if not ((np.all(np.linalg.eigvals(np.eye(self._n_dim) + self._eta_sigma*g_C)>0))) or not ((np.allclose((np.eye(self._n_dim) + self._eta_sigma*g_C), (np.eye(self._n_dim) + self._eta_sigma*g_C).T, atol=1e-8))):
        #         eigs = np.linalg.eigvals(g_C)
        #         # print(eigs)
        #         # self._eta = -1/min(eigs)

        #         if not(np.allclose((np.eye(self._n_dim) + self._eta_sigma*g_C), (np.eye(self._n_dim) + self._eta_sigma*g_C).T, atol=1e-8)):
        #             logger.info("symmetric")
        #             if not((np.all(np.linalg.eigvals(np.eye(self._n_dim) + self._eta_sigma*g_C)>0))):
        #                 logger.info("positive eig")

        #         elif not((np.all(np.linalg.eigvals(np.eye(self._n_dim) + self._eta_sigma*g_C)>0))):
        #             logger.info("only positive eig")
        #         else:
        #             logger.info("other reason")
        #         self.pre_eta_sigma=self._eta_sigma
        #         self._eta_sigma*=0.1
        #         self.eta_history.append((self._g, (self.pre_eta_sigma, self._eta_sigma)))
        #         logger.info(f"#{self._g}: pre_eta: {self.pre_eta_sigma}, updated eta: {self._eta_sigma}")
        #         continue
        #     break

        #parameter update
        w,_ = np.linalg.eig(self._C)
        if self.eig_calc=="mean":
            eta_by_c = np.mean(w)
        elif self.eig_calc=="min":
            eta_by_c = np.min(w)
        else:
            raise ValueError("informal eig calc  setting")

        G_mu = guarantee_symmetric(G_mu)
        self._mean = self._mean + self._eta_mean * G_mu*(eta_by_c)
        G_C = guarantee_symmetric(G_C)
        g_C = guarantee_symmetric(g_C)

        difference = self._eta_sigma *eta_by_c* G_C + (self._eta_sigma*eta_by_c)**2  * g_C.dot(self._C).dot(g_C)
        difference =  guarantee_symmetric(difference)
        self._C = self._C + difference
        # print(f"post: {self._C}")
        if not np.all(np.linalg.eigvals(self._C) > 0):
            print("not positive")
            self.is_positive_definite = False
        return self._mean,G_mu, self._C, difference, g_C

def guarantee_symmetric(x):
    return (x+x.T)/2