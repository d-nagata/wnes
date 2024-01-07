from __future__ import annotations

import math

import numpy as np
import scipy

from typing import cast
from typing import Optional
from logging import getLogger


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32

class WXNES:
    """xNES stochastic optimizer class with ask-and-tell interface.

    Example:

        .. code::

           import numpy as np
           from cmaes import XNES

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = XNES(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
               solutions = []
               for _ in range(optimizer.population_size):
                   # Ask a parameter
                   x = optimizer.ask()
                   value = quadratic(x[0], x[1])
                   solutions.append((x, value))
                   print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:

        mean:
            Initial mean vector of multi-variate gaussian distributions.

        sigma:
            Initial standard deviation of covariance matrix.

        bounds:
            Lower and upper domain boundaries for each parameter (optional).

        n_max_resampling:
            A maximum number of resampling parameters (default: 100).
            If all sampled parameters are infeasible, the last sampled one
            will be clipped with lower and upper bounds.

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
        eta:float=None
    ):
        assert sigma > 0, "sigma must be non-zero positive value"

        assert np.all(
            np.abs(mean) < _MEAN_MAX
        ), f"Abs of all elements of mean vector must be less than {_MEAN_MAX}"

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

        self._n_dim = n_dim
        self._popsize = population_size

        # weights
        self._weights = weights

        # learning rate
        if eta is None:
            eta=(3 / 5) * (3 + math.log(n_dim)) / (n_dim * math.sqrt(n_dim))
        self._eta_mean = 1.0
        self._eta = eta

        # distribution parameter
        self._mean = mean
        self._A = (sigma)*np.eye(n_dim)

        self._g = 0
        self._rng = np.random.RandomState(seed)

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

    def ask(self) -> np.ndarray:
        """Sample a parameter"""
        x = self._sample_solution()
        return x
        
    def _sample_solution(self) -> np.ndarray:
        # z = self._rng.randn(self._n_dim)  # ~ N(0, I)
        # x = self._mean + scipy.linalg.sqrtm(self._C).dot(z)  # ~ N(m, σ^2 B B^T)
        x = self._rng.multivariate_normal(mean=self._mean,cov= self._A.dot(self._A.T), size=1, check_valid="warn")
        # print(x)
        return x[0]

    def tell(self,i, solutions: list[tuple[np.ndarray, float]]) -> None:
        """Tell evaluation values"""
        logger = getLogger(__name__)

        assert len(solutions) == self._popsize, "Must tell popsize-length solutions."
        for s in solutions:
            assert np.all(
                np.abs(s[0]) < _MEAN_MAX
            ), f"Abs of all param values must be less than {_MEAN_MAX} to avoid overflow errors."
        
        self._g += 1
        solutions.sort(key=lambda s: s[1])

        z_k = np.array(
            [
                np.linalg.inv(self._A.dot(self._A.T)).dot(s[0] - self._mean)
                for s in solutions
            ]
        )

        # natural gradient estimation in local coordinate
        G_mu = np.sum(
            [self._weights[i] * z_k[i, :] for i in range(self.population_size)], axis=0
        )
        g_M = np.sum(
            [
                self._weights[i]
                * (np.linalg.inv(self._A).dot(np.outer(s[0] - self._mean, s[0] - self._mean)).dot(np.linalg.inv(self._A.T)).dot(self._A).dot(self._A.T)) + ((self._A).dot(self._A.T).dot(np.linalg.inv(self._A)).dot(np.outer(s[0] - self._mean, s[0] - self._mean)).dot(np.linalg.inv(self._A.T))) -2*self._A.dot(self._A.T)
                for i,s in enumerate(solutions)
            ],
            axis=0,
        )
        updated_count = 0
        # print(self._A)
        # print(_expm(g_M))

        # parameter update
        self._mean = self._mean + self._eta_mean * G_mu
        self._A = self._A.dot(_expm(((self._eta)/2.0)*g_M))


        # difference = self._eta * G_C + self._eta**2  * g_C.dot(self._C).dot(g_C)
        # self._C = self._C + difference
        # print(f"post: {self._C}")
        # if not np.all(np.linalg.eigvals(self._C) > 0):
        #     print("not positive!!")
        return self._mean,G_mu, updated_count

def _expm(mat: np.ndarray) -> np.ndarray:
    D, U = np.linalg.eigh(mat)
    expD = np.exp(D)
    return U @ np.diag(expD) @ U.T