from abc import ABC
from abc import abstractmethod

class BaseNES(ABC):
    @abstractmethod
    def ask(self):
        pass

    @abstractmethod
    def tell(self):
        pass

    @abstractmethod
    def _sample_solution(self):
        pass
    