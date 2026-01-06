from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class RNG:
    seed: int | None = None

    def __post_init__(self) -> None:
        self._gen = np.random.default_rng(self.seed)

    @property
    def gen(self) -> np.random.Generator:
        return self._gen

    def normal(self, size, mean=0.0, std=1.0) -> np.ndarray:
        return self._gen.normal(loc=mean, scale=std, size=size)

    def standard_normal(self, size) -> np.ndarray:
        return self._gen.standard_normal(size=size)

    def uniform(self, size) -> np.ndarray:
        return self._gen.uniform(size=size)
