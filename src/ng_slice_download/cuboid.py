from dataclasses import dataclass

import numpy as np


@dataclass(kw_only=True)
class Cuboid:
    shape: tuple[int, int, int]

    def contains(self, point: tuple[float, float, float]) -> bool:
        return (
            np.any(0 <= point[0])
            and np.any(point[0] <= self.shape[0])
            and np.any(0 <= point[1])
            and np.any(point[1] <= self.shape[1])
            and np.any(0 <= point[2])
            and np.any(point[2] <= self.shape[2])
        )
