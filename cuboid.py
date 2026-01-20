from dataclasses import dataclass


@dataclass(kw_only=True)
class Cuboid:
    shape: tuple[int, int, int]

    def contains(self, point: tuple[float, float, float]) -> bool:
        return (
            0 <= point[0] <= self.shape[0]
            and 0 <= point[1] <= self.shape[1]
            and 0 <= point[2] <= self.shape[2]
        )
