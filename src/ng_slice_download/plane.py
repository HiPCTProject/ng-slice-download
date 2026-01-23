from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ng_slice_download.cuboid import Cuboid


@dataclass(kw_only=True)
class Plane:
    """
    A single plane.
    """

    # A point on the plane
    point: tuple[float, float, float]
    # Perpendicular vector at the above point
    direction: tuple[float, float, float] | None = None
    quarternion: tuple[float, float, float, float] | None = None
    chunks: tuple[int, int] = (128, 128)

    @property
    def rotation(self) -> Rotation:
        if self.direction is not None:
            R, _ = Rotation.align_vectors((0, 0, 1), self.direction)
            return R
        else:
            return Rotation.from_quat(self.quarternion)

    def plane_coords_to_world(self, x: float, y: float) -> tuple[float, float, float]:
        world_coord = np.vstack((x, y, np.zeros_like(x)))
        # Rotate about common point
        world_coord = self.rotation.apply(world_coord.T)
        # Translate to common point
        world_coord += np.array(self.point)
        return world_coord[:, 0], world_coord[:, 1], world_coord[:, 2]

    def tile_coords(self, tile_idx: tuple[int, int]):
        """
        Given the tile index, return coordinates of all the pixels in that tile.
        """
        x, y = np.meshgrid(
            np.arange(self.chunks[0]) + tile_idx[0] * self.chunks[0],
            np.arange(self.chunks[1]) + tile_idx[1] * self.chunks[1],
            indexing="ij",
        )
        return x.ravel(), y.ravel()

    def tile_corners(
        self, tile_idx: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
        """
        All corners of a given chunk.
        """
        return (
            (tile_idx[0] * self.chunks[0], tile_idx[1] * self.chunks[1]),
            ((tile_idx[0] + 1) * self.chunks[0], tile_idx[1] * self.chunks[1]),
            (tile_idx[0] * self.chunks[0], (tile_idx[1] + 1) * self.chunks[1]),
            ((tile_idx[0] + 1) * self.chunks[0], (tile_idx[1] + 1) * self.chunks[1]),
        )

    def get_nspiral(self, cuboid: Cuboid) -> tuple[int, list[tuple[int, int]]]:
        """
        Get the maximum number n_spiral needed to fully tile a given cuboid.
        """
        n_spiral = 0
        tiles_in_bounds = []
        while True:
            # Check if this spiral of tiles is all outside the data volume or not
            any_inside = False
            # Loop through every tile in a spiral.
            # For each tile, check if the corners are outside the data bounds
            for tile_idx in spiral_coords(n_spiral):
                corners = self.tile_corners(tile_idx)
                corners_world = [self.plane_coords_to_world(*c) for c in corners]
                if any(cuboid.contains(c) for c in corners_world):
                    tiles_in_bounds.append(tile_idx)
                    any_inside = True

            # If it's all outside, we have finished spiralling
            # Add one because if corners are all outside, there's still the possibility that one of the
            # edges of the square clips the edge of the volume
            if not any_inside:
                return n_spiral + 1, tiles_in_bounds
            n_spiral = n_spiral + 1


def spiral_coords(n: int) -> list[tuple[int, int]]:
    """
    Return a spiral of coordinates.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    elif n == 0:
        return [(0, 0)]
    else:
        # Use list(OrderedDict.fromkeys()) to get rid of duplicate corners
        return list(
            OrderedDict.fromkeys(
                [(i, n) for i in range(-n, n + 1)]
                + [(i, -n) for i in range(-n, n + 1)]
                + [(n, i) for i in range(-n, n + 1)]
                + [(-n, i) for i in range(-n, n + 1)]
            )
        )
