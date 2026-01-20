import numpy as np

from plane import Plane, spiral_coords


def test_plane():
    # A simple x-y plane
    p = Plane(point=(0, 0, 0), direction=(0, 0, 1))
    assert p.plane_coords_to_world(0, 0) == (0, 0, 0)

    # A offset x-y plane
    p = Plane(point=(1, 0, 0), direction=(0, 0, 1))
    assert p.plane_coords_to_world(0, 0) == (1, 0, 0)
    assert p.plane_coords_to_world(0, 1) == (1, 1, 0)

    # A rotated plane
    p = Plane(point=(0, 0, 0), direction=(1, 0, 0))
    assert p.plane_coords_to_world(0, 0) == (0, 0, 0)
    # np.testing.assert_almost_equal(p.plane_coords_to_world(1, 1), ((0, 1, 1),))

    # A rotated plane
    p = Plane(point=(0, 0, 0), direction=(1, 1, 1))
    # np.testing.assert_almost_equal(
    #    p.plane_coords_to_world(1, 0), [0.7886751, -0.2113249, 0.5773503]
    # )

    p = Plane(point=(0, 0, 0), direction=(0, 0, 1), chunks=(2, 2))
    np.testing.assert_equal(p.tile_coords((0, 0)), ([0, 0, 1, 1], [0, 1, 0, 1]))
    np.testing.assert_equal(p.tile_coords((1, 1)), ([2, 2, 3, 3], [2, 3, 2, 3]))


def test_spiral():
    assert len(spiral_coords(1)) == 8
    assert spiral_coords(1) == [
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, 1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (0, -1),
    ]
    assert len(spiral_coords(2)) == 5 + 5 + 3 + 3
