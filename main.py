import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import tensorstore as ts
from hoa_tools.dataset import change_metadata_directory, get_dataset
from tqdm import tqdm

from cuboid import Cuboid
from plane import Plane, spiral_coords

# Get this from the first column HiP-CT Google sheet
# https://docs.google.com/spreadsheets/d/1eMJgBPcTjCvnZpJ3AfpGvWf5yqNaQHC5FuyVQuSzl5M/edit?gid=0#gid=0
DATASET_NAME = "LADAF-2023-69_spine_overview_16.26um_bm18"
PRIVATE_META_PATH = "/Users/dstansby/software/hoa/private-hoa-metadata/metadata"


def main():
    change_metadata_directory(Path(PRIVATE_META_PATH), skip_invalid_meta=True)
    dataset = get_dataset(DATASET_NAME)
    print(dataset.name)
    print(dataset.data.gcs_url)
    print(dataset.data.shape)

    driver, _, path = dataset.data.gcs_url.split("://")
    bucket, path = path.split("/", maxsplit=1)
    input_image = ts.open(
        {
            "driver": driver,
            "kvstore": {"driver": "gcs", "bucket": bucket, "path": path + "0/"},
            "context": {"cache_pool": {"total_bytes_limit": 100_000_000}},
            "recheck_cached_data": False,
        }
    ).result()

    bounds = Cuboid(shape=dataset.data.shape)
    plane = Plane(
        point=(5386, 4590, 19955),
        quarternion=(
            0.5823236107826233,
            -0.37698644399642944,
            -0.3739404082298279,
            0.615588366985321,
        ),
    )
    max_nspiral = plane.get_nspiral(bounds)
    print(f"Need to spiral {max_nspiral} times")
    offset_x, offset_y = [max_nspiral * c for c in plane.chunks]
    # Set up empty Zarr array
    output_image_shape = [(2 * max_nspiral + 1) * c for c in plane.chunks]
    print(f"Creating output image, shape={output_image_shape}")
    output_image = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": "rotated.zarr"},
            "create": True,
            "delete_existing": True,
            "metadata": {
                "data_type": "float32",
                "shape": output_image_shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": plane.chunks},
                },
                "codecs": [],
                "fill_value": np.nan,
            },
        }
    ).result()
    print("Output image size:", output_image.size * 4 / 1e6, "MB")
    exit()
    for n_spiral in tqdm(list(range(max_nspiral + 1)), desc="Spirals"):
        spiral_idxs = spiral_coords(n_spiral)
        for spiral_idx in tqdm(spiral_idxs, desc="Tiles"):
            x, y = plane.tile_coords(spiral_idx)
            world_coords = plane.plane_coords_to_world(x, y)
            # Get bounding box of world coords
            slc = tuple(
                slice(max(0, math.floor(min(c))), min(s, math.ceil(max(c))))
                for s, c in zip(input_image.shape, world_coords, strict=True)
            )
            # Get numpy array within bounding box from Zarr array
            arr = input_image[slc].read().result()

            points = tuple(np.arange(s.start, s.stop) for s in slc)
            # Interpolate data on plane coordinates
            tile_image = scipy.interpolate.interpn(
                points=points,
                values=arr,
                xi=np.vstack(world_coords).T,
                bounds_error=False,
            ).reshape(plane.chunks)

            output_slc = (
                slice(
                    plane.chunks[0] * spiral_idx[0] + offset_x,
                    plane.chunks[0] * (spiral_idx[0] + 1) + offset_x,
                ),
                slice(
                    plane.chunks[1] * spiral_idx[1] + offset_y,
                    plane.chunks[1] * (spiral_idx[1] + 1) + offset_y,
                ),
            )
            # rint(f"Writing tile to {output_slc}")
            output_image[output_slc].write(tile_image.astype(np.float32)).result()

        # if n_spiral >= 1:
        #    break


if __name__ == "__main__":
    main()
