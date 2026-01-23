import math
from pathlib import Path

import numpy as np
import scipy.interpolate
from hoa_tools.dataset import change_metadata_directory, get_dataset
from tqdm import tqdm

from ng_slice_download.cuboid import Cuboid
from ng_slice_download.plane import Plane, spiral_coords
from ng_slice_download.utils import (
    create_local_tensorstore_array,
    open_tensorstore_array,
)

# Get this from the first column HiP-CT Google sheet
# https://docs.google.com/spreadsheets/d/1eMJgBPcTjCvnZpJ3AfpGvWf5yqNaQHC5FuyVQuSzl5M/edit?gid=0#gid=0
DATASET_NAME = "LADAF-2023-69_spine_overview_16.26um_bm18"
PRIVATE_META_PATH = "/Users/dstansby/software/hoa/private-hoa-metadata/metadata"


def main():
    change_metadata_directory(Path(PRIVATE_META_PATH), skip_invalid_meta=True)
    dataset = get_dataset(DATASET_NAME)
    print(dataset.name)
    print(dataset.data.shape)

    downsample_level = 1
    bin_factor = 2**downsample_level
    input_image = open_tensorstore_array(
        dataset.data.gcs_url, downsample_level=downsample_level
    )
    print(input_image.shape)
    bounds = Cuboid(shape=input_image.shape)

    # Define the plane of the output image
    plane = Plane(
        point=(5420 / bin_factor, 4602 / bin_factor, 19818 / bin_factor),
        quarternion=(
            0.6761807799339294,
            0.16871704161167145,
            0.14734917879104614,
            0.7018563747406006,
        ),
    )
    max_nspiral = plane.get_nspiral(bounds)
    print(f"Need to spiral {max_nspiral} times")
    # Offset of the output image from the centre of the spiral
    offset_x, offset_y = [max_nspiral * c for c in plane.chunks]
    # Set up empty Zarr array
    output_image_shape = tuple((2 * max_nspiral + 1) * c for c in plane.chunks)

    print(f"Creating output image, shape={output_image_shape}")
    output_image = create_local_tensorstore_array(
        path="rotated.zarr", shape=output_image_shape, tile_shape=plane.chunks
    )
    print("Output image size:", output_image.size * 4 / 1e6, "MB")
    print()

    for n_spiral in tqdm(list(range(max_nspiral + 1)), desc="Spirals"):
        tile_idxs = spiral_coords(n_spiral)
        for tile_idx in tqdm(tile_idxs, desc="Tiles"):
            x, y = plane.tile_coords(tile_idx)
            world_coords = plane.plane_coords_to_world(x, y)
            if not np.any(bounds.contains(world_coords)):
                continue

            # Get bounding box of world coords
            slc = tuple(
                slice(max(0, math.floor(min(c)) - 2), min(s, math.ceil(max(c)) + 2))
                for s, c in zip(input_image.shape, world_coords, strict=True)
            )
            # Get NumPy array within bounding box from Zarr array
            arr = input_image[slc].read()
            arr_coords = tuple(np.arange(s.start, s.stop) for s in slc)
            xi = np.vstack(world_coords).T
            output_slc = (
                slice(
                    plane.chunks[0] * tile_idx[0] + offset_x,
                    plane.chunks[0] * (tile_idx[0] + 1) + offset_x,
                ),
                slice(
                    plane.chunks[1] * tile_idx[1] + offset_y,
                    plane.chunks[1] * (tile_idx[1] + 1) + offset_y,
                ),
            )

            # Interpolate data on plane coordinates
            tile_image = scipy.interpolate.interpn(
                points=arr_coords,
                values=arr.result(),
                xi=xi,
                bounds_error=False,
                fill_value=np.nan,
            ).reshape(plane.chunks)

            output_image[output_slc].write(tile_image.astype(np.float32)).result()


if __name__ == "__main__":
    main()
