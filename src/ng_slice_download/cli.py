import math
from pathlib import Path

import click
import inquirer
import neuroglancer
import numpy as np
import scipy.interpolate
import tifffile
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from ng_slice_download.cuboid import Cuboid
from ng_slice_download.plane import Plane
from ng_slice_download.utils import (
    create_local_tensorstore_array,
    open_tensorstore_array,
    yes_no_gate,
)

PREVIEW_LEVEL = 4


@click.command()
@click.argument("neuroglancer-url", type=str, required=True)
@click.option(
    "--output-dir",
    type=Path,
    required=False,
    default=Path.cwd(),
    help="Directory to download image files to.",
)
@click.option(
    "--skip-lowres-check", is_flag=True, help="Skip the low resolution check."
)
@click.option(
    "--overwrite-check", is_flag=True, help="Overwrite existing preview files without prompting."
)
def main(neuroglancer_url: str, output_dir: Path, skip_lowres_check: bool, overwrite_check: bool):
    print("Welcome to ng-slice-downloader!")

    ng_state = neuroglancer.url_state.parse_url(neuroglancer_url)
    selected_layer = {layer.name: layer for layer in ng_state.layers}.get(
        ng_state.selectedLayer.layer
    )
    check_image_layer(selected_layer)

    print(f"Selected layer: {selected_layer.name}")
    check_no_transform(selected_layer)

    image_url = str(selected_layer.source[0].url)
    check_ome_zarr_or_n5(image_url)
    print(f"Layer URL: {image_url}")

    position = ng_state.position
    rotation_quat = ng_state.crossSectionOrientation

    max_ring = None
    if not skip_lowres_check:
        print()
        print("Creating small image to check view is as expected")
        tiles_in_bounds, min_tile_idx, chunks = save_image(
            gcs_url=image_url,
            downsample_level=PREVIEW_LEVEL,
            position=position,
            rotation_quat=rotation_quat,
            output_path=output_dir / f"ng_slice_check_{selected_layer.name}",
            overwrite=overwrite_check,
        )
        preview_tiff = (output_dir / f"ng_slice_check_{selected_layer.name}").with_suffix(".tiff")
        annotate_preview(
            tiff_path=preview_tiff,
            tiles_in_bounds=tiles_in_bounds,
            min_tile_idx=min_tile_idx,
            chunks=chunks,
        )
        print()
        print(
            "Please check the TIFF and annotated PNG. "
            "Ring numbers show tile distance from the centre (0 = centre tile)."
        )
        yes_no_gate("Continue with large image?", default=True)

        max_preview_ring = max(max(abs(i), abs(j)) for (i, j) in tiles_in_bounds)
        answers = inquirer.prompt([
            inquirer.Text(
                "max_ring",
                message=f"Maximum ring number to include (0-{max_preview_ring}, blank = all)",
                validate=lambda _, x: x == "" or (x.isdigit() and 0 <= int(x) <= max_preview_ring),
            )
        ])
        max_ring = int(answers["max_ring"]) if answers["max_ring"].strip() else None

    shapes = [
        get_output_shape(
            gcs_url=image_url,
            downsample_level=downsample_level,
            position=position,
            rotation_quat=rotation_quat,
            max_ring=max_ring,
        )
        for downsample_level in range(4)
    ]
    questions = [
        inquirer.List(
            "downsample_level",
            message="What resolution output image do you want?",
            choices=[str(s) for s in shapes],
        ),
    ]
    answers = inquirer.prompt(questions)
    downsample_level = [str(s) for s in shapes].index(answers["downsample_level"])

    save_image(
        gcs_url=image_url,
        downsample_level=downsample_level,
        position=position,
        rotation_quat=rotation_quat,
        output_path=output_dir / f"ng_slice_{selected_layer.name}",
        max_ring=max_ring,
    )


def check_image_layer(layer: neuroglancer.ManagedLayer) -> None:
    if not layer.type == "image":
        print()
        print(
            f"Selected layer '{layer.name}' (type: {layer.type}) is not an image layer 😢"
        )
        exit()


def check_no_transform(layer) -> None:
    if (
        layer.source[0].transform is not None
        and layer.source[0].transform.matrix is not None
    ):
        print()
        print(
            "Selected layer has a transform matrix, "
            "but ng-slice-downloader does not currently support layers with transforms 😢"
        )
        exit()


def check_ome_zarr_or_n5(gcs_url: str) -> None:
    if not (gcs_url.startswith("zarr://") or gcs_url.startswith("n5://")):
        print()
        print("ng-slice-downloader only supports OME-Zarr or N5 images 😢")
        exit()


def filter_tiles_by_ring(
    tiles: list[tuple[int, int]],
    max_ring: int,
    downsample_level: int,
) -> list[tuple[int, int]]:
    """Filter tiles to those within max_ring rings at the preview level.

    Ring N at PREVIEW_LEVEL covers the physical region corresponding to
    tile indices [-N*scale, (N+1)*scale) at downsample_level, where
    scale = 2^(PREVIEW_LEVEL - downsample_level).
    """
    scale = 2 ** (PREVIEW_LEVEL - downsample_level)
    lo = -max_ring * scale
    hi = (max_ring + 1) * scale - 1
    return [(i, j) for (i, j) in tiles if lo <= i <= hi and lo <= j <= hi]


def annotate_preview(
    tiff_path: Path,
    tiles_in_bounds: list[tuple[int, int]],
    min_tile_idx: list[int],
    chunks: tuple[int, int],
) -> None:
    """Save an annotated PNG alongside the preview TIFF with ring numbers on each tile."""
    arr = tifffile.imread(tiff_path).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    arr_u8 = (((arr - lo) / (hi - lo)) * 255).astype(np.uint8) if hi > lo else np.zeros_like(arr, dtype=np.uint8)

    img = Image.fromarray(arr_u8).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=min(chunks) // 4)

    for i, j in tiles_in_bounds:
        ring = max(abs(i), abs(j))
        x0 = (i - min_tile_idx[0]) * chunks[0]
        y0 = (j - min_tile_idx[1]) * chunks[1]
        x1, y1 = x0 + chunks[0] - 1, y0 + chunks[1] - 1
        cx, cy = x0 + chunks[0] // 2, y0 + chunks[1] // 2
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0))
        text = str(ring)
        bbox = draw.textbbox((cx, cy), text, font=font, anchor="mm")
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((cx, cy), text, font=font, fill=(255, 255, 0), anchor="mm")

    out = tiff_path.with_name(tiff_path.stem + "_annotated.png")
    img.save(out)
    print(f"Annotated preview saved to: {out}")


def get_output_shape(
    *,
    gcs_url: str,
    downsample_level: int,
    position: list[float],
    rotation_quat: list[float],
    max_ring: int | None = None,
):
    input_image = open_tensorstore_array(gcs_url, downsample_level=downsample_level)
    bounds = Cuboid(shape=input_image.shape)
    plane = Plane(
        point=[p / 2**downsample_level for p in position], quarternion=rotation_quat
    )
    _, tiles_in_bounds = plane.get_nspiral(bounds)
    if max_ring is not None:
        tiles_in_bounds = filter_tiles_by_ring(tiles_in_bounds, max_ring, downsample_level)
    if not tiles_in_bounds:
        return (0, 0)
    min_tile_idx = np.min(tiles_in_bounds, axis=0).tolist()
    max_tile_idx = np.max(tiles_in_bounds, axis=0).tolist()

    return tuple(
        int((ma - mi + 1) * c)
        for c, mi, ma in zip(plane.chunks, min_tile_idx, max_tile_idx, strict=True)
    )


def save_image(
    *,
    gcs_url: str,
    downsample_level: int,
    position: list[int],
    rotation_quat: list[float],
    output_path: Path,
    max_ring: int | None = None,
    overwrite: bool = False,
) -> tuple[list[tuple[int, int]], list[int], tuple[int, int]]:
    input_image = open_tensorstore_array(gcs_url, downsample_level=downsample_level)
    print(f"Original image shape: {input_image.shape}")
    bounds = Cuboid(shape=input_image.shape)
    plane = Plane(
        point=[p / 2**downsample_level for p in position], quarternion=rotation_quat
    )
    _, tiles_in_bounds = plane.get_nspiral(bounds)
    if max_ring is not None:
        tiles_in_bounds = filter_tiles_by_ring(tiles_in_bounds, max_ring, downsample_level)
    min_tile_idx = np.min(tiles_in_bounds, axis=0).tolist()
    max_tile_idx = np.max(tiles_in_bounds, axis=0).tolist()
    offset = tuple(-c * mi for c, mi in zip(plane.chunks, min_tile_idx, strict=True))

    output_image_shape = tuple(
        int((ma - mi + 1) * c)
        for c, mi, ma in zip(plane.chunks, min_tile_idx, max_tile_idx, strict=True)
    )

    output_image_path = output_path.with_suffix(".zarr")
    TIFF_path = output_path.with_suffix(".tiff")

    if TIFF_path.exists() and not overwrite:
        yes_no_gate(f"{TIFF_path} already exists. Overwrite?", default=False)

    print(f"Creating output image, shape={output_image_shape}")
    print(f"Writing results to Zarr array at {output_image_path}")
    print(f"TIFF image will be updated every 10 tiles at {TIFF_path}")

    output_image = create_local_tensorstore_array(
        path=output_image_path,
        shape=output_image_shape,
        tile_shape=plane.chunks,
        dtype=str(input_image.dtype.numpy_dtype),
        fill_value=input_image.fill_value.tolist()
        if input_image.fill_value is not None
        else 0,
    )

    for i, tile_idx in enumerate(tqdm(tiles_in_bounds, desc="Downloading tiles")):
        x, y = plane.tile_coords(tile_idx)
        world_coords = plane.plane_coords_to_world(x, y)
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
                plane.chunks[0] * tile_idx[0] + offset[0],
                plane.chunks[0] * (tile_idx[0] + 1) + offset[0],
            ),
            slice(
                plane.chunks[1] * tile_idx[1] + offset[1],
                plane.chunks[1] * (tile_idx[1] + 1) + offset[1],
            ),
        )

        # Interpolate data on plane coordinates
        tile_image = scipy.interpolate.interpn(
            points=arr_coords,
            values=arr.result(),
            xi=xi,
            bounds_error=False,
            fill_value=input_image.fill_value.tolist()
            if input_image.fill_value is not None
            else 0,
        ).reshape(plane.chunks)

        output_image[output_slc].write(
            tile_image.astype(input_image.dtype.numpy_dtype)
        ).result()

        if i % 10 == 0:
            arr = output_image[:].read().result()
            tifffile.imwrite(TIFF_path, arr.T)
            del arr

    print("Finished downloading tiles!")
    print("Image saved to:", output_image_path)
    print("Converting to TIFF...")
    arr = output_image[:].read().result()
    tifffile.imwrite(TIFF_path, arr.T)
    print("TIFF saved to:", TIFF_path)
    return tiles_in_bounds, min_tile_idx, plane.chunks
