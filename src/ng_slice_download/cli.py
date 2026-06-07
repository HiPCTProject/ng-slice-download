import math
from pathlib import Path

import click
import inquirer
import neuroglancer
from neuroglancer.viewer_state import DataPanelLayout, LayerGroupViewer, StackLayout
import numpy as np
import scipy.interpolate
from scipy.spatial.transform import Rotation
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
@click.option(
    "--slab-thickness",
    type=int,
    default=1,
    show_default=True,
    help="Number of depth slices to download, centred on the view plane (1 = single plane).",
)
def main(neuroglancer_url: str, output_dir: Path, skip_lowres_check: bool, overwrite_check: bool, slab_thickness: int):
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

    position, rotation_quat = select_panel_state(ng_state)

    print_plane_info(position, rotation_quat)

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
            slab_thickness=slab_thickness,
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
        slab_thickness=slab_thickness,
    )


# Fixed extra rotations per panel type so that the plane normal (local z)
# maps to the correct world axis:
#   xy / xy-3d → normal along world z  (no extra rotation)
#   xz / xz-3d → normal along world y  (-90° around x: [0,0,1]→[0,1,0])
#   yz / yz-3d → normal along world x  (+90° around y: [0,0,1]→[1,0,0])
_PANEL_EXTRA_ROTATION: dict[str, tuple[str, float]] = {
    "xz":    ("x", -90.0),
    "xz-3d": ("x", -90.0),
    "yz":    ("y",  90.0),
    "yz-3d": ("y",  90.0),
}

# Cross-section panel types we can download (excludes pure-3d panels)
_CROSS_SECTION_TYPES = frozenset(["xy", "xz", "yz", "xy-3d", "xz-3d", "yz-3d"])

# Human-readable position labels for the standard 4-panel layout
_PANEL_LABELS = {
    "xy":    "xy   (top-left)",
    "xz":    "xz   (bottom-left)",
    "yz":    "yz   (top-right)",
    "xy-3d": "xy-3d",
    "xz-3d": "xz-3d",
    "yz-3d": "yz-3d",
}


def _panel_quaternion(global_quat: list[float], panel_type: str) -> list[float]:
    """Compose the global crossSectionOrientation with the panel-type's fixed rotation."""
    Q = Rotation.from_quat(global_quat)
    extra = _PANEL_EXTRA_ROTATION.get(panel_type)
    if extra is None:
        return global_quat
    axis, deg = extra
    return (Q * Rotation.from_euler(axis, deg, degrees=True)).as_quat().tolist()


def _collect_panels(layout) -> list[tuple[str, object]]:
    """
    Recursively walk the layout tree and return (panel_type, viewer_or_None)
    for every downloadable cross-section panel found.

    Handles all layout node types:
      DataPanelLayout  "4panel"/"4panel-alt" → expands to xy, xz, yz entries
      DataPanelLayout  single type (xy/xz/yz/…) → one entry, viewer=None
      LayerGroupViewer → one entry carrying the viewer object for per-panel state
      StackLayout (row/column) → recurse into children
    """
    if layout is None:
        return []

    if isinstance(layout, DataPanelLayout):
        ptype = layout.type or ""
        if ptype in ("4panel", "4panel-alt"):
            return [("xy", None), ("xz", None), ("yz", None)]
        if ptype in _CROSS_SECTION_TYPES:
            return [(ptype, None)]
        return []  # "3d" or unknown

    if isinstance(layout, LayerGroupViewer):
        ptype = getattr(layout.layout, "type", "xy") or "xy"
        if ptype in _CROSS_SECTION_TYPES:
            return [(ptype, layout)]
        return []  # pure 3-D viewer

    if isinstance(layout, StackLayout):
        result = []
        for child in layout.children:
            result.extend(_collect_panels(child))
        return result

    return []


def select_panel_state(ng_state) -> tuple[list, list]:
    """
    Build a choice list from the layout and always prompt the user.

    The choice list is assembled as follows:
      1. Any LayerGroupViewer panels with an independent (unlinked) orientation
         are listed first, labelled as the current custom view.
      2. Standard xy / xz / yz options (using the global orientation + fixed
         rotation) are always appended, so the user can pick any axis even if
         the layout parser didn't find those panels explicitly.

    This guarantees a prompt and full user control regardless of layout
    complexity.
    """
    global_quat = (
        list(ng_state.crossSectionOrientation)
        if ng_state.crossSectionOrientation is not None
        else [0.0, 0.0, 0.0, 1.0]
    )
    global_pos = list(ng_state.position)

    # ── Build choice list ────────────────────────────────────────────────────
    # Each entry: (display_label, panel_type_str, viewer_or_None)
    choices: list[tuple[str, str, object]] = []
    seen_types: set[str] = set()

    for ptype, viewer in _collect_panels(ng_state.layout):
        if viewer is not None:
            ori = viewer.crossSectionOrientation
            if str(ori.link) != "linked" and ori.value is not None:
                # Custom/independent orientation — label it clearly
                label = f"{_PANEL_LABELS.get(ptype, ptype)}  [current view orientation]"
                choices.append((label, ptype, viewer))
                seen_types.add(ptype)
                continue
        # Standard linked panel — add by type, avoid duplicates
        if ptype not in seen_types:
            choices.append((_PANEL_LABELS.get(ptype, ptype), ptype, None))
            seen_types.add(ptype)

    # Always ensure xy / xz / yz are available as standard choices
    for ptype in ("xy", "xz", "yz"):
        if ptype not in seen_types:
            choices.append((_PANEL_LABELS.get(ptype, ptype), ptype, None))

    # ── Prompt ───────────────────────────────────────────────────────────────
    labels = [label for label, _, _ in choices]
    answers = inquirer.prompt([
        inquirer.List(
            "panel",
            message="Which panel do you want to download?",
            choices=labels,
        )
    ])
    _, panel_type, viewer = choices[labels.index(answers["panel"])]

    # ── Resolve orientation ──────────────────────────────────────────────────
    if viewer is not None:
        ori = viewer.crossSectionOrientation
        rotation_quat = (
            list(ori.value)
            if str(ori.link) != "linked" and ori.value is not None
            else _panel_quaternion(global_quat, panel_type)
        )
        pos = viewer.position
        position = (
            list(pos.value)
            if str(pos.link) != "linked" and pos.value is not None
            else global_pos
        )
    else:
        rotation_quat = _panel_quaternion(global_quat, panel_type)
        position = global_pos

    return position, rotation_quat


def print_plane_info(position: list[float], rotation_quat: list[float]) -> None:
    """Print the equation and normal of the current view plane in voxel coordinates."""
    plane = Plane(point=list(position), quarternion=rotation_quat)
    normal = plane.rotation.apply([0.0, 0.0, 1.0])
    d = float(np.dot(normal, position))
    print()
    print("Image plane (native-resolution voxel coordinates):")
    print(f"  Normal vector : ({normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f})")
    print(f"  Centre point  : ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    print(f"  Plane equation: {normal[0]:.6f}·x + {normal[1]:.6f}·y + {normal[2]:.6f}·z = {d:.6f}")
    print()


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
    slab_thickness: int = 1,
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
        return (0, 0) if slab_thickness == 1 else (0, 0, 0)
    min_tile_idx = np.min(tiles_in_bounds, axis=0).tolist()
    max_tile_idx = np.max(tiles_in_bounds, axis=0).tolist()

    xy_shape = tuple(
        int((ma - mi + 1) * c)
        for c, mi, ma in zip(plane.chunks, min_tile_idx, max_tile_idx, strict=True)
    )
    return (*xy_shape, slab_thickness) if slab_thickness > 1 else xy_shape


def save_image(
    *,
    gcs_url: str,
    downsample_level: int,
    position: list[int],
    rotation_quat: list[float],
    output_path: Path,
    max_ring: int | None = None,
    overwrite: bool = False,
    slab_thickness: int = 1,
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

    xy_shape = tuple(
        int((ma - mi + 1) * c)
        for c, mi, ma in zip(plane.chunks, min_tile_idx, max_tile_idx, strict=True)
    )
    output_image_shape = (*xy_shape, slab_thickness) if slab_thickness > 1 else xy_shape
    tile_shape = (*plane.chunks, slab_thickness) if slab_thickness > 1 else plane.chunks

    # depth offsets centred on 0, e.g. thickness=5 → [-2,-1,0,1,2]
    depth_offsets = list(range(-(slab_thickness // 2), slab_thickness - slab_thickness // 2))

    output_image_path = output_path.with_suffix(".zarr")
    TIFF_path = output_path.with_suffix(".tiff")

    if TIFF_path.exists() and not overwrite:
        yes_no_gate(f"{TIFF_path} already exists. Overwrite?", default=False)

    print(f"Creating output image, shape={output_image_shape}")
    print(f"Writing results to Zarr array at {output_image_path}")
    print(f"TIFF image will be updated every 10 tiles at {TIFF_path}")

    fill_value = input_image.fill_value.tolist() if input_image.fill_value is not None else 0

    output_image = create_local_tensorstore_array(
        path=output_image_path,
        shape=output_image_shape,
        tile_shape=tile_shape,
        dtype=str(input_image.dtype.numpy_dtype),
        fill_value=fill_value,
    )

    for i, tile_idx in enumerate(tqdm(tiles_in_bounds, desc="Downloading tiles")):
        x, y = plane.tile_coords(tile_idx)

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

        if slab_thickness > 1:
            # Compute world coords for every depth offset, take a single
            # bounding-box read that covers all of them, then interpolate per depth.
            all_wc = [plane.plane_coords_to_world(x, y, float(k)) for k in depth_offsets]
            all_c = [np.concatenate([wc[dim] for wc in all_wc]) for dim in range(3)]
            slc = tuple(
                slice(max(0, math.floor(min(c)) - 2), min(s, math.ceil(max(c)) + 2))
                for s, c in zip(input_image.shape, all_c)
            )
            arr_np = input_image[slc].read().result()
            arr_coords = tuple(np.arange(s.start, s.stop) for s in slc)

            slab_tile = np.empty((*plane.chunks, slab_thickness), dtype=np.float64)
            for k_idx, wc in enumerate(all_wc):
                slab_tile[:, :, k_idx] = scipy.interpolate.interpn(
                    points=arr_coords,
                    values=arr_np,
                    xi=np.vstack(wc).T,
                    bounds_error=False,
                    fill_value=fill_value,
                ).reshape(plane.chunks)

            output_image[(*output_slc, slice(None))].write(
                slab_tile.astype(input_image.dtype.numpy_dtype)
            ).result()
        else:
            world_coords = plane.plane_coords_to_world(x, y)
            slc = tuple(
                slice(max(0, math.floor(min(c)) - 2), min(s, math.ceil(max(c)) + 2))
                for s, c in zip(input_image.shape, world_coords, strict=True)
            )
            arr_np = input_image[slc].read().result()
            arr_coords = tuple(np.arange(s.start, s.stop) for s in slc)
            tile_image = scipy.interpolate.interpn(
                points=arr_coords,
                values=arr_np,
                xi=np.vstack(world_coords).T,
                bounds_error=False,
                fill_value=fill_value,
            ).reshape(plane.chunks)
            output_image[output_slc].write(
                tile_image.astype(input_image.dtype.numpy_dtype)
            ).result()

        if i % 10 == 0:
            arr = output_image[:].read().result()
            tifffile.imwrite(TIFF_path, arr.T)  # works for both 2-D (y,x) and 3-D (z,y,x)
            del arr

    print("Finished downloading tiles!")
    print("Image saved to:", output_image_path)
    print("Converting to TIFF...")
    arr = output_image[:].read().result()
    tifffile.imwrite(TIFF_path, arr.T)
    print("TIFF saved to:", TIFF_path)
    return tiles_in_bounds, min_tile_idx, plane.chunks
