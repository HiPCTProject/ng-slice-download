# ng-slice-download

A simple command line utility to download the current neuroglancer view to a local TIFF file, with minimal compute requirements.

## Usage

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Open an OME-Zarr or N5 image in neuroglancer, and navigate to the view that you want to download. If your layout has more than one panel, you will be prompted to pick which one to download (see below).
3. Copy the full neuroglancer link from the browser
4. Run:

```shell
uvx ng-slice-download '[full neuroglancer link]'
```
replacing `[full neuroglancer link]` with your neuroglancer link.
It's important to surround the neuroglancer link in quotes because it will probably contain special characters.

### Workflow

1. **Panel selection** — If the neuroglancer layout has multiple views (e.g. xy/xz/yz, or a custom layer-group layout), you'll be asked which panel to download. For simple single-axis layouts (`layout=xz`, `layout=yz`, ...) the equivalent cross-section is reconstructed automatically.
2. **Low-resolution check** (skip with `--skip-lowres-check`) — a small preview TIFF and an annotated PNG are saved first. The PNG outlines each tile in red and labels it with its "ring" number (distance from the centre tile, 0 = centre), so you can check the crop is centred where you expect before committing to a large download.
3. **Ring cutoff** — after confirming the preview looks right, you can optionally restrict the download to tiles within a maximum ring number, cropping the output instead of downloading the full extent.
4. **Resolution selection** — pick the downsample level to download, with output image dimensions shown for each option (already accounting for any ring cutoff).
5. **Download** — tiles are streamed into a local Zarr array (updated on disk as it goes) and converted to a final TIFF once complete. The in-progress TIFF preview is refreshed every 10 tiles so you can monitor progress.

### Thick slab downloads

By default a single 2D plane is extracted. Pass `--slab-thickness N` to instead download a 3D slab of `N` depth slices, centred on the view plane (e.g. `--slab-thickness 5` downloads 2 slices above and below the plane, plus the plane itself). The output TIFF and Zarr array gain a third (depth) dimension when thickness is greater than 1.

### Command documentation
```shell
% ng-slice-download --help
Usage: ng-slice-download [OPTIONS] NEUROGLANCER_URL

Options:
  --output-dir PATH         Directory to download image files to.
  --skip-lowres-check       Skip the low resolution check.
  --overwrite-check         Overwrite existing preview files without
                             prompting.
  --slab-thickness INTEGER  Number of depth slices to download, centred on
                             the view plane (1 = single plane).  [default: 1]
  --help                    Show this message and exit.
```

## Changelog

### Unreleased

- Add `--slab-thickness` option to download a thick 3D slab (multiple depth slices) centred on the view plane, instead of a single 2D plane.
- Support selecting any panel in the neuroglancer layout (not just the upper-left one), including custom layer-group layouts with multiple viewers, and fix several panel-selection/orientation bugs.
- Save an annotated PNG alongside the low-resolution preview TIFF, labelling each tile with its ring distance from the centre, and allow cropping the download to a maximum ring number.
- Add `--overwrite-check` option to overwrite existing preview files without prompting.

### 0.2

- Ask before overwriting output TIFF files (.zarr files are still overwritten without asking)
- Saved images use the Neuroglancer layer name in the filename
- Added support for N5 images.
- Renamed the --output_dir flag to --output-dir.
- Add --skip-lowres-check option to skip the initial low resolution check that the orientation is correct.
- Add a more helpful error message if selected layer is not an image layer.
- Allow layers with a transform, but without a transformation matrix.

### 0.1.1

First release
