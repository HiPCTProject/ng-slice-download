# ng-slice-download

A simple command line utility to download the current neuroglancer view to a local TIFF file.

## Usage

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Open an OME-Zarr or N5 image in neuroglancer, and navigate to the 2D view that you want to download. **Important**: the view to be downloaded is in the upper left panel.
3. Copy the full neuroglancer link from the browser
4. Run:

```shell
uvx ng-slice-download '[full neuroglancer link]'
```
replacing `[full neuroglancer link]` with your neuroglancer link.
It's important to surround the neuroglancer link in quotes because it will probably contain special characters.



### Command documentation
```shell
% ng-slice-download --help
Usage: ng-slice-download [OPTIONS] NEUROGLANCER_URL

Options:
  --output-dir PATH    Directory to download image files to.
  --skip-lowres-check  Skip the low resolution check.
  --help               Show this message and exit.
```

## Changelog

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
