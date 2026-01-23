# ng-slice-download

A simple command line utility to download the current neuroglancer view to a local TIFF file.

## Usage

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Open an OME-Zarr image in neuroglancer, and navigate to the 2D view that you want to download. **Important**: the view to be downloaded is in the upper left panel.
3. Copy the full neuroglancer link from the browser
4. Run:

```shell
uvx ng-slice-download '[full neuroglancer link]'
```
replacing `[full neuroglancer link] ` with your neuroglancer link.
It's important to surround the neuroglancer link in quotes because it will probably contain special characters.

### Command documentation
```shell
Usage: ng-slice-download [OPTIONS] NEUROGLANCER_URL

Options:
  --output_dir PATH  Directory to download image files to.
  --help             Show this message and exit.
```
