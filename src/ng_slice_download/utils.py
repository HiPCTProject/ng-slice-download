import inquirer
import tensorstore as ts


def open_tensorstore_array(
    url: str, *, downsample_level: int = 0
) -> ts.TensorStore:
    # URL forms from neuroglancer state:
    #   zarr://gs://bucket/path          — GCS
    #   zarr://http://host:port/path     — HTTP (e.g. ngtools local fileserver)
    #   zarr://https://host/path         — HTTPS
    #   n5://gs://bucket/path            — N5 on GCS
    #   n5://http://host:port/path       — N5 via HTTP

    format_driver, rest = url.split("://", maxsplit=1)
    # rest: "gs://bucket/path" | "http://host/path" | "https://host/path"

    if format_driver == "n5":
        level_suffix = f"s{downsample_level}"
    else:
        level_suffix = str(downsample_level)

    if rest.startswith("gs://"):
        gcs_path = rest[5:]  # strip "gs://"
        bucket, path = gcs_path.split("/", maxsplit=1)
        kvstore = {
            "driver": "gcs",
            "bucket": bucket,
            "path": path.rstrip("/") + f"/{level_suffix}/",
        }
    elif rest.startswith("http://") or rest.startswith("https://"):
        # Served over HTTP — e.g. from ngtools local fileserver
        kvstore = {
            "driver": "http",
            "base_url": rest.rstrip("/") + f"/{level_suffix}/",
        }
    else:
        # Local file path
        kvstore = {
            "driver": "file",
            "path": rest.rstrip("/") + f"/{level_suffix}/",
        }

    return ts.open(
        {
            "driver": format_driver,
            "kvstore": kvstore,
            "context": {"cache_pool": {"total_bytes_limit": 100_000_000}},
            "recheck_cached_data": False,
        }
    ).result()


def create_local_tensorstore_array(
    *,
    path: str,
    shape: tuple[int, ...],
    tile_shape: tuple[int, ...],
    dtype: str,
    fill_value: float,
) -> ts.TensorStore:
    """
    Warnings
    --------
    This will overwrite any existing array!
    """
    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "create": True,
            "delete_existing": True,
            "metadata": {
                "data_type": dtype,
                "shape": shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": tile_shape},
                },
                "codecs": [],
                "fill_value": fill_value,
            },
        }
    ).result()


def yes_no_gate(message: str, *, default: bool) -> None:
    questions = [
        inquirer.Confirm("continue", message=message, default=default),
    ]
    answers = inquirer.prompt(questions)
    if answers is None or not answers["continue"]:
        exit()
