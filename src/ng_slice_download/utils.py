import inquirer
import tensorstore as ts


def open_tensorstore_array(
    gcs_url: str, *, downsample_level: int = 0
) -> ts.TensorStore:
    driver, _, path = gcs_url.split("://")
    bucket, path = path.split("/", maxsplit=1)
    if driver == "n5":
        downsample_level = f"s{downsample_level}"
    else:
        downsample_level = f"{downsample_level}"
    return ts.open(
        {
            "driver": driver,
            "kvstore": {
                "driver": "gcs",
                "bucket": bucket,
                "path": path + f"{downsample_level}/",
            },
            "context": {"cache_pool": {"total_bytes_limit": 100_000_000}},
            "recheck_cached_data": False,
        }
    ).result()


def create_local_tensorstore_array(
    *,
    path: str,
    shape: tuple[int, int],
    tile_shape: tuple[int, int],
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
