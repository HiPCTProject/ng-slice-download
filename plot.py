import numpy as np
import tifffile
import zarr

arr = zarr.open_array("rotated.zarr")[:]
print("writing tiff...")
tifffile.imwrite("rotated.tiff", arr.astype(np.uint16).T)
