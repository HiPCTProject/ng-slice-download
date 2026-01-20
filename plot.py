import time

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr

# plt.imshow(arr, cmap="gray")
# plt.show()
while True:
    arr = zarr.open_array("rotated.zarr")[:]
    print("writing tiff...")
    tifffile.imwrite("rotated.tiff", arr.astype(np.uint16).T)
    time.sleep(5)
