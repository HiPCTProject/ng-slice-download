import numpy as np

from ng_slice_download.utils import crop_to_fill_value


def test_crop():
    arr = np.zeros((3, 4))
    arr[1, 1] = 1

    result = crop_to_fill_value(arr, 0)
    np.testing.assert_equal(result, [[1]])
