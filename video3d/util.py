from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

NDArrayFloat = NDArray[np.float32] | NDArray[np.float16]
NDArrayUint = NDArray[np.uint8] | NDArray[np.uint16]
NDArrayUint8 = NDArray[np.uint8]
DTypeUint = np.uint8 | np.uint16


def depth_to_uint(depth: NDArray, dtype: DTypeUint = np.uint16) -> NDArrayUint:
    """depthをuintに変換

    :param NDArray depth: Depth画像
    :param DTypeLike dtype: 変換先の型, defaults to np.uint16
    :return NDArrayUint: 変換後のDepth画像
    """
    bits = 8 * dtype(0).nbytes
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**bits) - 1
    if depth_max - depth_min > np.finfo("float").eps:
        depth = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros(depth.shape, dtype=dtype)
    return depth.astype(dtype)


def save_depth(path: Path, depth: NDArray, dtype: DTypeUint = np.uint16):
    """Depth画像保存

    :param path Path: 保存パス
    :param depth NDArray: Depth画像
    :param dtype DTypeUint: 保存型
    """
    if depth.dtype != dtype:
        depth = depth_to_uint(depth, dtype)
    if path.suffix == ".png":
        Image.fromarray(depth).save(path, compress_level=3)
    else:
        Image.fromarray(depth).save(path)
