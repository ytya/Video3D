import dataclasses
from pathlib import Path

import numpy as np
import skvideo.io
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


@dataclasses.dataclass
class VideoInfo:
    width: int
    height: int
    frame_rate_n: int
    frame_rate_d: int
    frame_num: int


def read_video_info(path: Path) -> VideoInfo:
    """動画情報読み込み

    :param Path path: 動画パス
    :return: VideoInfo
    """
    video_info = skvideo.io.ffprobe(path)["video"]
    width = video_info["@width"]
    height = video_info["@height"]
    avg_frame_rate = video_info["@avg_frame_rate"]
    nb_frames = int(video_info.get("@nb_frames", 0))
    if nb_frames == 0:
        for tag in video_info["tag"]:
            if tag["@key"] == "NUMBER_OF_FRAMES":
                nb_frames = int(tag["@value"])
                break
    a, b = avg_frame_rate.split("/")

    return VideoInfo(
        int(width),
        int(height),
        int(a),
        int(b),
        nb_frames,
    )


class VideoReader:
    def __init__(self, path: Path):
        np.float = np.float64  # skvideoバグ対策
        np.int = np.int_
        self._reader = skvideo.io.FFmpegReader(str(path))

    def __del__(self):
        self.close()

    def __iter__(self):
        for i, image in enumerate(self._reader.nextFrame()):
            yield i, f"{i:05d}", image

    def close(self):
        self._reader.close()


class ImageReader:
    def __init__(self, paths: list[Path]):
        self._paths = paths

    def __del__(self):
        self.close()

    def __iter__(self):
        for i, p in enumerate(self._paths):
            yield i, p.stem, np.asarray(Image.open(p))

    def close(self):
        pass
