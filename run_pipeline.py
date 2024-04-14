import argparse
import subprocess
from pathlib import Path

import numpy as np
import skvideo

# ffmpeg_path = r""
# skvideo.setFFmpegPath(ffmpeg_path)
import skvideo.io
import torch
import torch.multiprocessing as mp
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from video3d.midas import MiDaSModel
from video3d.nvds import NVDSModel
from video3d.stereo import SCONFIGS, SBSCreator, StereoConfig
from video3d.util import NDArray, NDArrayUint8, depth_to_uint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class VideoReader:
    def __init__(self, path: Path):
        np.float = np.float64  # skvideoバグ対策
        np.int = np.int_
        self._reader = skvideo.io.FFmpegReader(str(path))

    def __iter__(self):
        for i, image in enumerate(self._reader.nextFrame()):
            yield f"{i:05d}", image

    def close(self):
        self._reader.close()


class ImageReader:
    def __init__(self, paths: list[Path]):
        self._paths = paths

    def __iter__(self):
        for p in self._paths:
            yield p.stem, np.asarray(Image.open(p))

    def close(self):
        pass


def midas_worker(proc_idx: int, input_images: Path | list[Path], output_queue: mp.Queue, use_half: bool = True):
    """MiDaS推定"""
    try:
        midas = MiDaSModel("DPT_BEiT_L_512", use_half=use_half, device=DEVICE)

        if isinstance(input_images, Path):
            reader = VideoReader(input_images)
        else:
            reader = ImageReader(input_images)

        for image_stem, image in reader:
            # Depth推定
            depth = midas(image, inter_mode=None).cpu()  # [height, width]

            output_queue.put((image_stem, image, depth))

        output_queue.put(None)
        reader.close()
    except Exception as e:
        print(f"midas_worker: {e}")
        output_queue.put(None)


def nvds_worker(proc_idx: int, input_queue: mp.Queue, output_queue: mp.Queue, use_half: bool = True):
    """NVDS推定"""
    try:
        nvds = NVDSModel(use_half=use_half, device=DEVICE)

        images = []
        image_stems = []
        last_depth = None
        is_end = False
        while True:
            inp: tuple[str, NDArrayUint8, Tensor] | None = input_queue.get()
            if inp is None:
                if is_end is False:
                    # 最後のフレームがきたら最終フレームを繰り返す
                    is_end = True
                    for _ in range(nvds.seq_len - 1):
                        input_queue.put(("", images[-1], last_depth))
                    input_queue.put(None)
                    continue
                else:
                    # 2回目のNoneで終了
                    output_queue.put(None)
                    break

            image_stem, image, depth = inp
            first_frame = last_depth is None
            images.append(image)
            image_stems.append(image_stem)
            last_depth = depth

            if first_frame:
                # 最初のフレームは外挿分繰り返す
                for _ in range(nvds.seq_len):
                    nvds(image, depth)
                continue

            # NVDS
            idx, depth = nvds(image, depth)
            if depth is None:
                continue

            # 出力
            depth = depth.cpu().numpy()
            output_queue.put((image_stems[0], images[0], depth))
            image_stems = image_stems[1:]
            images = images[1:]
    except Exception as e:
        print(f"nvds_worker: {e}")
        output_queue.put(None)


def create_sbs_worker(
    proc_idx: int, input_queue: mp.Queue, output_queue: mp.Queue, depth_dir: Path | None, config: StereoConfig
) -> np.ndarray:
    """SBS画像作成"""

    try:
        creator = SBSCreator(config)

        while True:
            inp: tuple[str, NDArrayUint8, NDArray] | None = input_queue.get()
            if inp is None:
                output_queue.put(None)
                break

            # 画像取得
            image_stem, image, depth = inp
            depth = depth_to_uint(depth, np.uint16)
            if depth_dir is not None:
                depth_path = depth_dir / f"{image_stem}.png"
                Image.fromarray(depth).save(depth_path, compress_level=3)

            # SBS作成
            frame = creator(image, depth)
            output_queue.put((image_stem, frame))
    except Exception as e:
        print(f"create_sbs_worker: {e}")
        output_queue.put(None)


def run(
    src_path: Path,
    image_dir: Path | None,
    depth_dir: Path | None,
    output_path: Path,
    frame_interpolated: int,
    config: StereoConfig,
    crf: int,
):
    """MiDaS -> NVDS -> SBS -> 3D動画生成

    :param Path src_path: 入力動画
    :param Path | None image_dir: 入力画像フォルダ
    :param Path | None depth_dir: 出力Depthフォルダ
    :param Path output_path: 出力動画
    :param int frame_interpolated: フレーム補間数
    :param StereoConfig config: ステレオ設定
    :param int crf: 出力CRF
    """
    # ソース動画情報取得
    video_info = skvideo.io.ffprobe(src_path)["video"]
    avg_frame_rate = video_info["@avg_frame_rate"]
    nb_frames = int(video_info.get("@nb_frames", 0))
    if nb_frames == 0:
        for tag in video_info["tag"]:
            if tag["@key"] == "NUMBER_OF_FRAMES":
                nb_frames = int(tag["@value"])
                break
    a, b = avg_frame_rate.split("/")
    r_frame_rate = f"{int(a)*frame_interpolated}/{b}"
    print(f"video frame rate: {r_frame_rate},  video frames: {nb_frames}")

    # 入力動画
    midas_input = src_path
    if image_dir is not None:
        image_paths = list(sorted(image_dir.glob("*.jpg")))
        midas_input = image_paths
        nb_frames = len(image_paths)

    if depth_dir is not None:
        depth_dir.mkdir(exist_ok=True, parents=True)

    # worker起動
    queue_size = 5
    nvds_queue = mp.Queue(maxsize=queue_size)
    sbs_queue = mp.Queue(maxsize=queue_size)
    output_queue = mp.Queue(maxsize=queue_size)
    midas_proc = mp.spawn(midas_worker, args=(midas_input, nvds_queue, True), nprocs=1, join=False, daemon=True)
    nvds_proc = mp.spawn(nvds_worker, args=(nvds_queue, sbs_queue, True), nprocs=1, join=False, daemon=True)
    sbs_proc = mp.spawn(
        create_sbs_worker,
        args=(sbs_queue, output_queue, depth_dir, config),
        nprocs=1,
        join=False,
        daemon=True,
    )

    # 動画Writer作成
    output_path.parent.mkdir(exist_ok=True, parents=True)
    nosound_path = output_path.parent / f"{output_path.stem}_nosound{output_path.suffix}"
    writer = skvideo.io.FFmpegWriter(
        nosound_path,
        inputdict={
            "-r": r_frame_rate,
        },
        outputdict={"-vcodec": "libx264", "-pix_fmt": "yuv420p", "-r": r_frame_rate, "-crf": str(crf)},
    )

    # 出力
    with tqdm(total=nb_frames) as pbar:
        while True:
            out: tuple[str, np.ndarray] | None = output_queue.get()
            pbar.set_postfix({"nvds": nvds_queue.qsize(), "sbs": sbs_queue.qsize()}, refresh=False)
            pbar.update(1)
            if out is None:
                break

            image_stem, frame = out
            writer.writeFrame(frame[:, :, ::-1])

    writer.close()

    # 音声コピー
    print("audio copy")
    command = f'ffmpeg -i "{src_path}" -i "{nosound_path}" -c:v copy -c:a copy -map 0:a -map 1:v "{output_path}"'
    print(command)
    ret = subprocess.call(command, shell=True)
    if ret != 0:
        print("audio copy error")
    else:
        nosound_path.unlink()

    midas_proc.join(1)
    nvds_proc.join(1)
    sbs_proc.join(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", help="src video path")
    parser.add_argument("-i", "--image_dir", default=None, help="folder with input images")
    parser.add_argument("-o", "--output_path", default=None, help="output video path")
    parser.add_argument("-d", "--depth_dir", default=None, help="folder with depth images")
    parser.add_argument("-f", "--frame_interpolated", default=1, type=int, help="input images is frame interpolated")
    parser.add_argument("-t", "--target_device", default="projector", help="target device (projector or mobile)")
    parser.add_argument("--crf", default=20, type=int, help="output video quality (lower is better)")
    args = parser.parse_args()

    src_path = Path(args.src_path)
    title = src_path.stem
    image_dir = args.image_dir
    depth_dir = args.depth_dir
    if image_dir is not None:
        image_dir = Path(image_dir)
    if depth_dir is not None:
        depth_dir = Path(depth_dir)
    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = src_path.parent.parent / "3d" / src_path.name
    config = SCONFIGS[args.target_device]
    run(src_path, image_dir, depth_dir, output_path, args.frame_interpolated, config, args.crf)
