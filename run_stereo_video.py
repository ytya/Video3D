import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import skvideo

# ffmpeg_path = r""
# skvideo.setFFmpegPath(ffmpeg_path)
import skvideo.io
from PIL import Image
from tqdm import tqdm

from video3d.stereo import SCONFIGS, SBSCreator, StereoConfig
from video3d.util import read_video_info

_creator: SBSCreator | None = None


def _init_create_sbs(config: StereoConfig):
    global _creator
    _creator = SBSCreator(config)


def _create_sbs(image_path: Path, depth_path: Path) -> np.ndarray:
    global _creator
    image = np.asarray(Image.open(image_path))
    depth = np.asarray(Image.open(depth_path))
    frame, blank = _creator(image, depth)
    return frame


def run(
    src_path: Path,
    image_dir: Path,
    depth_dir: Path,
    output_path: Path,
    frame_interpolated: int,
    config: StereoConfig,
    worker_num: int,
    crf: int,
):
    """Side by Sideの3D動画生成

    :param Path src_path: 入力動画
    :param Path image_dir: 入力画像フォルダ
    :param Path depth_dir: 入力Depthフォルダ
    :param Path output_path: 出力動画
    :param int frame_interpolated: フレーム補間数
    :param StereoConfig config: ステレオ設定
    :param int worker_num: 並列数
    :param int crf: 出力CRF
    """
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # ソース動画情報取得
    vinfo = read_video_info(src_path)
    nb_frames = vinfo.frame_num
    r_frame_rate = f"{int(vinfo.frame_rate_n)*frame_interpolated}/{vinfo.frame_rate_d}"
    print(f"video frame rate: {r_frame_rate},  video frames: {nb_frames}")

    # 動画Writer作成
    writer = skvideo.io.FFmpegWriter(
        output_path,
        inputdict={
            "-r": r_frame_rate,
        },
        outputdict={"-vcodec": "libx264", "-pix_fmt": "yuv420p", "-r": r_frame_rate, "-crf": str(crf)},
    )

    # 3D動画生成
    image_paths = sorted(image_dir.glob("*.jpg"))
    depth_paths = [depth_dir / f"{img_path.stem}.png" for img_path in image_paths]
    chunk_num = 60
    with ProcessPoolExecutor(max_workers=worker_num, initializer=_init_create_sbs, initargs=(config,)) as executor:
        # メモリ消費を抑制するためにチャンクに分割して処理
        for i in tqdm(range(0, len(image_paths), chunk_num)):
            futures = [
                executor.submit(_create_sbs, img_path, depth_path)
                for img_path, depth_path in zip(image_paths[i : i + chunk_num], depth_paths[i : i + chunk_num])
            ]
            for future in futures:
                frame = future.result()
                writer.writeFrame(frame)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", help="src video path")
    parser.add_argument("-i", "--image_dir", default=None, help="folder with input images")
    parser.add_argument("-d", "--depth_dir", default=None, help="folder with depth images")
    parser.add_argument("-o", "--output_path", default=None, help="output video path")
    parser.add_argument("-f", "--frame_interpolated", default=1, type=int, help="input images is frame interpolated")
    parser.add_argument("-t", "--target_device", default="projector", help="target device (projector or mobile)")
    parser.add_argument("-w", "--worker_num", default=3, type=int, help="number of workers")
    parser.add_argument("--crf", default=20, type=int, help="output video quality (lower is better)")
    args = parser.parse_args()

    src_path = Path(args.src_path)
    title = src_path.stem
    if args.image_dir is not None:
        image_dir = Path(args.image_dir)
    else:
        image_dir = src_path.parent.parent / "src_jpg" / title
    if args.depth_dir is not None:
        depth_dir = Path(args.depth_dir)
    else:
        depth_dir = src_path.parent.parent / "depth" / title
    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = src_path.parent.parent / "3d" / src_path.name
    config = SCONFIGS[args.target_device]

    run(src_path, image_dir, depth_dir, output_path, args.frame_interpolated, config, args.worker_num, args.crf)
