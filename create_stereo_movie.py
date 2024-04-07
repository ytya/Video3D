import argparse
import dataclasses
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import skvideo

# ffmpeg_path = r""
# skvideo.setFFmpegPath(ffmpeg_path)
import skvideo.io
from tqdm import tqdm


# プロジェクター用設定
@dataclasses.dataclass(frozen=True)
class StereoConfig:
    """変換設定"""

    dist_screen: int  # 観察距離[mm]
    dist_forward: int  # 凸方向の距離の最大値[mm]
    dist_back: int  # 凹方向の距離の最大値[mm]
    pd: int  # 瞳孔間距離[mm]
    screen_w: int  # スクリーン幅[mm]
    image_w: int  # 画像幅[px]
    is_half: bool  # half side by sideで出力
    dist_min: int = dataclasses.field(init=False)
    dist_max: int = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "dist_min", self.dist_screen - self.dist_forward)
        object.__setattr__(self, "dist_max", self.dist_screen + self.dist_back)


CONFIG = {
    "projector": StereoConfig(
        dist_screen=2200, dist_forward=818, dist_back=3176, pd=65, screen_w=2037, image_w=1920, is_half=True
    ),
    "mobile": StereoConfig(
        dist_screen=250, dist_forward=18, dist_back=21, pd=65, screen_w=250, image_w=1920, is_half=False
    ),
}

WORKER_NUM = 3  # 並列数


def invert_map(f):
    """remap用に指定する"""
    inv = np.zeros_like(f)
    inv[:, :, 1], inv[:, :, 0] = np.indices(f.shape[:2])
    p = np.copy(inv)
    for i in range(10):
        cor = inv - cv2.remap(f, p, None, interpolation=cv2.INTER_LINEAR)
        p += cor * 0.5
    return p


def create_sbs(jpg_path: Path, depth_path: Path, config: StereoConfig) -> np.ndarray:
    """SBS画像作成"""
    src_img = cv2.imread(str(jpg_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    height, width = src_img.shape[:2]

    if (depth.shape[0] != height) or (depth.shape[1] != width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)

    # 境界部のdepth推定エラーをごまかす
    depth2 = cv2.dilate(depth, np.ones((7, 7), np.uint8), iterations=3)
    gap = depth2 - depth > (depth.max() - depth.min()) / 4
    depth[gap] = depth2[gap]

    # 距離計算 (depth = 1 / 距離)
    depth_max = int(depth.max())
    depth_min = int(depth.min())
    scale = (1 / config.dist_max - 1 / config.dist_min) / (depth_min - depth_max)
    shift = 1 / config.dist_min - depth_max * scale
    dist = 1 / (depth * scale + shift)
    dist = cv2.blur(dist, ksize=(11, 11))

    # 視差計算
    dist -= config.dist_screen  # スクリーンより手前がマイナス
    disp = dist * config.pd / (config.dist_screen + dist)  # プラスならR画素がL画素より右、マイナスなら逆
    disp_px = disp / config.screen_w * config.image_w / 2  # 視差

    # SBS用remap計算
    left_map_x, left_map_y = np.meshgrid(range(width), range(height))
    right_map_x = left_map_x.copy()
    right_map_y = left_map_y.copy()

    left_map_x = left_map_x - disp_px
    right_map_x = right_map_x + disp_px
    left_map = np.stack((left_map_x, left_map_y), axis=2).astype(np.float32)
    right_map = np.stack((right_map_x, right_map_y), axis=2).astype(np.float32)
    left_map_inv = invert_map(left_map)
    right_map_inv = invert_map(right_map)

    # 画像生成
    left_img = cv2.remap(src_img, left_map_inv, None, interpolation=cv2.INTER_LINEAR)
    right_img = cv2.remap(src_img, right_map_inv, None, interpolation=cv2.INTER_LINEAR)
    frame = np.hstack((left_img, right_img))[:, :, ::-1]
    if config.is_half:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    return frame


def create_sbs_zeodepth(jpg_path: Path, depth_path: Path, config: StereoConfig) -> np.ndarray:
    """SBS画像作成"""
    src_img = cv2.imread(str(jpg_path))
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    height, width = src_img.shape[:2]

    if (depth.shape[0] != height) or (depth.shape[1] != width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)

    # 境界部のdepth推定エラーをごまかす
    # depth2 = cv2.dilate(depth, np.ones((7, 7), np.uint8), iterations=3)
    # gap = depth2 - depth > (depth.max() - depth.min()) / 4
    # depth[gap] = depth2[gap]

    # 距離計算 (depth = 距離)
    depth_max = int(depth.max())
    depth_min = int(depth.min())
    scale = (1 / config.dist_max - 1 / config.dist_min) / (1 / depth_max - 1 / depth_min)
    shift = 1 / config.dist_min - (1 / depth_min) * scale
    dist = 1 / ((1 / depth) * scale + shift)
    dist = cv2.blur(dist, ksize=(11, 11))

    # 視差計算
    dist -= config.dist_screen  # スクリーンより手前がマイナス
    disp = dist * config.pd / (config.dist_screen + dist)  # プラスならR画素がL画素より右、マイナスなら逆
    disp_px = disp / config.screen_w * config.image_w / 2  # 視差

    # SBS用remap計算
    left_map_x, left_map_y = np.meshgrid(range(width), range(height))
    right_map_x = left_map_x.copy()
    right_map_y = left_map_y.copy()

    left_map_x = left_map_x - disp_px
    right_map_x = right_map_x + disp_px
    left_map = np.stack((left_map_x, left_map_y), axis=2).astype(np.float32)
    right_map = np.stack((right_map_x, right_map_y), axis=2).astype(np.float32)
    left_map_inv = invert_map(left_map)
    right_map_inv = invert_map(right_map)

    # 画像生成
    left_img = cv2.remap(src_img, left_map_inv, None, interpolation=cv2.INTER_LINEAR)
    right_img = cv2.remap(src_img, right_map_inv, None, interpolation=cv2.INTER_LINEAR)
    frame = np.hstack((left_img, right_img))[:, :, ::-1]
    if config.is_half:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    return frame


def run(
    src_path: Path, jpg_dir: Path, depth_dir: Path, output_path: Path, frame_interpolated: int, config: StereoConfig
):
    """3D動画生成"""
    # ステレオ画像生成
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # ソース動画情報取得
    video_info = skvideo.io.ffprobe(src_path)["video"]
    r_frame_rate = video_info["@r_frame_rate"]
    a, b = r_frame_rate.split("/")
    r_frame_rate = f"{int(a)*frame_interpolated}/{b}"
    print(r_frame_rate)
    codec_name = video_info["@codec_name"]
    pix_fmt = video_info["@pix_fmt"]

    # 動画Writer作成
    writer = skvideo.io.FFmpegWriter(
        output_path,
        inputdict={
            "-r": r_frame_rate,
        },
        outputdict={"-vcodec": codec_name, "-pix_fmt": pix_fmt, "-r": r_frame_rate},
    )

    # 3D動画生成
    jpg_paths = sorted(jpg_dir.glob("*.jpg"))
    depth_paths = sorted(depth_dir.glob("*.png"))
    chunk_num = 60
    with ProcessPoolExecutor(max_workers=WORKER_NUM) as executor:
        # メモリ消費を抑制するためにチャンクに分割して処理
        for i in tqdm(range(0, len(jpg_paths), chunk_num)):
            futures = [
                executor.submit(create_sbs, jpg_path, depth_path, config)
                for jpg_path, depth_path in zip(jpg_paths[i : i + chunk_num], depth_paths[i : i + chunk_num])
            ]
            for future in futures:
                frame = future.result()
                writer.writeFrame(frame)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", help="src movie path")
    parser.add_argument("-i", "--image_path", default=None, help="folder with input images")
    parser.add_argument("-d", "--depth_path", default=None, help="folder with depth images")
    parser.add_argument("-o", "--output_path", default=None, help="output movie path")
    parser.add_argument("-f", "--frame_interpolated", default=1, type=int, help="input images is frame interpolated")
    parser.add_argument("-t", "--target_device", default="projector", help="target device (projector or mobile)")
    args = parser.parse_args()

    src_path = Path(args.src_path)
    title = src_path.stem
    if args.image_path is not None:
        jpg_dir = Path(args.image_path)
    else:
        jpg_dir = src_path.parent.parent / "src_jpg" / title
    if args.depth_path is not None:
        depth_dir = Path(args.depth_path)
    else:
        depth_dir = src_path.parent.parent / "depth" / title
    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = src_path.parent.parent / "3d" / src_path.name
    config = CONFIG[args.target_device]
    run(src_path, jpg_dir, depth_dir, output_path, args.frame_interpolated, config)
