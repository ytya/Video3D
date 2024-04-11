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
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from nvds.full_model import NVDS


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

WORKER_NUM = 2  # 並列数

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class NVDSModel:
    def __init__(self, seq_len: int = 4, use_half: bool = True):
        self._seq_len = seq_len
        self._use_half = use_half

        # NVDS
        self.__mean = [0.485, 0.456, 0.406]
        self.__std = [0.229, 0.224, 0.225]
        checkpoint = torch.load("./NVDS/NVDS_checkpoints/NVDS_Stabilizer.pth", map_location="cpu")
        self._nvds = NVDS()
        self._nvds = torch.nn.DataParallel(self._nvds, device_ids=[0]).cuda()
        self._nvds.load_state_dict(checkpoint)
        if use_half:
            self._nvds = self._nvds.half()
        self._nvds = self._nvds.to(DEVICE)
        self._nvds.eval()

        # MiDaS
        self._midas = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512", pretrained=True).to(DEVICE)
        self._midas.eval()
        if use_half:
            self._midas = self._midas.half()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self._midas_transform = midas_transforms.beit512_transform

    def _infer_nvds(self, fwd_frames: list[Tensor], bwd_frames: list[Tensor]) -> Tensor:
        fwd_frames = torch.cat(fwd_frames, dim=0).unsqueeze(0)  # [1, seq_len, channel, height, width]
        bwd_frames = torch.cat(bwd_frames, dim=0).unsqueeze(0)
        frames = torch.cat([fwd_frames, bwd_frames], dim=0)  # [2, seq_len, channel, height, width]
        if self._use_half:
            frames = frames.half()
        pred = self._nvds(frames.to(DEVICE))
        return (pred[0] + pred[1]) / 2

    def _generate_rgbd(self, path: Path) -> Tensor:
        frame = Image.open(path)

        # Depth推定
        batch = self._midas_transform(np.asarray(frame))
        if self._use_half:
            batch = batch.half()
        depth = self._midas(batch.to(DEVICE)).cpu()  # [1, height, width]

        # 正規化
        if not torch.isfinite(depth).all():
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_min = torch.min(depth)
        depth_max = torch.max(depth)
        if depth_max - depth_min > torch.finfo(torch.float).eps:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = torch.zeros(depth.shape, dtype=depth.dtype)

        # rgbdフレーム化
        rgb = np.asarray(frame.resize(depth.shape[1:][::-1], Image.Resampling.BICUBIC)) / 255.0
        rgb = (rgb - self.__mean) / self.__std
        rgb = torch.Tensor(np.transpose(rgb, (2, 0, 1)).astype(np.float32))
        rgbd = torch.cat([rgb, depth], dim=0)
        return rgbd

    def infer(self, dst_dir: Path, frame_paths: list[Path]):
        frame_paths = sorted(frame_paths)

        with torch.no_grad():
            # 初期フレーム読み込み
            seq_frames = []
            for i in range(self._seq_len - 1):
                rgbd = self._generate_rgbd(frame_paths[i]).unsqueeze(0)  # [1, channel, height, width]

                if i == 0:
                    seq_frames = [rgbd] * (self._seq_len + 1)
                else:
                    seq_frames.append(rgbd)

            # フレーム処理
            for i in range(len(frame_paths)):
                # 次フレーム読み込み
                next_idx = i + self._seq_len - 1
                if next_idx < len(frame_paths):
                    rgbd = self._generate_rgbd(frame_paths[next_idx]).unsqueeze(0)
                else:
                    rgbd = seq_frames[-1]
                seq_frames = seq_frames[1:] + [rgbd]

                # NVDS
                depth = self._infer_nvds(seq_frames[: self._seq_len], seq_frames[self._seq_len - 1 :][::-1])

                # 16bit化
                depth = depth.squeeze().cpu().numpy()
                if not np.isfinite(depth).all():
                    print("inf")
                depth_min = depth.min()
                depth_max = depth.max()

                bits = np.uint16(0).nbytes
                max_val = (2 ** (8 * bits)) - 1
                if depth_max - depth_min > np.finfo("float").eps:
                    depth = max_val * (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth = np.zeros(depth.shape, dtype=depth.dtype)
                depth = depth.astype(np.uint16)

                yield depth


def invert_map(f):
    """remap用に指定する"""
    inv = np.zeros_like(f)
    inv[:, :, 1], inv[:, :, 0] = np.indices(f.shape[:2])
    p = np.copy(inv)
    for i in range(10):
        cor = inv - cv2.remap(f, p, None, interpolation=cv2.INTER_LINEAR)
        p += cor * 0.5
    return p


def create_sbs(jpg_path: Path, depth: np.ndarray, config: StereoConfig) -> np.ndarray:
    """SBS画像作成"""
    src_img = cv2.imread(str(jpg_path))
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
    jpg_paths = list(sorted(jpg_dir.glob("*.jpg")))
    # depth_paths = sorted(depth_dir.glob("*.png"))

    # モデル準備
    nvds = NVDSModel(seq_len=4, use_half=True)

    # 処理開始
    with ProcessPoolExecutor(max_workers=WORKER_NUM) as executor:
        features = [None] * WORKER_NUM
        for i, (jpg_path, depth) in tqdm(
            enumerate(zip(jpg_paths, nvds.infer(depth_dir, jpg_paths))), total=len(jpg_paths)
        ):
            future = executor.submit(create_sbs, jpg_path, depth, config)
            idx = i % WORKER_NUM
            if features[idx] is not None:
                writer.writeFrame(features[idx].result())
            features[idx] = future

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
