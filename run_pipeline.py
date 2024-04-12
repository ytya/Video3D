import argparse
import dataclasses
from pathlib import Path

import cv2
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def midas_worker(proc_idx: int, input_queue: mp.Queue, output_queue: mp.Queue, use_half: bool = True):
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512", pretrained=True)
        if use_half:
            midas = midas.half()
        midas = midas.to(DEVICE)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.beit512_transform

        with torch.no_grad():
            while True:
                inp: tuple[str, Path] | None = input_queue.get()
                if inp is None:
                    output_queue.put(None)
                    break

                image_stem, image = inp
                image = Image.open(image)

                # Depth推定
                batch = transform(np.asarray(image))
                if use_half:
                    batch = batch.half()
                depth = midas(batch.to(DEVICE)).cpu()  # [1, height, width]
                if not torch.isfinite(depth).all():
                    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

                output_queue.put((image_stem, image, depth))
    except Exception as e:
        print(f"midas_worker: {e}")
        output_queue.put(None)


def nvds_worker(proc_idx: int, input_queue: mp.Queue, output_queue: mp.Queue, use_half: bool = True):
    try:
        _mean = [0.485, 0.456, 0.406]
        _std = [0.229, 0.224, 0.225]
        checkpoint = torch.load("./NVDS/NVDS_checkpoints/NVDS_Stabilizer.pth", map_location="cpu")
        nvds = NVDS()
        nvds = torch.nn.DataParallel(nvds, device_ids=[0]).cuda()
        nvds.load_state_dict(checkpoint)
        if use_half:
            nvds = nvds.half()
        nvds = nvds.to(DEVICE)
        nvds.eval()

        def generate_rgbd(image: Image.Image, depth: Tensor) -> Tensor:
            # 正規化
            depth_min = torch.min(depth)
            depth_max = torch.max(depth)
            if depth_max - depth_min > torch.finfo(torch.float).eps:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = torch.zeros(depth.shape, dtype=depth.dtype)

            # rgbdフレーム化
            rgb = np.asarray(image.resize(depth.shape[1:][::-1], Image.Resampling.BICUBIC)) / 255.0
            rgb = (rgb - _mean) / _std
            rgb = torch.Tensor(np.transpose(rgb, (2, 0, 1)))
            if use_half:
                rgb = rgb.half()
            rgbd = torch.cat([rgb, depth], dim=0)
            return rgbd

        seq_len = 4
        seq_frames = []
        cuda_seq_frames = None
        seq_images = []
        image_stems = []
        last_depth = None
        is_end = False
        image = None
        with torch.no_grad():
            while True:
                inp: tuple[str, Image.Image, Tensor] | None = input_queue.get()
                if inp is None:
                    if is_end:
                        # 終了
                        output_queue.put(None)
                        break
                    else:
                        # 最後のフレームがきたら最終フレームを繰り返す
                        is_end = True
                        for _ in range(seq_len - 1):
                            input_queue.put(("", image, last_depth))
                        input_queue.put(None)
                        continue

                # rgbとdepthを結合
                image_stem, image, depth = inp
                last_depth = depth
                rgbd = generate_rgbd(image, depth).unsqueeze(0)  # [1, channel, height, width]

                if len(seq_frames) == 0:
                    # 最初のフレームを繰り返す
                    seq_frames = [rgbd] * (seq_len + 1)
                    seq_images = [image] * (seq_len + 1)
                    image_stems = [image_stem] * (seq_len + 1)
                    continue
                elif len(seq_frames) < seq_len * 2 - 1:
                    # 推定に必要なフレーム数分貯める
                    seq_frames.append(rgbd)
                    seq_images.append(image)
                    image_stems.append(image_stem)
                    if len(seq_frames) == seq_len * 2 - 1:
                        # 貯まったらGPUに転送
                        fwd_frames = torch.cat(seq_frames[:seq_len], dim=0)  # [seq_len, channel, height, width]
                        bwd_frames = torch.cat(seq_frames[seq_len - 1 :][::-1], dim=0)
                        cuda_seq_frames = torch.cat(
                            [fwd_frames.unsqueeze(0), bwd_frames.unsqueeze(0)], dim=0
                        )  # [2, seq_len, channel, height, width]
                        if use_half:
                            cuda_seq_frames = cuda_seq_frames.half()
                        cuda_seq_frames = cuda_seq_frames.to(DEVICE)
                    continue

                # 次のフレームにシフト
                # [[0, 1, 2, 3]  -> [[1, 2, 3, 4]
                #  [6, 5, 4, 3]]     [7, 6, 5, 4]]
                cuda_seq_frames[1, 1:] = cuda_seq_frames[1, :-1].clone()  # bwd
                cuda_seq_frames[1, 0] = rgbd[0]
                cuda_seq_frames[0, :-1] = cuda_seq_frames[0, 1:].clone()  # fwd
                cuda_seq_frames[0, -1] = cuda_seq_frames[1, -1]
                seq_images = seq_images[1:] + [image]
                image_stems = image_stems[1:] + [image_stem]

                # NVDS
                preds = nvds(cuda_seq_frames)
                depth = (preds[0] + preds[1]) / 2

                depth = depth.squeeze().cpu().numpy()
                output_queue.put((image_stems[seq_len - 1], seq_images[seq_len - 1], depth))
    except Exception as e:
        print(f"nvds_worker: {e}")
        output_queue.put(None)


def create_sbs_worker(
    proc_idx: int, input_queue: mp.Queue, output_queue: mp.Queue, depth_dir: Path | None, config: StereoConfig
) -> np.ndarray:
    """SBS画像作成"""

    def depth_to_uint16(depth: np.ndarray) -> np.ndarray:
        bits = np.uint16(0).nbytes
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2 ** (8 * bits)) - 1
        if depth_max - depth_min > np.finfo("float").eps:
            depth = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros(depth.shape, dtype=depth.dtype)
        return depth.astype(np.uint16)

    def invert_map(f):
        """逆remap"""
        inv = np.zeros_like(f)
        inv[:, :, 1], inv[:, :, 0] = np.indices(f.shape[:2])
        p = np.copy(inv)
        for i in range(10):
            cor = inv - cv2.remap(f, p, None, interpolation=cv2.INTER_LINEAR)
            rate = max(0.05, 1.0 - i * 0.2)  # 徐々に収束幅を狭める
            p += cor * rate
        return p

    try:
        while True:
            inp: tuple[str, Image.Image, np.ndarray] | None = input_queue.get()
            if inp is None:
                output_queue.put(None)
                break

            # 画像取得
            image_stem, image, depth = inp
            depth = depth_to_uint16(depth)
            if depth_dir is not None:
                depth_path = depth_dir / f"{image_stem}.png"
                Image.fromarray(depth).save(depth_path, compress_level=3)
            image = np.asarray(image)[:, :, ::-1]
            height, width = image.shape[:2]

            if (depth.shape[0] != height) or (depth.shape[1] != width):
                depth = cv2.resize(
                    depth, (width, height), interpolation=cv2.INTER_LINEAR
                )  # CUBICだとアンダーシュート・オーバーシュートが発生する

            # 境界部のdepth推定エラーをごまかす
            depth2 = cv2.dilate(depth, np.ones((7, 7), np.uint8), iterations=5)
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
            left_img = cv2.remap(image, left_map_inv, None, interpolation=cv2.INTER_LINEAR)
            right_img = cv2.remap(image, right_map_inv, None, interpolation=cv2.INTER_LINEAR)
            frame = np.hstack((left_img, right_img))[:, :, ::-1]
            if config.is_half:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            output_queue.put((image_stem, frame))
    except Exception as e:
        print(f"create_sbs_worker: {e}")
        output_queue.put(None)


def run(
    src_path: Path,
    jpg_dir: Path,
    depth_dir: Path | None,
    output_path: Path,
    frame_interpolated: int,
    config: StereoConfig,
):
    """3D動画生成"""
    # worker起動
    queue_size = 5
    midas_queue = mp.Queue()
    nvds_queue = mp.Queue(maxsize=queue_size)
    sbs_queue = mp.Queue(maxsize=queue_size)
    output_queue = mp.Queue(maxsize=queue_size)
    midas_proc = mp.spawn(midas_worker, args=(midas_queue, nvds_queue, True), nprocs=1, join=False, daemon=True)
    nvds_proc = mp.spawn(nvds_worker, args=(nvds_queue, sbs_queue, True), nprocs=1, join=False, daemon=True)
    sbs_proc = mp.spawn(
        create_sbs_worker,
        args=(sbs_queue, output_queue, depth_dir, config),
        nprocs=1,
        join=False,
        daemon=True,
    )

    # ソース動画情報取得
    video_info = skvideo.io.ffprobe(src_path)["video"]
    r_frame_rate = video_info["@r_frame_rate"]
    a, b = r_frame_rate.split("/")
    r_frame_rate = f"{int(a)*frame_interpolated}/{b}"
    print(r_frame_rate)
    codec_name = video_info["@codec_name"]
    pix_fmt = video_info["@pix_fmt"]

    # 動画Writer作成
    output_path.parent.mkdir(exist_ok=True, parents=True)
    writer = skvideo.io.FFmpegWriter(
        output_path,
        inputdict={
            "-r": r_frame_rate,
        },
        outputdict={"-vcodec": codec_name, "-pix_fmt": pix_fmt, "-r": r_frame_rate},
    )

    # 入力
    jpg_paths = list(sorted(jpg_dir.glob("*.jpg")))
    for jpg_path in jpg_paths:
        midas_queue.put((jpg_path.stem, jpg_path))
    midas_queue.put(None)

    # 出力
    with tqdm(jpg_paths) as pbar:
        for jpg_path in pbar:
            pbar.set_postfix({"nvds": nvds_queue.qsize(), "sbs": sbs_queue.qsize()})
            out: tuple[int, np.ndarray] | None = output_queue.get()
            if out is None:
                break

            image_stem, frame = out
            if jpg_path.stem != image_stem:
                print(f"frame {jpg_path.stem} is missing, {image_stem} returned")
            writer.writeFrame(frame)

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", help="src movie path")
    parser.add_argument("-i", "--image_path", default=None, help="folder with input images")
    parser.add_argument("-o", "--output_path", default=None, help="output movie path")
    parser.add_argument("-d", "--depth_path", default=None, help="folder with depth images")
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
        depth_dir.mkdir(exist_ok=True, parents=True)
    else:
        depth_dir = None
    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = src_path.parent.parent / "3d" / src_path.name
    config = CONFIG[args.target_device]
    run(src_path, jpg_dir, depth_dir, output_path, args.frame_interpolated, config)
