from pathlib import Path

import fire
import numpy as np
import torch
from numpy.typing import DTypeLike, NDArray
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from nvds.full_model import NVDS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_depth(path: Path, depth: NDArray, dtype: DTypeLike = np.uint8):
    """Depth画像保存

    :param path Path: 保存パス
    :param depth NDArray: Depth画像
    :param dtype DTypeLike: 保存型(np.uint8 or np.uint16)
    """
    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_min = depth.min()
    depth_max = depth.max()

    bits = dtype(0).nbytes
    max_val = (2 ** (8 * bits)) - 1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.dtype)

    image = Image.fromarray(out.astype(dtype))
    if path.suffix == ".png":
        image.save(path, compress_level=3)
    else:
        image.save(path)


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

    def _infer_nvds(self, seq_frames: list[Tensor]) -> Tensor:
        seq_frames = torch.cat(seq_frames, dim=0).unsqueeze(0)  # [1, seq_len, channel, height, width]
        if self._use_half:
            seq_frames = seq_frames.half()
        pred = self._nvds(seq_frames.to(DEVICE)).squeeze(1)
        return pred

    def _generate_rgbd(self, path: Path) -> Tensor:
        frame = Image.open(path)

        # Depth推定
        batch = self._midas_transform(np.asarray(frame))
        if self._use_half:
            batch = batch.half()
        depth = self._midas(batch.to(DEVICE)).cpu()  # [1, height, width]
        depth = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))

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
            for i in tqdm(range(len(frame_paths))):
                # 次フレーム読み込み
                next_idx = i + self._seq_len - 1
                if next_idx < len(frame_paths):
                    rgbd = self._generate_rgbd(frame_paths[next_idx]).unsqueeze(0)
                else:
                    rgbd = seq_frames[-1]
                seq_frames = seq_frames[1:] + [rgbd]

                # NVDS
                fwd = self._infer_nvds(seq_frames[: self._seq_len])
                bwd = self._infer_nvds(seq_frames[self._seq_len - 1 :][::-1])
                depth = (fwd + bwd) / 2

                # 保存
                dst_path = dst_dir / f"{frame_paths[i].stem}.png"
                depth = depth.squeeze().cpu().numpy()
                save_depth(dst_path, depth, np.uint16)


def main(jpg_dir: str, dst_dir: str = None, use_half: bool = True):
    """MiDaS実行

    :param str jpg_dir: 入力フォルダ
    :param str dst_dir: 保存フォルダ, defaults to None
    :param bool use_half: 半精度推定を使うか, defaults to True
    """
    jpg_dir = Path(jpg_dir)
    if dst_dir is not None:
        dst_dir = Path(dst_dir)
    else:
        dst_dir = jpg_dir.parent.parent / "depth" / jpg_dir.name
    dst_dir.mkdir(exist_ok=True, parents=True)

    # モデル準備
    nvds = NVDSModel(seq_len=4, use_half=use_half)

    # 処理開始
    nvds.infer(dst_dir, sorted(jpg_dir.glob("*.jpg")))


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    fire.Fire(main)
