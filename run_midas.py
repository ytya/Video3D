from pathlib import Path

import fire
import numpy as np
import torch
from numpy.typing import DTypeLike, NDArray
from PIL import Image
from tqdm import tqdm

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

    Image.fromarray(out.astype(dtype)).save(path)


def run(jpg_dir: Path, dst_dir: Path, use_half: bool = True):
    # モデル準備
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_BEiT_L_512", pretrained=True).to(DEVICE)
    midas.eval()
    if use_half:
        midas = midas.half()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.beit512_transform

    # 処理開始
    dst_dir.mkdir(exist_ok=True, parents=True)
    jpg_paths = list(sorted(jpg_dir.glob("*.jpg")))
    with torch.no_grad():
        for jpg_path in tqdm(jpg_paths):
            # Depth推定
            image = np.asarray(Image.open(jpg_path))
            batch = transform(image).to(DEVICE)
            if use_half:
                batch = batch.half()
            pred = midas(batch)
            pred = (
                torch.nn.functional.interpolate(
                    pred.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            pred[pred < 0] = 0  # bicubicによってマイナスになることがあるので対策

            # 保存
            dst_path = dst_dir / f"{jpg_path.stem}.png"
            save_depth(dst_path, pred, np.uint16)


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

    run(jpg_dir, dst_dir, use_half)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    fire.Fire(main)
