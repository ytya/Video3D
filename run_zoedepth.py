import torch
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.typing import NDArray, DTypeLike
from tqdm import tqdm
import fire
import math

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


def run(jpg_dir: Path, dst_dir: Path, use_half: bool = True, model_name: str = "ZoeD_NK"):
    """ZoeDepth実行

    :param str jpg_dir: 入力フォルダ
    :param str dst_dir: 保存フォルダ, defaults to None
    :param bool use_half: 半精度推定を使うか, defaults to True
    :param str model_name: モデル名(ZoeD_N/ZoeD_K/ZoeD_NK), defaults to "ZoeD_NK"
    """
    torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

    model_zoe = torch.hub.load("isl-org/ZoeDepth", model_name, pretrained=True, config_mode="eval").to(DEVICE)
    model_zoe.eval()

    dst_dir.mkdir(exist_ok=True, parents=True)
    jpg_paths = list(jpg_dir.glob("*.jpg"))
    for jpg_path in tqdm(jpg_paths):
        image = Image.open(jpg_path)
        depth = model_zoe.infer_pil(image)
        if (depth < 0).any():
            depth[depth < 0] = depth

        depth_min = depth.min()
        depth_max = depth.max()

        vmax = int(depth_max) // 10 + 10

        dst_path = dst_dir / f"{jpg_path.stem}_{depth_min:.1f}_{depth_max:.1f}.png"
        Image.fromarray((depth * 65535 / vmax).astype(np.uint16)).save(dst_path)


def main(jpg_dir: str, dst_dir: str = None, model_name: str = "ZoeD_NK"):
    """MiDaS実行

    :param str jpg_dir: 入力フォルダ
    :param str dst_dir: 保存フォルダ, defaults to None
    :param str model_name: モデル名(ZoeD_N/ZoeD_K/ZoeD_NK), defaults to "ZoeD_NK"
    """
    jpg_dir = Path(jpg_dir)
    if dst_dir is not None:
        dst_dir = Path(dst_dir)
    else:
        dst_dir = jpg_dir.parent.parent / "depth" / jpg_dir.name

    run(jpg_dir, dst_dir, model_name=model_name)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # fire.Fire(main)
    main(
        r"H:\work\Video3D\src_jpg\BD_ROM.Title1.chapter22",
        r"H:\work\Video3D\depth\BD_ROM.Title1.chapter22_NK",
        model_name="ZoeD_K",
    )
