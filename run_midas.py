from pathlib import Path

import fire
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from video3d.midas import MiDaSModel
from video3d.util import save_depth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _run(image_dir: Path, depth_dir: Path, use_half):
    # モデル準備
    midas = MiDaSModel("DPT_BEiT_L_512", use_half=use_half, device=DEVICE)

    # 処理開始
    depth_dir.mkdir(exist_ok=True, parents=True)
    jpg_paths = list(sorted(image_dir.glob("*.jpg")))
    for jpg_path in tqdm(jpg_paths):
        # Depth推定
        image = np.asarray(Image.open(jpg_path))
        depth = midas(image, inter_mode="bilinear").cpu().numpy()

        # 保存
        dst_path = depth_dir / f"{jpg_path.stem}.png"
        save_depth(dst_path, depth, np.uint16)


def main(image_dir: str, depth_dir: str = None, use_half: bool = True):
    """MiDaS実行

    :param str image_dir: 入力フォルダ
    :param str depth_dir: 保存フォルダ, defaults to None
    :param bool use_half: 半精度推定を使うか, defaults to True
    """
    image_dir = Path(image_dir)
    if depth_dir is not None:
        depth_dir = Path(depth_dir)
    else:
        depth_dir = image_dir.parent.parent / "depth" / image_dir.name

    _run(image_dir, depth_dir, use_half)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    fire.Fire(main)
