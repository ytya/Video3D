from pathlib import Path

import fire
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from video3d.midas import MiDaSModel
from video3d.nvds import NVDSModel
from video3d.util import NDArrayUint8, save_depth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _run(image_dir: Path, depth_dir: Path | None, dst_depth_dir: Path, use_half):
    dst_depth_dir.mkdir(exist_ok=True, parents=True)

    # モデル準備
    nvds = NVDSModel(use_half=use_half, device=DEVICE)
    midas = None
    if depth_dir is None:
        midas = MiDaSModel("DPT_BEiT_L_512", use_half=use_half, device=DEVICE)

    @torch.no_grad()
    def get_depth(image_path: Path, image: NDArrayUint8) -> Tensor:
        if midas is not None:
            return midas(image, inter_mode=None)
        else:
            depth_path = depth_dir / f"{image_path.stem}.png"
            return torch.tensor(np.asarray(Image.open(depth_path)))

    # 処理開始
    image_paths = list(sorted(image_dir.glob("*.jpg")))
    extra_image_paths = (
        ([image_paths[0]] * (nvds.seq_len - 1)) + image_paths + ([image_paths[-1]] * (nvds.seq_len - 1))
    )  # 最初と最後のフレームは推定用に外挿
    for image_path in tqdm(extra_image_paths):
        # 読み込み
        image = np.asarray(Image.open(image_path))
        depth = get_depth(image_path, image).cpu()

        # 推定
        idx, depth = nvds(image, depth)
        if depth is None:
            # 推定に必要なフレームが貯まるまではdepthが返ってこない
            continue

        # 保存
        depth = depth.cpu().numpy()
        dst_path = dst_depth_dir / f"{image_paths[idx].stem}.png"
        save_depth(dst_path, depth, np.uint16)


def main(image_dir: str, depth_dir: str = None, dst_depth_dir: str = None, use_half: bool = True):
    """MiDaS実行

    :param str image_dir: 入力画像フォルダ
    :param str depth_dir: 入力Depthフォルダ(Noneの場合はMiDaSで推定), defaults to None
    :param str dst_depth_dir: 保存フォルダ, defaults to None
    :param bool use_half: 半精度推定を使うか, defaults to True
    """
    image_dir = Path(image_dir)
    if depth_dir is not None:
        depth_dir = Path(depth_dir)
    if dst_depth_dir is not None:
        dst_depth_dir = Path(dst_depth_dir)
    elif depth_dir is not None:
        dst_depth_dir = depth_dir
    else:
        dst_depth_dir = image_dir.parent.parent / "depth" / image_dir.name

    _run(image_dir, depth_dir, dst_depth_dir, use_half)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    fire.Fire(main)
