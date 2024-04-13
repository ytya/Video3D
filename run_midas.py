from pathlib import Path

import fire
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from util import NDArrayUint8, save_depth

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MiDaSModel:
    def __init__(self, model_name: str = "DPT_BEiT_L_512", use_half: bool = True):
        self._use_half = use_half

        midas = torch.hub.load("intel-isl/MiDaS", model_name, pretrained=True)
        if use_half:
            midas = midas.half()
        self._midas = midas.eval().to(DEVICE)

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if "DPT" in model_name:
            if "512" in model_name:
                self._transform = transforms.beit512_transform
            elif ("Swin" in model_name) or ("384" in model_name):
                self._transform = transforms.swin384_transform
            elif ("Swin" in model_name) or ("256" in model_name):
                self._transform = transforms.swin256_transform
            elif "256" in model_name:
                self._transform = transforms.small_transform
            elif "LeViT" in model_name:
                self._transform = transforms.levit_transform
            else:
                self._transform = transforms.dpt_transform
        elif "256" in model_name:
            self._transform = transforms.small_transform
        else:
            self._transform = transforms.default_transform

    @torch.no_grad()
    def __call__(self, image: NDArrayUint8, inter_mode: str | None = "bilinear") -> Tensor:
        batch = self._transform(image)
        if self._use_half:
            batch = batch.half()
        pred = self._midas(batch.to(DEVICE))  # [1, height, width]
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)  # inf対策

        if inter_mode is not None:
            # 元画像のサイズに合わせる
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=image.shape[:2], mode=inter_mode, align_corners=False
            ).squeeze()
            pred[pred < 0] = 0  # 補間によってマイナスになることがあるので対策
        else:
            pred = pred.squeeze()
        return pred  # [height, width]


def run(image_dir: Path, depth_dir: Path, use_half: bool = True):
    # モデル準備
    midas = MiDaSModel("DPT_BEiT_L_512", use_half=use_half)

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

    run(image_dir, depth_dir, use_half)


if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    fire.Fire(main)
