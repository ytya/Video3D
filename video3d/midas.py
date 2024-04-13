import torch
from torch import Tensor

from .util import NDArrayUint8


class MiDaSModel:
    def __init__(self, model_name: str = "DPT_BEiT_L_512", use_half: bool = True, device: str | torch.device = "cuda"):
        self._use_half = use_half
        self._device = torch.device(device)

        midas = torch.hub.load("intel-isl/MiDaS", model_name, pretrained=True)
        if use_half:
            midas = midas.half()
        self._midas = midas.eval().to(self._device)

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
        pred = self._midas(batch.to(self._device))  # [1, height, width]
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
