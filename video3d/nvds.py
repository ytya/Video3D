import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from nvds.full_model import NVDS
from video3d.util import NDArrayUint8


class NVDSModel:
    def __init__(
        self,
        model_path: str = "./NVDS/NVDS_checkpoints/NVDS_Stabilizer.pth",
        use_half: bool = True,
        device: str | torch.device = "cuda",
    ):
        self.seq_len = 4
        self._use_half = use_half
        self._device = torch.device(device)

        # NVDS
        self._mean = torch.as_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self._std = torch.as_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        checkpoint = torch.load(model_path, map_location="cpu")
        nvds = NVDS()
        nvds = torch.nn.DataParallel(nvds, device_ids=[0])
        nvds.load_state_dict(checkpoint)
        if use_half:
            nvds = nvds.half()
        self._nvds = nvds.eval().to(self._device)

        # バッファ
        self._first_seq_frames: list[Tensor] = []
        self._tensor_seq_frames: Tensor | None = None
        self._return_idx = -1

    def reset(self) -> None:
        self._first_seq_frames = []
        self._tensor_seq_frames = None
        self._return_idx = -1

    def _generate_rgbd(self, image: Tensor, depth: Tensor) -> Tensor:
        # 正規化
        depth = depth.squeeze()
        depth_min = torch.min(depth)
        depth_max = torch.max(depth)
        if depth_max - depth_min > torch.finfo(torch.float).eps:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = torch.zeros(depth.shape, dtype=depth.dtype)

        # rgbdフレーム化
        image = F.resize(image, size=depth.shape, interpolation=InterpolationMode.BICUBIC, antialias=True) / 255.0
        image = (image - self._mean) / self._std
        rgbd = torch.cat([image, depth.unsqueeze(0)], dim=0)
        if self._use_half:
            rgbd = rgbd.half()
        return rgbd

    @torch.no_grad()
    def __call__(self, image: NDArrayUint8 | Tensor, depth: NDArray | Tensor) -> tuple[int, Tensor | None]:
        """Depth推定

        :param NDArrayUint8|Tensor image: 入力画像
        :param NDArray|Tensor depth: 入力Depth画像
        :return tuple[int, Tensor|None]: (出力インデックス, 推定Depth画像)
        """
        if isinstance(image, np.ndarray):
            image = torch.tensor(image.transpose((2, 0, 1)))
        if isinstance(depth, np.ndarray):
            depth = torch.tensor(depth)
        rgbd = self._generate_rgbd(image, depth)  # [4, height, width]

        if len(self._first_seq_frames) < self.seq_len * 2 - 1:
            # 推定に必要なフレーム数分貯める
            self._first_seq_frames.append(rgbd.unsqueeze(0))
            if len(self._first_seq_frames) < self.seq_len * 2 - 1:
                # 貯まっていなかったら結果なしで返す
                return -1, None

            # 貯まったらTensorに変換
            fwd_frames = torch.cat(self._first_seq_frames[: self.seq_len], dim=0)  # [seq_len, 4, height, width]
            bwd_frames = torch.cat(self._first_seq_frames[self.seq_len - 1 :][::-1], dim=0)
            self._tensor_seq_frames = torch.cat(
                [fwd_frames.unsqueeze(0), bwd_frames.unsqueeze(0)], dim=0
            )  # [2, seq_len, 4, height, width]
            if self._use_half:
                self._tensor_seq_frames = self._tensor_seq_frames.half()
            self._tensor_seq_frames = self._tensor_seq_frames.to(self._device)
        else:
            # 次のフレームにシフト
            # [[0, 1, 2, 3]  -> [[1, 2, 3, 4]
            #  [6, 5, 4, 3]]     [7, 6, 5, 4]]
            self._tensor_seq_frames[1, 1:] = self._tensor_seq_frames[1, :-1].clone()  # bwd
            self._tensor_seq_frames[1, 0] = rgbd
            self._tensor_seq_frames[0, :-1] = self._tensor_seq_frames[0, 1:].clone()  # fwd
            self._tensor_seq_frames[0, -1] = self._tensor_seq_frames[1, -1]

        # Depth推定
        preds = self._nvds(self._tensor_seq_frames)
        depth = (preds[0] + preds[1]) / 2
        depth = depth.squeeze()

        self._return_idx += 1
        return self._return_idx, depth
