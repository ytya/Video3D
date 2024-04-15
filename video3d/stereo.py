import dataclasses

import cv2
import numpy as np

from .util import NDArray, NDArrayUint, NDArrayUint8


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


SCONFIGS = {
    "projector": StereoConfig(
        dist_screen=2200, dist_forward=818, dist_back=3176, pd=65, screen_w=2037, image_w=1920, is_half=True
    ),
    "mobile": StereoConfig(
        dist_screen=250, dist_forward=18, dist_back=21, pd=65, screen_w=250, image_w=1920, is_half=False
    ),
}


class SBSCreator:
    """Side by Side画像作成"""

    def __init__(self, config: StereoConfig):
        self._config = config
        self._gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._gap_iter = 3
        self._gap_thr = 4
        self._blur_size = (11, 11)

    def _invert_map(self, f: NDArray) -> NDArray:
        """逆remap"""
        inv = np.zeros_like(f)
        inv[:, :, 1], inv[:, :, 0] = np.indices(f.shape[:2])
        p = np.copy(inv)
        for i in range(10):
            cor = inv - cv2.remap(f, p, None, interpolation=cv2.INTER_LINEAR)
            rate = max(0.05, 1.0 - i * 0.2)  # 徐々に収束幅を狭める
            p += cor * rate
        return p

    def __call__(self, image: NDArrayUint8, depth: NDArrayUint) -> np.ndarray:
        height, width = image.shape[:2]
        image = image[:, :, ::-1]  # 関数内はBGR
        conf = self._config

        if (depth.shape[0] != height) or (depth.shape[1] != width):
            depth = cv2.resize(
                depth, (width, height), interpolation=cv2.INTER_LINEAR
            )  # CUBICだとアンダーシュート・オーバーシュートが発生する

        # 境界部のdepth推定エラーをごまかす
        # 輪郭部 ≒ depthギャップが大きい部分を拡張
        if self._gap_kernel is not None:
            depth_dil = depth.copy()
            depth2 = cv2.dilate(depth_dil, self._gap_kernel, iterations=self._gap_iter)
            gap = depth2 - depth_dil > (depth_dil.max() - depth_dil.min()) / self._gap_thr
            gap = cv2.dilate(gap.astype(np.uint8), self._gap_kernel, iterations=self._gap_iter)
            depth_dil[gap == 1] = depth2[gap == 1]
        else:
            depth_dil = depth

        # 距離計算 (depth = 1 / 距離)
        depth_max = int(depth_dil.max())
        depth_min = int(depth_dil.min())
        scale = (1 / conf.dist_max - 1 / conf.dist_min) / (depth_min - depth_max)
        shift = 1 / conf.dist_min - depth_max * scale
        dist = 1 / (depth_dil * scale + shift)
        dist = cv2.GaussianBlur(dist, self._blur_size, 0)

        # 視差計算
        dist -= conf.dist_screen  # スクリーンより手前がマイナス
        disp = dist * conf.pd / (conf.dist_screen + dist)  # プラスならR画素がL画素より右、マイナスなら逆
        disp_px = disp / conf.screen_w * conf.image_w / 2  # 視差

        # SBS用remap計算
        left_map_x, left_map_y = np.meshgrid(range(width), range(height))
        right_map_x = left_map_x.copy()
        right_map_y = left_map_y.copy()

        left_map_x = left_map_x - disp_px
        right_map_x = right_map_x + disp_px
        left_map = np.stack((left_map_x, left_map_y), axis=2).astype(np.float32)
        right_map = np.stack((right_map_x, right_map_y), axis=2).astype(np.float32)
        left_map_inv = self._invert_map(left_map)
        right_map_inv = self._invert_map(right_map)

        # 画像生成
        left_img = cv2.remap(image, left_map_inv, None, interpolation=cv2.INTER_LINEAR)
        right_img = cv2.remap(image, right_map_inv, None, interpolation=cv2.INTER_LINEAR)
        frame = np.hstack((left_img, right_img))[:, :, ::-1]
        if conf.is_half:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        return frame[:, :, ::-1]
