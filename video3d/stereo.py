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
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Depth境界を太らせるフィルタサイズ
        self._morph_iter = 3
        self._max_gap_thr = 10  # Depth境界だとみなす最大閾値
        self._min_gap_thr = 100  # Depth境界だとみなす最小閾値
        self._blur_size = (11, 11)  # 距離画像の平滑化サイズ
        self._disp_diff_cut = 0.5  # 視差で手前オブジェクトを判定するための閾値

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

    def _complement_blank(self, map_x: NDArray, map_x_inv: NDArray, direction: int = 1) -> NDArray:
        # 視差情報からremap元がない画素を特定
        height, width = map_x.shape[:2]
        _, my = np.meshgrid(range(width), range(height))
        dmx1 = np.floor(map_x).astype(int)
        dmx2 = np.ceil(map_x).astype(int)
        dmx1[dmx1 < 0] = 0
        dmx1[dmx1 >= width] = width - 1
        dmx2[dmx2 < 0] = 0
        dmx2[dmx2 >= width] = width - 1
        blank = np.ones((height, width), dtype=np.uint8)
        blank[my, dmx1] = 0
        blank[my, dmx2] = 0

        # 空白画素をいい感じにまとめる
        kernel1 = np.ones((3, 3), np.uint8)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blank = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel1, iterations=1)
        blank = cv2.morphologyEx(blank, cv2.MORPH_OPEN, kernel2, iterations=1)

        # 最初に膨張した分不可視領域を増やす
        ksize = self._morph_kernel.shape[1]
        kernel = np.ones((1, ksize), np.uint8)
        if direction == 1:
            kernel[:, ksize // 2 + 1 :] = 0
        else:
            kernel[:, : ksize // 2] = 0
        blank = cv2.dilate(blank, kernel, iterations=self._morph_iter)
        blank = blank == 1

        # 空白画素を補完
        map_x_inv = map_x_inv.copy()
        if direction == 1:
            rng = range(1, map_x_inv.shape[1])
        else:
            rng = range(map_x_inv.shape[1] - 2, -1, -1)
        for x in rng:
            # 空白画素（≒ 隠れてる背景画素）を平坦化
            map_x_inv[:, x][blank[:, x]] = map_x_inv[:, x - direction][blank[:, x]]

        return map_x_inv, blank

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
        if self._morph_kernel is not None:
            depth_dil = depth.copy()
            max_gap = (depth_dil.max() - depth_dil.min()) / self._max_gap_thr
            min_gap = (depth_dil.max() - depth_dil.min()) / self._min_gap_thr
            depth2 = cv2.dilate(depth_dil, self._morph_kernel, iterations=self._morph_iter)
            gap = depth2 - depth
            gap[gap < min_gap] = min_gap
            gap[gap > max_gap] = max_gap
            gap = 1 / gap
            gap = (gap - gap.min()) / (gap.max() - gap.min())
            depth_dil = depth_dil * gap + depth2 * (1 - gap)
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

        # 境界部のdepth推定の曖昧さをごまかす
        # 手前のオブジェクトを優先
        if self._disp_diff_cut > 0:
            disp_diff = disp_px - np.insert(disp_px, 0, disp_px[:, 0], axis=1)[:, :-1]
            right_cut = disp_diff < -self._disp_diff_cut
            left_cut = disp_diff > self._disp_diff_cut
            left_disp_px = disp_px.copy()
            right_disp_px = disp_px.copy()
            for x in range(width - 2, -1, -1):
                right_disp_px[:, x][right_cut[:, x]] = right_disp_px[:, x + 1][right_cut[:, x]]
            for x in range(1, width):
                left_disp_px[:, x][left_cut[:, x]] = left_disp_px[:, x - 1][left_cut[:, x]]

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

        # 境界部のdepth推定の曖昧さをごまかす
        if self._morph_kernel is not None:
            left_map_inv, left_blank = self._complement_blank(left_map_x, left_map_inv, direction=1)
            right_map_inv, right_blank = self._complement_blank(right_map_x, right_map_inv, direction=-1)

        # 画像生成
        left_img = cv2.remap(image, left_map_inv, None, interpolation=cv2.INTER_LINEAR)
        right_img = cv2.remap(image, right_map_inv, None, interpolation=cv2.INTER_LINEAR)
        frame = np.hstack((left_img, right_img))
        blank = np.hstack((left_blank.astype(np.uint8) * 255, right_blank.astype(np.uint8) * 255))
        if conf.is_half:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            blank = cv2.resize(blank, (width, height), interpolation=cv2.INTER_LINEAR)
            blank = (blank > 0).astype(np.uint8) * 255

        return frame[:, :, ::-1], blank
