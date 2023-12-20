import torch
from pathlib import Path
from PIL import Image
import numpy as np
from numpy.typing import NDArray, DTypeLike
from tqdm import tqdm
import fire
import math
from functools import lru_cache
from scipy.ndimage import gaussian_filter

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


@lru_cache
def create_filter(M: int, N: int):
    filters = {
        name: np.zeros((M, N))
        for name in [
            "right",
            "left",
            "top",
            "bottom",
            "top_right",
            "top_left",
            "bottom_right",
            "bottom_left",
            "filter",
        ]
    }
    for i in range(M):
        for j in range(N):
            x_value = 0.998 * np.cos((abs(M / 2 - i) / M) * np.pi) ** 2
            y_value = 0.998 * np.cos((abs(N / 2 - j) / M) * np.pi) ** 2

            if j > N / 2:
                filters["right"][i, j] = x_value
            else:
                filters["right"][i, j] = x_value * y_value

            if j < N / 2:
                filters["left"][i, j] = x_value
            else:
                filters["left"][i, j] = x_value * y_value

            if i < M / 2:
                filters["top"][i, j] = y_value
            else:
                filters["top"][i, j] = x_value * y_value

            if i > M / 2:
                filters["bottom"][i, j] = y_value
            else:
                filters["bottom"][i, j] = x_value * y_value

            if j > N / 2 and i < M / 2:
                filters["top_right"][i, j] = 0.998
            elif j > N / 2:
                filters["top_right"][i, j] = x_value
            elif i < M / 2:
                filters["top_right"][i, j] = y_value
            else:
                filters["top_right"][i, j] = x_value * y_value

            if j < N / 2 and i < M / 2:
                filters["top_left"][i, j] = 0.998
            elif j < N / 2:
                filters["top_left"][i, j] = x_value
            elif i < M / 2:
                filters["top_left"][i, j] = y_value
            else:
                filters["top_left"][i, j] = x_value * y_value

            if j > N / 2 and i > M / 2:
                filters["bottom_right"][i, j] = 0.998
            elif j > N / 2:
                filters["bottom_right"][i, j] = x_value
            elif i > M / 2:
                filters["bottom_right"][i, j] = y_value
            else:
                filters["bottom_right"][i, j] = x_value * y_value

            if j < N / 2 and i > M / 2:
                filters["bottom_left"][i, j] = 0.998
            elif j < N / 2:
                filters["bottom_left"][i, j] = x_value
            elif i > M / 2:
                filters["bottom_left"][i, j] = y_value
            else:
                filters["bottom_left"][i, j] = x_value * y_value

            filters["filter"][i, j] = x_value * y_value
    return filters


def infer(image: Image.Image, model: torch.nn.Module):
    low_depth = model.infer_pil(image)
    low_scaled_depth = 2**16 - (low_depth - np.min(low_depth)) * 2**16 / (np.max(low_depth) - np.min(low_depth))
    low_depth_map_image = Image.fromarray((0.999 * low_scaled_depth).astype("uint16"))

    # filter作成
    img = np.asarray(image)
    num_x, num_y = [2, 2]
    M = img.shape[0] // num_x
    N = img.shape[1] // num_y
    filters = create_filter(M, N)

    # 高解像度推定
    compiled_tiles = np.zeros(img.shape[:2])
    x_coords = list(range(0, img.shape[0], img.shape[0] // num_x))[:num_x]
    y_coords = list(range(0, img.shape[1], img.shape[1] // num_y))[:num_y]
    x_coords_between = list(range((img.shape[0] // num_x) // 2, img.shape[0], img.shape[0] // num_x))[: num_x - 1]
    y_coords_between = list(range((img.shape[1] // num_y) // 2, img.shape[1], img.shape[1] // num_y))[: num_y - 1]
    x_coords_all = x_coords + x_coords_between
    y_coords_all = y_coords + y_coords_between

    for x in x_coords_all:
        for y in y_coords_all:
            depth = model.infer_pil(Image.fromarray(np.uint8(img[x : x + M, y : y + N])))
            scaled_depth = 2**16 - (depth - np.min(depth)) * 2**16 / (np.max(depth) - np.min(depth))

            if y == min(y_coords_all) and x == min(x_coords_all):
                selected_filter = filters["top_left"]
            elif y == min(y_coords_all) and x == max(x_coords_all):
                selected_filter = filters["bottom_left"]
            elif y == max(y_coords_all) and x == min(x_coords_all):
                selected_filter = filters["top_right"]
            elif y == max(y_coords_all) and x == max(x_coords_all):
                selected_filter = filters["bottom_right"]
            elif y == min(y_coords_all):
                selected_filter = filters["left"]
            elif y == max(y_coords_all):
                selected_filter = filters["right"]
            elif x == min(x_coords_all):
                selected_filter = filters["top"]
            elif x == max(x_coords_all):
                selected_filter = filters["bottom"]
            else:
                selected_filter = filters["filter"]

            compiled_tiles[x : x + M, y : y + N] += selected_filter * (
                np.mean(low_scaled_depth[x : x + M, y : y + N])
                + np.std(low_scaled_depth[x : x + M, y : y + N])
                * ((scaled_depth - np.mean(scaled_depth)) / np.std(scaled_depth))
            )

    compiled_tiles[compiled_tiles < 0] = 0

    grey = np.mean(img, axis=2)
    tiles_blur = gaussian_filter(grey, sigma=20)
    tiles_diff = tiles_blur - grey
    tiles_diff = tiles_diff / np.max(tiles_diff)
    tiles_diff = gaussian_filter(tiles_diff, sigma=40)
    tiles_diff *= 5
    tiles_diff = np.clip(tiles_diff, 0, 0.999)

    combined_result = (compiled_tiles + low_scaled_depth) / 2
    combined_image = Image.fromarray((2**16 * 0.999 * combined_result / np.max(combined_result)).astype(np.uint16))
    return combined_image


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
        depth = infer(image, model_zoe)

        # depth_min = depth.min()
        # depth_max = depth.max()

        # vmax = int(depth_max) // 10 + 10

        # dst_path = dst_dir / f"{jpg_path.stem}_{depth_min:.1f}_{depth_max:.1f}.png"
        # Image.fromarray((depth * 65535 / vmax).astype(np.uint16)).save(dst_path)
        dst_path = dst_dir / f"{jpg_path.stem}.png"
        depth.save(dst_path)


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
        r"H:\work\Video3D\src_jpg\BD_ROM.Title1.chapter09_clip01",
        r"H:\work\Video3D\depth\BD_ROM.Title1.chapter09_clip01_high",
        model_name="ZoeD_K",
    )
