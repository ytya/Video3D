from pathlib import Path
from typing import Union, Optional

import subprocess
import fire


def main(src_path: Union[str, Path]):
    """動画分割

    :param Union[str, Path] src_path: 元動画
    """
    # パス設定
    src_path = Path(src_path)
    jpg_dir = src_path.parent.parent / "src_jpg" / src_path.stem
    depth_dir = jpg_dir.parent.parent / "depth" / jpg_dir.name
    stereo_path = src_path.parent.parent / "3d" / f"{src_path.stem}_nosound{src_path.suffix}"
    dst_path = stereo_path.parent / src_path.name

    # 前処理
    command = f'poetry run python preprocess.py "{src_path}" "{jpg_dir}"'
    subprocess.call(command, shell=True)

    # depth推定
    command = f'poetry run python run_midas.py "{jpg_dir}" "{depth_dir}"'
    subprocess.call(command, shell=True)

    # 3D動画生成
    command = f'poetry run python create_stereo_movie.py -s "{src_path}" -i "{jpg_dir}" -d "{depth_dir}" -o "{stereo_path}" --half'
    subprocess.call(command, shell=True)

    # 音声コピー
    command = f'ffmpeg -i "{src_path}" -i "{stereo_path}" -c:v copy -c:a copy -map 0:a -map 1:v "{dst_path}"'
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    fire.Fire(main)
