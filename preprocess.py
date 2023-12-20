from pathlib import Path
from typing import Union, Optional

import subprocess
import fire


def main(src_path: Union[str, Path], jpg_dir: Optional[Union[str, Path]] = None):
    """動画分割

    :param Union[str, Path] src_path: 元動画
    :param Optional[Union[str, Path]] jpg_dir: 出力フォルダ, defaults to None
    """
    # パス設定
    src_path = Path(src_path)
    if jpg_dir is None:
        jpg_dir = src_path.parent.parent / "src_jpg" / src_path.stem
    else:
        jpg_dir = Path(jpg_dir)
    jpg_dir.mkdir(exist_ok=True, parents=True)

    # 実行
    command = f'ffmpeg -i "{src_path}" -q 2 "{jpg_dir}/%05d.jpg"'
    subprocess.call(command, shell=True)


if __name__ == "__main__":
    fire.Fire(main)
