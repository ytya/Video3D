import subprocess
from pathlib import Path

import fire


def main(src_dir: str | Path, dst_dir: str | Path, target: str = "projector"):
    """動画分割

    :param str|Path src_dir: 元動画があるフォルダ
    :param str|Path dst_dir: 出力先フォルダ
    :param str target: 出力先デバイス(projector or mobile)
    """
    for src_path in Path(src_dir).glob("*"):
        if src_path.suffix not in [".mp4", ".mkv"]:
            continue
        print(src_path)
        dst_path = Path(dst_dir) / src_path.name
        command = f'poetry run python run_pipeline.py -s "{str(src_path)}" -o "{str(dst_path)}" -t {target}'
        subprocess.call(command, shell=True)


if __name__ == "__main__":
    fire.Fire(main)
