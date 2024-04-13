from pathlib import Path

from run_midas import main as midas_main
from run_nvds import main as nvds_main

TEST_DATA_DIR = Path("tests/data")
TEST_VIDEO_DIR = TEST_DATA_DIR / "video/circus"
TEST_IMAGE_DIR = TEST_DATA_DIR / "image/circus"


def test_midas_main(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    midas_main(TEST_IMAGE_DIR, depth_dir, use_half=True)

    image_num = len(list(TEST_IMAGE_DIR.glob("*.jpg")))
    depth_num = len(list(depth_dir.glob("*.png")))

    assert image_num == depth_num


def test_nvds_main(tmp_path: Path):
    depth_dir = tmp_path / "depth"
    nvds_main(TEST_IMAGE_DIR, None, depth_dir, use_half=True)

    image_num = len(list(TEST_IMAGE_DIR.glob("*.jpg")))
    depth_num = len(list(depth_dir.glob("*.png")))

    assert image_num == depth_num
