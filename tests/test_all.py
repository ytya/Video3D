from pathlib import Path

from run_midas import main as midas_main
from run_nvds import main as nvds_main
from run_pipeline import run as pipeline_run
from run_stereo_video import SCONFIGS
from run_stereo_video import run as stereo_run

TEST_DATA_DIR = Path("tests/data")
TEST_VIDEO_PATH = TEST_DATA_DIR / "video/circus.mp4"
TEST_IMAGE_DIR = TEST_DATA_DIR / "image/circus"
TEST_DEPTH_DIR = TEST_DATA_DIR / "depth/circus"


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


def test_stereo_run(tmp_path: Path):
    dst_path = tmp_path / "stereo.mp4"
    stereo_run(TEST_VIDEO_PATH, TEST_IMAGE_DIR, TEST_DEPTH_DIR, dst_path, 1, SCONFIGS["projector"], 20)

    assert dst_path.exists()


def test_pipeline_run(tmp_path: Path):
    dst_path = tmp_path / "stereo.mp4"
    depth_dir = tmp_path / "depth"
    pipeline_run(TEST_VIDEO_PATH, TEST_IMAGE_DIR, depth_dir, dst_path, 1, SCONFIGS["projector"], 20)

    assert dst_path.exists()


def test_pipeline_run2(tmp_path: Path):
    dst_path = tmp_path / "stereo.mp4"
    pipeline_run(TEST_VIDEO_PATH, None, None, dst_path, 1, SCONFIGS["projector"], 20)

    assert dst_path.exists()
