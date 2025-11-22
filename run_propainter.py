import sys

sys.path.append("ProPainter")

import argparse
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
import skvideo.io
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from ProPainter.core.utils import to_tensors
from ProPainter.model.misc import get_device
from ProPainter.model.modules.flow_comp_raft import RAFT_bi
from ProPainter.model.propainter import InpaintGenerator
from ProPainter.model.recurrent_flow_completion import RecurrentFlowCompleteNet
from ProPainter.utils.download_util import load_file_from_url
from video3d.util import ImageReader, VideoReader, read_video_info

warnings.filterwarnings("ignore")

pretrain_model_url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]

    return frames, process_size, out_size


#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(("mp4", "mov", "avi", "MP4", "MOV", "AVI")):  # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit="sec")  # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info["video_fps"]
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask > th] = 1
    mask[mask <= th] = 0
    return mask


# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []

    if mpath.endswith(("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")):  # input single img path
        masks_img = [Image.open(mpath)]
    else:
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))

    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert("L"))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))

        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))

    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting."""
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start : H_start + imgH, W_start : W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []

    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)

    mask[H_start + dilate_h : H_start + imgH - dilate_h, W_start + dilate_w : W_start + imgW - dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start : H_start + imgH, W_start : W_start + imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))

    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame

    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(
    mid_neighbor_id: int, neighbor_ids: list[int], length: int, ref_stride: int = 10, ref_num: int = -1
) -> list[int]:
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index


class FrameReader:
    def __init__(
        self,
        video_path: Path,
        mask_dir: Path,
        size: tuple[int, int],
        flow_mask_dilates: int = 8,
        mask_dilates: int = 5,
    ):
        self._video_reader = VideoReader(video_path)
        self._mask_reader = ImageReader(sorted(mask_dir.glob("*.png")))
        self._size = size
        self._flow_mask_dilates = flow_mask_dilates
        self._mask_dilates = mask_dilates

    def __iter__(self):
        return self

    def __next__(self):
        for (_, _, src_frame), (_, _, src_mask) in zip(self._video_reader, self._mask_reader):
            frame = Image.fromarray(src_frame).resize(self._size, Image.Resampling.BICUBIC)
            mask = Image.fromarray(src_mask).resize(self._size, Image.Resampling.NEAREST)
            mask_img = np.asarray(mask.convert("L"))
            if self._flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=self._flow_mask_dilates).astype(
                    np.uint8
                )
            else:
                flow_mask_img = binary_mask(mask_img).astype(np.uint8)
            flow_mask_img = Image.fromarray(flow_mask_img * 255)
            if self._mask_dilates > 0:
                mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=self._mask_dilates).astype(np.uint8)
            else:
                mask_img = binary_mask(mask_img).astype(np.uint8)
            mask_img = Image.fromarray(mask_img * 255)
            yield src_frame, src_mask, frame, flow_mask_img, mask_img
        raise StopIteration


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--src_path",
        type=str,
        default="inputs/object_removal/bmx-trees",
        help="Path of the input video or image folder.",
    )
    parser.add_argument(
        "-m",
        "--mask_dir",
        type=str,
        default="inputs/object_removal/bmx-trees_mask",
        help="Path of the mask(s) or mask folder.",
    )
    parser.add_argument("-o", "--output_path", type=str, default="", help="Output video path.")
    parser.add_argument("--resize_ratio", type=float, default=1.0, help="Resize scale for processing video.")
    parser.add_argument("--mask_dilation", type=int, default=4, help="Mask dilation for video and flow masking.")
    parser.add_argument("--ref_stride", type=int, default=10, help="Stride of global reference frames.")
    parser.add_argument("--neighbor_length", type=int, default=10, help="Length of local neighboring frames.")
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help="Length of sub-video for long video inference."
    )
    parser.add_argument("--raft_iter", type=int, default=20, help="Iterations for RAFT inference.")
    parser.add_argument("--save_frames", action="store_true", help="Save output frames. Default: False")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 (half precision) during inference. Default: fp32 (single precision).",
    )

    args = parser.parse_args()

    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if args.fp16 else False
    if device == torch.device("cpu"):
        use_half = False

    # 動画情報読み込み
    src_path = Path(args.src_path)
    mask_dir = Path(args.mask_dir)
    vinfo = read_video_info(src_path)
    r_frame_rate = f"{vinfo.frame_rate_n}/{vinfo.frame_rate_d}"
    size = (vinfo.width, vinfo.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * vinfo.width), int(args.resize_ratio * vinfo.height))
    size = (size[0] - size[0] % 8, size[1] - size[1] % 8)
    w, h = size

    frame_reader = next(
        FrameReader(src_path, mask_dir, size, flow_mask_dilates=args.mask_dilation, mask_dilates=args.mask_dilation)
    )

    # 出力動画
    output_path = Path(args.output_path)
    if args.output_path == "":
        output_path = Path(src_path.name)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    writer = skvideo.io.FFmpegWriter(
        output_path,
        inputdict={
            "-r": r_frame_rate,
        },
        outputdict={"-vcodec": "libx264", "-pix_fmt": "yuv420p", "-r": r_frame_rate, "-crf": str(20)},
    )

    raft_iter = args.raft_iter
    subvideo_length = args.subvideo_length
    neighbor_length = args.neighbor_length
    ref_stride = args.ref_stride

    ##############################################
    # set up RAFT and flow competition model
    ##############################################
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, "raft-things.pth"), model_dir="weights", progress=True, file_name=None
    )
    fix_raft = RAFT_bi(ckpt_path, device)

    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, "recurrent_flow_completion.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    ##############################################
    # set up ProPainter model
    ##############################################
    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, "ProPainter.pth"), model_dir="weights", progress=True, file_name=None
    )
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    ##############################################
    # ProPainter inference
    ##############################################
    video_length = vinfo.frame_num
    print(f"\nProcessing: {src_path.name} [{video_length} frames]...")
    with torch.no_grad():
        # ---- compute flow ----
        if w <= 640:
            short_clip_len = 12
        elif w <= 720:
            short_clip_len = 8
        elif w <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        loaded_frame_cnt = 0  # 読み込み済みのフレーム数
        pad_len = 10  # フロー推定用のパディング
        neighbor_stride = neighbor_length // 2  # インペイント用のストライド
        ref_num = subvideo_length // ref_stride  # インペイント用の参照フレーム数
        ref_pad_len = ref_stride * (ref_num // 2)  # インペイント用のパディング
        loaded_frames = []  # 読み込み済みのフレーム
        loaded_flow_masks = []
        loaded_dilated_masks = []
        gt_flows_f = torch.zeros((1, 0, 2, h, w), dtype=torch.float32).to(device)  # フロー推定キャッシュ
        gt_flows_b = gt_flows_f.clone()
        inpaint_frames = torch.zeros((1, 0, 3, h, w), dtype=torch.float32).to(
            device
        )  # インペイント用のフレームキャッシュ
        inpaint_masks_dilated = torch.zeros((1, 0, 1, h, w), dtype=torch.float32).to(
            device
        )  # インペイント用のマスクキャッシュ
        inpaint_masks = inpaint_masks_dilated.clone()
        inpaint_flows_f = gt_flows_f.clone()  # インペイント用のフロー推定キャッシュ
        inpaint_flows_b = gt_flows_f.clone()
        inpaint_frame_cnt = 0  # 処理済みのフレーム数
        inpaint_first_frame_cnt = 0  # インペイント用キャッシュの最初のフレーム番号
        ori_frames = [None] * video_length
        comp_frames = [None] * video_length
        src_frames = [None] * video_length
        src_masks = [None] * video_length
        for f in tqdm(range(0, video_length, subvideo_length)):
            # subvideo_length + パディング分を読み込み
            # 推定に必要なフレーム数
            # RAFT: short_clip_len
            # flow: pad_len + subvideo_length + pad_len
            # ProPainter: ref_pad_len + neighbor_length + neighbor_length + ref_pad_len
            # short_clip_len: 10ぐらい
            # pad_len: 10ぐらい
            # subvideo_length: 60ぐらい
            # neighbor_length: 10ぐらい
            # ref_pad_len: subvideo_length // 2
            pad_len_s = pad_len if len(loaded_frames) > 0 else 0
            loaded_frames = loaded_frames[-pad_len * 2 :]
            loaded_flow_masks = loaded_flow_masks[-pad_len * 2 :]
            loaded_dilated_masks = loaded_dilated_masks[-pad_len * 2 :]
            for _ in range(loaded_frame_cnt, min(f + subvideo_length + pad_len, video_length)):
                src_frame, src_mask, frame, flow_mask, mask = next(frame_reader)
                loaded_frames.append(frame)
                loaded_flow_masks.append(flow_mask)
                loaded_dilated_masks.append(mask)
                src_frames[loaded_frame_cnt] = src_frame
                src_masks[loaded_frame_cnt] = src_mask
                loaded_frame_cnt += 1
            sub_length = len(loaded_frames)
            pad_len_e = max(0, sub_length - subvideo_length - pad_len_s)
            frames = (to_tensors()(loaded_frames).unsqueeze(0) * 2 - 1).to(device)  # [1, sub_length, 3, h, w]
            flow_masks = to_tensors()(loaded_flow_masks).unsqueeze(0).to(device)  # [1, sub_length, 1, h, w]
            masks_dilated = to_tensors()(loaded_dilated_masks).unsqueeze(0).to(device)

            # use fp32 for RAFT
            # 計算済みのフレームは引き継ぐ
            gt_flows_f_list = [gt_flows_f[:, -pad_len * 2 + 1 :]]
            gt_flows_b_list = [gt_flows_b[:, -pad_len * 2 + 1 :]]
            start_idx = 0 if pad_len_s == 0 else gt_flows_f_list[0].shape[1] + 1
            for i in range(start_idx, sub_length, short_clip_len):
                raft_start_f = max(0, i - 1)
                raft_end_f = min(sub_length, i + short_clip_len)
                flows_f, flows_b = fix_raft(frames[:, raft_start_f:raft_end_f], iters=raft_iter)

                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()

            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)  # [1, sub_length-1, 2, h, w]
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_b, gt_flows_b)

            # ---- complete flow ----
            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                fix_flow_complete = fix_flow_complete.half()
                model = model.half()

            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(
                gt_flows_bi, pred_flows_bi, flow_masks
            )  # [1, sub_length, 2, h, w]

            # ---- image propagation ----
            masked_frames = frames * (1 - masks_dilated)
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated, "nearest"
            )
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)  # [1, sub_length, 1, h, w]

            # ---- feature propagation + transformer ----
            inpaint_flows_f = torch.cat([inpaint_flows_f, pred_flows_bi[0][:, pad_len_s:]], dim=1)
            inpaint_flows_b = torch.cat([inpaint_flows_b, pred_flows_bi[1][:, pad_len_s:]], dim=1)
            inpaint_frames = torch.cat([inpaint_frames, updated_frames[:, pad_len_s:]], dim=1)
            inpaint_masks_dilated = torch.cat([inpaint_masks_dilated, updated_masks[:, pad_len_s:]], dim=1)
            inpaint_masks = torch.cat([inpaint_masks, updated_masks[:, pad_len_s:]], dim=1)
            ori_frames[f : f + subvideo_length] = loaded_frames[pad_len_s : pad_len_s + subvideo_length]

            for f2 in range(inpaint_frame_cnt, video_length, neighbor_stride):
                # 推定に使うグローバルなフレーム番号を算出
                global_neighbor_ids = [
                    i for i in range(max(0, f2 - neighbor_stride), min(video_length, f2 + neighbor_stride))
                ]
                global_ref_ids = get_ref_index(f2, global_neighbor_ids, video_length, ref_stride, ref_num)
                is_end = (video_length - 1) in global_neighbor_ids

                # グローバルなフレーム番号をキャッシュのフレーム番号に変換
                neighbor_ids = [i - inpaint_first_frame_cnt for i in global_neighbor_ids]
                ref_ids = [i - inpaint_first_frame_cnt for i in global_ref_ids]
                if max(neighbor_ids + ref_ids) >= inpaint_frames.size(1):
                    # まだ読み込んでいないフレームが必要な場合は次のループへ
                    break

                selected_imgs = inpaint_frames[:, neighbor_ids + ref_ids, :, :, :]
                selected_masks = inpaint_masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
                selected_update_masks = inpaint_masks[:, neighbor_ids + ref_ids, :, :, :]
                selected_pred_flows_bi = (
                    inpaint_flows_f[:, neighbor_ids[:-1], :, :, :],
                    inpaint_flows_b[:, neighbor_ids[:-1], :, :, :],
                )

                # 1.0 indicates mask
                l_t = len(neighbor_ids)

                # pred_img = selected_imgs # results of image propagation
                pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)

                pred_img = pred_img.view(-1, 3, h, w)

                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = (
                    inpaint_masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                )
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    gidx = global_neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[gidx] * (
                        1 - binary_masks[i]
                    )
                    is_save = True
                    if comp_frames[gidx] is None:
                        comp_frames[gidx] = img
                        if is_end is False:
                            is_save = False
                    else:
                        comp_frames[gidx] = comp_frames[gidx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    comp_frames[gidx] = comp_frames[gidx].astype(np.uint8)
                    if is_save:
                        output_frame = cv2.resize(
                            comp_frames[gidx], (vinfo.width, vinfo.height), interpolation=cv2.INTER_CUBIC
                        )
                        src_frame = src_frames[gidx]
                        src_mask = src_masks[gidx]
                        output_frame[src_mask == 0] = src_frame[src_mask == 0]
                        writer.writeFrame(output_frame)
                        ori_frames[gidx] = None
                        comp_frames[gidx] = None
                        src_frames[gidx] = None
                        src_masks[gidx] = None

                torch.cuda.empty_cache()
                inpaint_frame_cnt = f2 + neighbor_stride

            # 不要なフレームを削除
            ff = inpaint_frame_cnt - inpaint_first_frame_cnt - ref_pad_len
            inpaint_frames = inpaint_frames[:, ff:-pad_len_e]
            inpaint_masks_dilated = inpaint_masks_dilated[:, ff:-pad_len_e]
            inpaint_masks = inpaint_masks[:, ff:-pad_len_e]
            inpaint_flows_f = inpaint_flows_f[:, ff : -pad_len_e + 1]
            inpaint_flows_b = inpaint_flows_b[:, ff : -pad_len_e + 1]
            inpaint_first_frame_cnt = inpaint_frame_cnt - ref_pad_len

    writer.close()
