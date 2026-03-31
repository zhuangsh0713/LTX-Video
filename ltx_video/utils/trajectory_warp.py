import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ltx_video.models.autoencoders.vae_encode import vae_encode


Box = List[float]
LatentTrajectory = Dict[int, Dict[int, Box]]


def load_results_trajectory(results_json_path: str | Path) -> dict:
    results_json_path = Path(results_json_path)
    with results_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("vlm_planning", {}).get("Frames")
    if not isinstance(frames, dict) or not frames:
        raise ValueError(
            f"No valid `vlm_planning.Frames` found in results json: {results_json_path}"
        )
    return data


def step_gate(
    step_idx: int,
    num_steps: int,
    start_ratio: float,
    end_ratio: float,
    alpha: float,
) -> float:
    if num_steps <= 1:
        return alpha

    ratio = step_idx / float(num_steps - 1)
    if ratio < start_ratio:
        return 0.0
    if ratio < end_ratio:
        return alpha
    return alpha * 0.35


def should_apply_warp(step_idx: int, warp_every: int) -> bool:
    return warp_every > 0 and (step_idx % warp_every) == 0


def frame_to_latent_index(frame_idx: int, video_scale_factor: int) -> int:
    if frame_idx <= 0:
        return 0
    return 1 + ((frame_idx - 1) // max(video_scale_factor, 1))


def _latent_index_to_frame_range(
    latent_idx: int,
    num_frames: int,
    video_scale_factor: int,
) -> Tuple[int, int]:
    if latent_idx <= 0:
        return 0, min(1, num_frames)

    start = 1 + (latent_idx - 1) * max(video_scale_factor, 1)
    end = min(num_frames, start + max(video_scale_factor, 1))
    return start, end


def _normalize_box_xywh(box: List[float]) -> Box:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def clip_bbox(box: Box, frame_width: int, frame_height: int) -> Optional[Box]:
    x1, y1, x2, y2 = box
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    x1 = max(0.0, min(float(frame_width), x1))
    y1 = max(0.0, min(float(frame_height), y1))
    x2 = max(0.0, min(float(frame_width), x2))
    y2 = max(0.0, min(float(frame_height), y2))

    if x1 >= x2 or y1 >= y2:
        return None
    return [x1, y1, x2, y2]


def ann_to_xyxy(
    ann: dict,
    frame_width: int,
    frame_height: int,
) -> Optional[Box]:
    bbox = ann.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return clip_bbox(
        _normalize_box_xywh(bbox),
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _sort_keyframe_name(name: str) -> Tuple[int, str]:
    stem = Path(name).stem
    try:
        return int(stem), name
    except ValueError:
        return math.inf, name


def _results_json_to_exp_name(results_json_path: str | Path) -> str:
    stem = Path(results_json_path).stem
    suffix = "_results"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _tokenize_name(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    return {token for token in normalized.split() if token and not token.isdigit()}


def name_similarity(name_a: str, name_b: str) -> float:
    ta = _tokenize_name(name_a)
    tb = _tokenize_name(name_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def compute_box_iou(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return inter / union


def build_object_name_map(results_data: dict) -> Dict[int, str]:
    object_names: Dict[int, str] = {}
    planning_frames = results_data.get("vlm_planning", {}).get("Frames", {})
    for frame_idx in sorted(planning_frames.keys(), key=lambda value: int(value)):
        objects = planning_frames[frame_idx]
        if not isinstance(objects, list):
            continue
        for obj in objects:
            obj_id = int(obj["id"])
            obj_name = str(obj.get("name") or "").strip()
            if obj_name and obj_id not in object_names:
                object_names[obj_id] = obj_name
    return object_names


def match_anchor_annotations(
    annotations: List[dict],
    object_names: Dict[int, str],
    frame_tracks: Dict[int, List[Optional[Box]]],
    latent_tracks: LatentTrajectory,
    start_frame: int,
    tau_anchor: int,
    vae_scale_factor: int,
    frame_width: int,
    frame_height: int,
) -> Dict[int, Box]:
    matched_boxes: Dict[int, Box] = {}
    if not annotations:
        return matched_boxes

    candidates: List[Tuple[float, int, int, Box]] = []
    for obj_id, obj_latent_boxes in latent_tracks.items():
        if tau_anchor not in obj_latent_boxes:
            continue
        traj_box_frame = None
        obj_track = frame_tracks.get(obj_id)
        if obj_track and 0 <= start_frame < len(obj_track):
            traj_box_frame = obj_track[start_frame]
        for ann_idx, ann in enumerate(annotations):
            ann_box_frame = ann_to_xyxy(
                ann,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if ann_box_frame is None:
                continue
            score_name = name_similarity(
                object_names.get(obj_id, ""),
                str(ann.get("class_name") or ""),
            )
            score = score_name
            if traj_box_frame is not None:
                score = 0.65 * score_name + 0.35 * compute_box_iou(
                    traj_box_frame, ann_box_frame
                )
            elif len(annotations) == 1:
                score = max(score, 0.2)
            candidates.append(
                (
                    score,
                    obj_id,
                    ann_idx,
                    project_bbox_to_latent_grid(ann_box_frame, vae_scale_factor),
                )
            )

    used_objects = set()
    used_annotations = set()
    for score, obj_id, ann_idx, ann_box in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        if obj_id in used_objects or ann_idx in used_annotations:
            continue
        if score <= 0.05:
            continue
        matched_boxes[obj_id] = ann_box
        used_objects.add(obj_id)
        used_annotations.add(ann_idx)
    return matched_boxes


def build_anchor_boxes_from_mapping(
    mapping_json_path: str | Path,
    results_json_path: str | Path,
    results_data: dict,
    conditioning_items: Optional[List[Any]],
    frame_tracks: Dict[int, List[Optional[Box]]],
    latent_tracks: LatentTrajectory,
    video_scale_factor: int,
    vae_scale_factor: int,
    frame_width: int = 720,
    frame_height: int = 480,
) -> Dict[Tuple[int, int], Box]:
    if not mapping_json_path or not conditioning_items:
        return {}

    mapping_json_path = Path(mapping_json_path)
    if not mapping_json_path.is_file():
        return {}

    mapping_data = json.loads(mapping_json_path.read_text(encoding="utf-8"))
    exp_name = _results_json_to_exp_name(results_json_path)
    mapping_entry = mapping_data.get(exp_name) or {}
    keyframe_result = mapping_entry.get("keyframe_result") or {}
    ordered_keyframes = sorted(keyframe_result.keys(), key=_sort_keyframe_name)
    if not ordered_keyframes:
        return {}

    object_names = build_object_name_map(results_data)
    anchor_boxes: Dict[Tuple[int, int], Box] = {}

    for idx, item in enumerate(conditioning_items):
        start_frame = int(item.media_frame_number)
        tau_anchor = frame_to_latent_index(start_frame, video_scale_factor)
        annotations = keyframe_result.get(ordered_keyframes[idx], []) if idx < len(ordered_keyframes) else []
        matched_boxes = match_anchor_annotations(
            annotations=annotations,
            object_names=object_names,
            frame_tracks=frame_tracks,
            latent_tracks=latent_tracks,
            start_frame=start_frame,
            tau_anchor=tau_anchor,
            vae_scale_factor=vae_scale_factor,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        for obj_id, obj_latent_boxes in latent_tracks.items():
            fallback_box = obj_latent_boxes.get(tau_anchor)
            anchor_box = matched_boxes.get(obj_id, fallback_box)
            if anchor_box is not None:
                anchor_boxes[(obj_id, tau_anchor)] = anchor_box
    return anchor_boxes


def build_frame_level_tracks(
    results_data: dict,
    target_num_frames: int,
    frame_width: int = 720,
    frame_height: int = 480,
) -> Dict[int, List[Optional[Box]]]:
    """
    Build the canonical trajectory exactly like `VisCoT/align.py`:
    1. read sparse planned boxes using their original planning timestamps
    2. organize them by object id
    3. interpolate each object track to a fixed length `target_num_frames`
    """
    planning_frames = results_data["vlm_planning"]["Frames"]
    frames: Dict[int, List[Dict[str, Box]]] = {}
    object_appearance: Dict[int, int] = {}

    for frame_str, objects in planning_frames.items():
        t = int(frame_str)
        frames[t] = []
        if not isinstance(objects, list):
            continue
        for obj in objects:
            obj_id = int(obj["id"])
            box = obj.get("box")
            if not isinstance(box, list) or len(box) != 4:
                continue
            bbox = clip_bbox(
                _normalize_box_xywh(box),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if bbox is None:
                continue

            if obj_id not in object_appearance:
                object_appearance[obj_id] = t
            frames[t].append({"id": obj_id, "bbox": bbox})

    all_frames = sorted(frames.keys())
    obj_ids = sorted(object_appearance.keys())
    if not all_frames or not obj_ids:
        return {}

    bboxes = np.zeros((len(all_frames), len(obj_ids), 4), dtype=np.float32)
    frame_map = {f: i for i, f in enumerate(all_frames)}

    for j, obj_id in enumerate(obj_ids):
        for t in all_frames:
            for obj in frames[t]:
                if obj["id"] == obj_id:
                    bboxes[frame_map[t], j] = np.asarray(obj["bbox"], dtype=np.float32)

    interp = np.zeros((target_num_frames, len(obj_ids), 4), dtype=np.float32)
    src_idx = np.linspace(0, target_num_frames - 1, num=len(all_frames), dtype=np.float32)
    tgt_idx = np.arange(target_num_frames, dtype=np.float32)

    for j in range(len(obj_ids)):
        for k in range(4):
            valid = np.where(bboxes[:, j, k] != 0)[0]
            if len(valid) > 1:
                interp[:, j, k] = np.interp(
                    tgt_idx,
                    src_idx[valid],
                    bboxes[valid, j, k],
                )

    frame_tracks: Dict[int, List[Optional[Box]]] = {}
    for j, obj_id in enumerate(obj_ids):
        obj_track: List[Optional[Box]] = []
        for t in range(target_num_frames):
            box = interp[t, j]
            if np.all(box == 0):
                obj_track.append(None)
            else:
                obj_track.append(box.tolist())
        frame_tracks[obj_id] = obj_track
    return frame_tracks


def project_bbox_to_latent_grid(box: Box, vae_scale_factor: int) -> Box:
    scale = float(max(vae_scale_factor, 1))
    return [coord / scale for coord in box]


def aggregate_frame_tracks_to_latent(
    frame_tracks: Dict[int, List[Optional[Box]]],
    num_frames: int,
    latent_num_frames: int,
    video_scale_factor: int,
    vae_scale_factor: int,
) -> LatentTrajectory:
    latent_tracks: LatentTrajectory = {}
    for obj_id, boxes in frame_tracks.items():
        obj_latents: Dict[int, Box] = {}
        for latent_idx in range(latent_num_frames):
            start, end = _latent_index_to_frame_range(
                latent_idx, num_frames, video_scale_factor
            )
            valid = [boxes[t] for t in range(start, min(end, len(boxes))) if boxes[t] is not None]
            if not valid:
                continue
            box = np.mean(np.asarray(valid, dtype=np.float32), axis=0).tolist()
            obj_latents[latent_idx] = project_bbox_to_latent_grid(box, vae_scale_factor)
        if obj_latents:
            latent_tracks[obj_id] = obj_latents
    return latent_tracks


def shrink_box(box: Box, ratio: float) -> Box:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = max(1e-6, (x2 - x1) * ratio)
    h = max(1e-6, (y2 - y1) * ratio)
    return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h]


def expand_box(box: Box, ratio: float) -> Box:
    return shrink_box(box, ratio)


def _box_to_bounds(box: Box, height: int, width: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(math.floor(x1))))
    top = max(0, min(height, int(math.floor(y1))))
    right = max(0, min(width, int(math.ceil(x2))))
    bottom = max(0, min(height, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def build_soft_mask(
    box: Box,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    bounds = _box_to_bounds(box, height, width)
    if bounds is None:
        return None

    left, top, right, bottom = bounds
    h = bottom - top
    w = right - left
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype),
        indexing="ij",
    )
    sigma = torch.tensor(0.55, device=device, dtype=dtype)
    local = torch.exp(-0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2))
    canvas = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    canvas[:, :, top:bottom, left:right] = local
    return canvas


def crop_latent_patch(frame_latent: torch.Tensor, box: Box) -> Optional[torch.Tensor]:
    _, _, height, width = frame_latent.shape
    bounds = _box_to_bounds(box, height, width)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    patch = frame_latent[:, :, top:bottom, left:right]
    if patch.numel() == 0:
        return None
    return patch


def paste_patch_to_canvas(
    patch: torch.Tensor,
    dst_box: Box,
    height: int,
    width: int,
) -> Optional[torch.Tensor]:
    bounds = _box_to_bounds(dst_box, height, width)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    dst_h = bottom - top
    dst_w = right - left
    if dst_h <= 0 or dst_w <= 0:
        return None

    resized = F.interpolate(
        patch,
        size=(dst_h, dst_w),
        mode="bilinear",
        align_corners=False,
    )
    canvas = torch.zeros(
        (patch.shape[0], patch.shape[1], height, width),
        device=patch.device,
        dtype=patch.dtype,
    )
    canvas[:, :, top:bottom, left:right] = resized
    return canvas


def warp_crop(
    frame_latent: torch.Tensor,
    src_box: Box,
    dst_box: Box,
) -> Optional[torch.Tensor]:
    patch = crop_latent_patch(frame_latent, src_box)
    if patch is None:
        return None
    _, _, height, width = frame_latent.shape
    return paste_patch_to_canvas(patch, dst_box, height, width)


def triangle_weights(u: float) -> Tuple[float, float, float]:
    w_prev_anchor = max(0.0, 1.0 - 2.0 * u)
    w_prev_frame = 1.0 - abs(2.0 * u - 1.0)
    w_next_anchor = max(0.0, 2.0 * u - 1.0)
    denom = w_prev_anchor + w_prev_frame + w_next_anchor + 1e-8
    return (
        w_prev_anchor / denom,
        w_prev_frame / denom,
        w_next_anchor / denom,
    )


def init_anchor_memory(
    conditioning_items,
    anchor_boxes: Dict[Tuple[int, int], Box],
    vae,
    vae_per_channel_normalize: bool,
    video_scale_factor: int,
    source_shrink: float,
) -> Dict[Tuple[int, int], torch.Tensor]:
    anchor_memory: Dict[Tuple[int, int], torch.Tensor] = {}
    if not conditioning_items:
        return anchor_memory

    with torch.no_grad():
        for item in conditioning_items:
            media_item = item.media_item
            start_frame = int(item.media_frame_number)
            tau_anchor = frame_to_latent_index(start_frame, video_scale_factor)
            media_latents = vae_encode(
                media_item.to(dtype=vae.dtype, device=vae.device),
                vae,
                vae_per_channel_normalize=vae_per_channel_normalize,
            )
            anchor_frame = media_latents[:, :, 0]
            for (obj_id, tau_box), box in anchor_boxes.items():
                if tau_box != tau_anchor:
                    continue
                if box is None:
                    continue
                patch = crop_latent_patch(anchor_frame, shrink_box(box, source_shrink))
                if patch is None:
                    continue
                anchor_memory[(obj_id, tau_anchor)] = patch.to(dtype=media_latents.dtype)
    return anchor_memory


def _nearest_anchor(anchor_frames: List[int], tau: int) -> Tuple[Optional[int], Optional[int]]:
    prev_anchors = [a for a in anchor_frames if a <= tau]
    next_anchors = [a for a in anchor_frames if a >= tau]
    prev_anchor = prev_anchors[-1] if prev_anchors else None
    next_anchor = next_anchors[0] if next_anchors else None
    return prev_anchor, next_anchor


def apply_latent_warp_prior(
    latents_tok: torch.Tensor,
    patchifier,
    latent_height: int,
    latent_width: int,
    out_channels: int,
    num_cond_latents: int,
    latent_tracks: LatentTrajectory,
    anchor_memory: Dict[Tuple[int, int], torch.Tensor],
    anchor_frames: List[int],
    step_idx: int,
    num_steps: int,
    warp_every: int,
    alpha: float,
    start_ratio: float,
    end_ratio: float,
    source_shrink: float,
    target_expand: float,
) -> torch.Tensor:
    if not latent_tracks or not should_apply_warp(step_idx, warp_every):
        return latents_tok

    step_alpha = step_gate(step_idx, num_steps, start_ratio, end_ratio, alpha)
    if step_alpha <= 0:
        return latents_tok

    cond_tokens = None
    video_tokens = latents_tok
    if num_cond_latents > 0 and latents_tok.shape[1] > num_cond_latents:
        cond_tokens = latents_tok[:, :num_cond_latents]
        video_tokens = latents_tok[:, num_cond_latents:]

    video_latents = patchifier.unpatchify(
        latents=video_tokens,
        output_height=latent_height,
        output_width=latent_width,
        out_channels=out_channels,
    )

    _, _, latent_num_frames, height, width = video_latents.shape

    for tau in range(1, latent_num_frames):
        for obj_id, obj_latent_boxes in latent_tracks.items():
            target_box = obj_latent_boxes.get(tau)
            if target_box is None:
                continue

            blend_box = expand_box(target_box, target_expand)
            blend_mask = build_soft_mask(
                blend_box,
                height,
                width,
                device=video_latents.device,
                dtype=video_latents.dtype,
            )
            if blend_mask is None:
                continue

            prev_anchor, next_anchor = _nearest_anchor(anchor_frames, tau)
            if prev_anchor is not None and next_anchor is not None and next_anchor > prev_anchor:
                u = (tau - prev_anchor) / float(next_anchor - prev_anchor)
            else:
                u = 0.5
            w_prev_anchor, w_prev_frame, w_next_anchor = triangle_weights(u)

            blended = None
            total_weight = 0.0

            if prev_anchor is not None:
                patch = anchor_memory.get((obj_id, prev_anchor))
                if patch is not None:
                    canvas = paste_patch_to_canvas(patch, target_box, height, width)
                    if canvas is not None:
                        blended = canvas * w_prev_anchor if blended is None else blended + canvas * w_prev_anchor
                        total_weight += w_prev_anchor

            prev_box = obj_latent_boxes.get(tau - 1)
            if prev_box is not None:
                canvas = warp_crop(
                    video_latents[:, :, tau - 1],
                    shrink_box(prev_box, source_shrink),
                    target_box,
                )
                if canvas is not None:
                    blended = canvas * w_prev_frame if blended is None else blended + canvas * w_prev_frame
                    total_weight += w_prev_frame

            if next_anchor is not None and next_anchor != prev_anchor:
                patch = anchor_memory.get((obj_id, next_anchor))
                if patch is not None:
                    canvas = paste_patch_to_canvas(patch, target_box, height, width)
                    if canvas is not None:
                        blended = canvas * w_next_anchor if blended is None else blended + canvas * w_next_anchor
                        total_weight += w_next_anchor

            if blended is None or total_weight <= 0:
                continue

            blended = blended / total_weight
            video_latents[:, :, tau] = (
                (1.0 - step_alpha * blend_mask) * video_latents[:, :, tau]
                + (step_alpha * blend_mask) * blended
            )

    video_tokens_new, _ = patchifier.patchify(video_latents)
    if cond_tokens is not None:
        return torch.cat([cond_tokens, video_tokens_new], dim=1)
    return video_tokens_new
