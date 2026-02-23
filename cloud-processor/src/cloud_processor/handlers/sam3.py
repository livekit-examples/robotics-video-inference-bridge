import asyncio
import json
import os
import time

import numpy as np
import torch
from livekit import rtc
from PIL import Image

from cloud_processor.state import FRAME_INTERVAL, TARGET_FPS, get_prompt, gpu_lock, logger


# --- Model loading and inference utilities ---

def load_model(confidence_threshold: float = 0.5, warmup: bool = True):
    """Load SAM3 model and return a ready-to-use processor.

    Args:
        confidence_threshold: Minimum confidence for detections.
        warmup: If True, run a dummy inference to trigger torch.compile.
    """
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.float16).__enter__()

    sam3_root = os.path.dirname(__import__("sam3").__file__)
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path)

    # Compile for faster inference
    model = torch.compile(model, mode="reduce-overhead")

    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    if warmup:
        logger.info("Warming up model (compiling)...")
        dummy_image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        processor.set_text_prompt(state=processor.set_image(dummy_image), prompt="object")
        torch.cuda.synchronize()
        logger.info("Warmup complete")

    return processor


def encode_mask_rle(mask: np.ndarray) -> dict:
    """RLE-encode a binary mask (COCO-style, column-major).

    Args:
        mask: 2D boolean/uint8 array of shape (H, W).

    Returns:
        {"counts": [int, ...], "size": [H, W]}
        Counts alternate between 0-runs and 1-runs, starting with a 0-run.
    """
    flat = mask.flatten(order="F").astype(np.uint8)

    diff = np.diff(flat)
    change_indices = np.where(diff != 0)[0] + 1
    runs = np.diff(np.concatenate([[0], change_indices, [len(flat)]]))

    # Ensure counts start with a 0-run
    if flat[0] == 1:
        runs = np.concatenate([[0], runs])

    return {"counts": runs.tolist(), "size": [mask.shape[0], mask.shape[1]]}


def decode_mask_rle(rle: dict) -> np.ndarray:
    """Decode an RLE-encoded mask back to a binary numpy array.

    Args:
        rle: {"counts": [int, ...], "size": [H, W]}

    Returns:
        2D uint8 array of shape (H, W).
    """
    h, w = rle["size"]
    counts = rle["counts"]

    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:  # 1-runs at odd indices
            flat[pos : pos + count] = 1
        pos += count

    return flat.reshape((h, w), order="F")


def run_inference(processor, image: Image.Image, prompt: str) -> list[dict]:
    """Run SAM3 segmentation and return detections with RLE-encoded masks.

    Args:
        processor: Sam3Processor instance.
        image: PIL Image.
        prompt: Text prompt for detection.

    Returns:
        List of detections, each containing:
            - score: float confidence
            - box: {x1, y1, x2, y2} normalized to [0, 1]
            - mask_rle: RLE-encoded binary mask
    """
    w, h = image.size

    state = processor.set_image(image)
    state = processor.set_text_prompt(state=state, prompt=prompt)

    # Batch transfer to CPU (single sync instead of per-detection)
    masks_np = state["masks"][:, 0].cpu().numpy().astype(np.uint8)  # (N, H, W)
    boxes_np = state["boxes"].cpu().float().numpy()  # (N, 4)
    scores_np = state["scores"].cpu().float().numpy()  # (N,)

    detections = []
    for i in range(len(scores_np)):
        box = boxes_np[i]
        detections.append(
            {
                "score": float(scores_np[i]),
                "box": {
                    "x1": box[0] / w,
                    "y1": box[1] / h,
                    "x2": box[2] / w,
                    "y2": box[3] / h,
                },
                "mask_rle": encode_mask_rle(masks_np[i]),
            }
        )

    return detections


# --- Frame processing handler ---

async def handle_sam3(
    track: rtc.Track,
    track_name: str,
    participant_identity: str,
    processor,
    room: rtc.Room,
):
    """Process video frames using SAM3 segmentation."""
    video_stream = rtc.VideoStream(track)
    last_frame_time = 0.0
    frames_processed = 0
    frames_skipped_fps = 0
    frames_skipped_gpu = 0
    current_prompt = get_prompt(track_name)

    logger.info(
        f"[{participant_identity}:{track_name}] SAM3 handler started "
        f"(target {TARGET_FPS} fps, prompt='{current_prompt}')"
    )

    async for frame_event in video_stream:
        now = time.monotonic()

        # Drop frame if arriving too soon after the last processed frame
        if now - last_frame_time < FRAME_INTERVAL:
            frames_skipped_fps += 1
            continue

        # Drop frame if GPU is busy with another track
        if gpu_lock.locked():
            frames_skipped_gpu += 1
            continue

        last_frame_time = now

        # Get current prompt (may have changed via RPC)
        prompt = get_prompt(track_name)
        if prompt != current_prompt:
            logger.info(
                f"[{participant_identity}:{track_name}] Prompt changed: "
                f"'{current_prompt}' -> '{prompt}'"
            )
            current_prompt = prompt

        try:
            frame = frame_event.frame
            rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
            arr = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape(
                (rgb_frame.height, rgb_frame.width, 3)
            )
            image = Image.fromarray(arr)

            if frames_processed == 0:
                logger.info(
                    f"[{participant_identity}:{track_name}] First frame: "
                    f"{rgb_frame.width}x{rgb_frame.height}"
                )

            t0 = time.monotonic()
            async with gpu_lock:
                detections = await asyncio.to_thread(
                    run_inference, processor, image, prompt
                )
            inference_ms = (time.monotonic() - t0) * 1000

            frames_processed += 1

            logger.info(
                f"[{participant_identity}:{track_name}] Frame #{frames_processed}: "
                f"{len(detections)} detection(s), "
                f"inference {inference_ms:.0f}ms "
                f"(skipped fps: {frames_skipped_fps}, skipped gpu: {frames_skipped_gpu})"
            )

            await room.local_participant.publish_data(
                payload=json.dumps({
                    "source": participant_identity,
                    "track": track_name,
                    "timestamp": time.time(),
                    "frame_width": rgb_frame.width,
                    "frame_height": rgb_frame.height,
                    "prompt": prompt,
                    "detections": detections,
                }).encode(),
                reliable=False,
                topic="data/sam3_detections",
            )

            # Reset skip counters after successful inference
            frames_skipped_fps = 0
            frames_skipped_gpu = 0
        except Exception:
            logger.exception(f"[{participant_identity}:{track_name}] Error processing frame")
