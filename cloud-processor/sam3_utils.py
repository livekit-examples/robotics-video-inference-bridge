import logging
import os

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("sam3-utils")


def load_model(confidence_threshold: float = 0.5):
    """Load SAM3 model and return a ready-to-use processor."""
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    sam3_root = os.path.dirname(__import__("sam3").__file__)
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path)
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


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

    masks = state["masks"]  # (N, 1, H, W) bool
    boxes = state["boxes"]  # (N, 4) float32, absolute pixels [x1, y1, x2, y2]
    scores = state["scores"]  # (N,) bfloat16

    detections = []
    for i in range(len(scores)):
        mask_np = masks[i, 0].cpu().numpy().astype(np.uint8)
        box = boxes[i].cpu().float().tolist()

        detections.append(
            {
                "score": float(scores[i]),
                "box": {
                    "x1": box[0] / w,
                    "y1": box[1] / h,
                    "x2": box[2] / w,
                    "y2": box[3] / h,
                },
                "mask_rle": encode_mask_rle(mask_np),
            }
        )

    return detections
