"""Quick smoke test — loads SAM3, runs inference on a solid color image, and prints results."""

import time

import numpy as np
from PIL import Image

from sam3_utils import load_model, run_inference, decode_mask_rle

PROMPT = "object"
IMG_W, IMG_H = 640, 480


def make_test_image() -> Image.Image:
    """Create a simple image with a white rectangle on a black background."""
    arr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    # white box in the center
    arr[140:340, 170:470] = 255
    return Image.fromarray(arr)


def main():
    print(f"Loading SAM3 model...")
    t0 = time.monotonic()
    processor = load_model()
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")

    image = make_test_image()
    print(f"Test image: {IMG_W}x{IMG_H} with white rectangle at center")
    print(f"Prompt: '{PROMPT}'")
    print()

    # warm-up run
    print("Warm-up inference...")
    t0 = time.monotonic()
    _ = run_inference(processor, image, PROMPT)
    print(f"Warm-up done in {(time.monotonic() - t0) * 1000:.0f}ms")
    print()

    # timed run
    print("Timed inference...")
    t0 = time.monotonic()
    detections = run_inference(processor, image, PROMPT)
    elapsed_ms = (time.monotonic() - t0) * 1000
    print(f"Inference done in {elapsed_ms:.0f}ms")
    print(f"Detections: {len(detections)}")
    print()

    for i, det in enumerate(detections):
        box = det["box"]
        mask_rle = det["mask_rle"]
        mask = decode_mask_rle(mask_rle)
        mask_pixels = int(mask.sum())
        total_pixels = mask_rle["size"][0] * mask_rle["size"][1]
        print(
            f"  [{i}] score={det['score']:.3f}  "
            f"box=({box['x1']:.2f}, {box['y1']:.2f}, {box['x2']:.2f}, {box['y2']:.2f})  "
            f"mask={mask_pixels}/{total_pixels} px ({mask_pixels / total_pixels * 100:.1f}%)"
        )

    if not detections:
        print("  (no detections — try a different prompt or image)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
