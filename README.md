<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Robotics: Video Inference Bridge

Real-time cloud inference transport for computer vision using [LiveKit](https://livekit.io/). Stream video from edge devices to the cloud for [SAM3](https://github.com/anthropics/sam3) segmentation, receiving masks and bounding boxes in real-time.

## Architecture

```
┌─────────────────┐                           ┌─────────────────┐
│   Edge Client   │                           │ Cloud Processor │
│  (Python/ESP32) │──── H.264 Video ────────▶│     (SAM3)      │
│                 │◀─── Detections ──────────│                 │
└─────────────────┘         LiveKit           └─────────────────┘
```

The cloud processor runs SAM3 — a text-prompted segmentation model. Given a video frame and a text prompt (e.g. `"wheel"`), it returns per-object binary masks, bounding boxes, and confidence scores. Masks are RLE-encoded to keep payloads small enough for real-time transport.

## Components

| Component                                       | Description                        |
| ----------------------------------------------- | ---------------------------------- |
| [cloud-processor](./cloud-processor/)           | SAM3 segmentation on video streams |
| [edge-client](./edge-client/)                   | Python webcam client               |
| [edge-embedded-client](./edge-embedded-client/) | ESP32-P4 camera client             |

## Prerequisites

- [LiveKit Cloud](https://cloud.livekit.io/) account (or self-hosted LiveKit server)
- [uv](https://github.com/astral-sh/uv) — Python package management
- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/) v5.4+ — ESP32 development (optional)

### System dependencies

**Cloud processor (GPU machine):**

- Python 3.10+
- NVIDIA GPU with CUDA support
- NVIDIA drivers (compatible with your GPU)
- CUDA Toolkit 11.8+ or 12.x

**Edge client (any machine):**

- Python 3.10+
- ffmpeg — required by `imageio` for webcam capture
- Webcam (or video device at `/dev/video0`)

Install system deps on Ubuntu/Debian:

```sh
# Edge client
sudo apt install ffmpeg

# Cloud processor — install CUDA toolkit
# See https://developer.nvidia.com/cuda-downloads
```

On macOS (edge client only):

```sh
brew install ffmpeg
```

### SAM3 model weights

The SAM3 model weights are downloaded automatically on first run via HuggingFace. If the model is gated, you'll need to:

1. Create a [HuggingFace](https://huggingface.co/) account
2. Accept the model license on the SAM3 model page
3. Log in locally:

```sh
uv tool install huggingface-hub
hf auth login
```

If the model is **not** gated, no HuggingFace login is needed — weights download automatically.

## Quick Start

### 1. Get LiveKit credentials

Create a project at [LiveKit Cloud](https://cloud.livekit.io/) and copy your API key and secret.

### 2. Configure environment

```sh
# Cloud processor
cd cloud-processor
cp .env.local.example .env.local
# Edit .env.local with your credentials

# Edge client
cd ../edge-client
cp .env.local.example .env.local
# Edit .env.local with your credentials
```

### 3. Run

```sh
# Terminal 1: Start cloud processor
cd cloud-processor && uv sync
uv run python main.py

# Terminal 2: Start edge client
cd edge-client && uv sync
uv run python main.py
```

## Environment Variables

| Variable             | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| `LIVEKIT_URL`        | LiveKit server URL (e.g., `wss://your-project.livekit.cloud`) |
| `LIVEKIT_API_KEY`    | API key from LiveKit Cloud                                    |
| `LIVEKIT_API_SECRET` | API secret from LiveKit Cloud                                 |
| `LIVEKIT_ROOM`       | Room name (default: `edge-cv`)                                |
| `SAM3_PROMPT`        | Text prompt for segmentation (default: `object`)              |

## Detection Format

Detections are published on the `detections` topic as JSON. Each message includes the `source` identity of the edge client whose frame was processed:

```json
{
  "source": "edge-client",
  "timestamp": 1234567890.123,
  "frame_width": 640,
  "frame_height": 480,
  "prompt": "wheel",
  "detections": [
    {
      "score": 0.95,
      "box": { "x1": 0.1, "y1": 0.2, "x2": 0.5, "y2": 0.8 },
      "mask_rle": {
        "counts": [52800, 120, 480, 95, ...],
        "size": [480, 640]
      }
    }
  ]
}
```

| Field      | Description                                          |
| ---------- | ---------------------------------------------------- |
| `source`   | Identity of the edge client that published the frame |
| `score`    | Detection confidence (0.0–1.0)                       |
| `box`      | Bounding box with normalized coordinates (0.0–1.0)   |
| `mask_rle` | Binary segmentation mask, RLE-encoded (see below)    |

## RLE Mask Encoding

Binary masks are compressed using COCO-style Run-Length Encoding (RLE) to reduce payload size. A raw 640x480 mask is 307,200 bytes; RLE typically compresses this to a few hundred integers.

### Format

```json
{
  "counts": [52800, 120, 480, 95, 480, 110, ...],
  "size": [480, 640]
}
```

- **`size`**: `[height, width]` of the mask
- **`counts`**: alternating run lengths of `0`s and `1`s, always starting with a `0`-run. The values represent consecutive pixels of the same value when the mask is read in **column-major (Fortran) order** — i.e., top-to-bottom, then left-to-right — matching the COCO RLE convention.

For example, `counts = [3, 2, 1]` on a 3x2 mask means:

```
Flat (column-major): [0, 0, 0, 1, 1, 0]

As 3x2 matrix (col-major fill):
  col0  col1
  0     1
  0     1
  0     0
```

### Encoding (Python)

```python
import numpy as np

def encode_mask_rle(mask: np.ndarray) -> dict:
    # Flatten 2D mask (H×W) to 1D in column-major order (top-to-bottom, then left-to-right)
    flat = mask.flatten(order="F").astype(np.uint8)
    # Diff consecutive elements — nonzero where pixel value changes (0→1 or 1→0)
    diff = np.diff(flat)
    # Indices where transitions occur (+1 because diff shifts indices by one)
    change_indices = np.where(diff != 0)[0] + 1
    # Compute run lengths by diffing [0, ...change_points..., total_length]
    runs = np.diff(np.concatenate([[0], change_indices, [len(flat)]]))
    # Counts must start with a 0-run; if mask starts with 1, prepend a zero-length 0-run
    if flat[0] == 1:
        runs = np.concatenate([[0], runs])
    return {"counts": runs.tolist(), "size": [mask.shape[0], mask.shape[1]]}
```

### Decoding (Python)

```python
import numpy as np

def decode_mask_rle(rle: dict) -> np.ndarray:
    h, w = rle["size"]  # Original mask dimensions
    counts = rle["counts"]  # Alternating run lengths: [0-run, 1-run, 0-run, ...]
    flat = np.zeros(h * w, dtype=np.uint8)  # Start with all zeros
    pos = 0
    for i, count in enumerate(counts):
        # Even indices are 0-runs (already zero), odd indices are 1-runs
        if i % 2 == 1:
            flat[pos:pos + count] = 1
        pos += count
    # Reshape back to 2D using column-major order (matches encoding)
    return flat.reshape((h, w), order="F")
```

### Decoding (C — for embedded clients)

```c
void decode_mask_rle(const int *counts, int num_counts,
                     int height, int width, uint8_t *mask) {
    memset(mask, 0, height * width);
    int pos = 0;
    for (int i = 0; i < num_counts; i++) {
        if (i % 2 == 1) {  // 1-runs at odd indices
            memset(mask + pos, 1, counts[i]);
        }
        pos += counts[i];
    }
    // Note: mask is in column-major order.
    // mask[col * height + row] = pixel at (row, col).
}
```

## Resources

- [LiveKit Docs](https://docs.livekit.io/)
- [LiveKit ESP32 SDK](https://github.com/livekit/client-sdk-esp32)
- [SAM3](https://github.com/anthropics/sam3)
