# Cloud Processor

Subscribes to video streams from edge clients via [LiveKit](https://livekit.io/), runs [SAM3](https://github.com/anthropics/sam3) text-prompted segmentation, and publishes detection results back to the room.

## Features

- Receives video streams from edge clients
- Runs SAM3 segmentation at ~2 FPS per client
- Text-prompted detection (e.g. `"wheel"`, `"person"`)
- Publishes masks (RLE-encoded), bounding boxes, and scores on `detections` topic
- GPU lock ensures fair scheduling across multiple clients
- Cleans up automatically when clients disconnect

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- GPU with CUDA support

## Setup

```sh
# Install dependencies
uv sync

# (Optional) Install with Flash Attention for faster inference
# Requires Python dev headers: sudo apt-get install python3.12-dev
uv sync --extra flash

# Configure environment
cp .env.local.example .env.local
# Edit .env.local with your LiveKit credentials
```

## Usage

```sh
uv run -m cloud_processor
```

## Environment Variables

| Variable             | Required | Description                                                   |
| -------------------- | -------- | ------------------------------------------------------------- |
| `LIVEKIT_URL`        | Yes      | LiveKit server URL (e.g., `wss://your-project.livekit.cloud`) |
| `LIVEKIT_API_KEY`    | Yes      | API key from LiveKit Cloud                                    |
| `LIVEKIT_API_SECRET` | Yes      | API secret from LiveKit Cloud                                 |
| `LIVEKIT_ROOM`       | No       | Room name (default: `edge-cv`)                                |
| `SAM3_PROMPT`        | No       | Text prompt for segmentation (default: `object`)              |

## Output Format

Detections are published as JSON on the `detections` topic:

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
      "mask_rle": { "counts": [52800, 120, 480, 95], "size": [480, 640] }
    }
  ]
}
```

- `source`: identity of the edge client whose frame was processed
- `score`: detection confidence (0.0–1.0)
- `box`: bounding box with normalized coordinates (0.0–1.0)
- `mask_rle`: COCO-style RLE binary mask (see root README for encoding details)

## Files

```
src/cloud_processor/
├── __main__.py          # Entry point, room lifecycle
├── state.py             # Config, shared state, RPC handlers
└── handlers/
    ├── router.py        # Track routing dispatch
    └── sam3.py          # SAM3 model, inference, frame processing
```

## Configuration

| Setting      | Default  | Description                  |
| ------------ | -------- | ---------------------------- |
| `TARGET_FPS` | 2        | Max inference rate per track |
| `confidence` | 0.5      | Minimum detection confidence |
