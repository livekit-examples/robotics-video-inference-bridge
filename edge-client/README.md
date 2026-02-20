# Edge Client

Python client that captures video from a webcam and streams it to the cloud processor via [LiveKit](https://livekit.io/). Receives SAM3 segmentation results (masks, bounding boxes, scores) in real-time.

## Features

- Captures 640x480 video from webcam
- Streams RGB24 frames via LiveKit
- Receives detections on `detections` topic
- Decodes RLE masks to numpy arrays
- Prints detection results to console

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- Webcam (or video device at `/dev/video0`)

## Setup

```sh
# Install dependencies
uv sync

# Configure environment
cp .env.local.example .env.local
# Edit .env.local with your LiveKit credentials
```

## Usage

```sh
uv run python main.py
```

## Environment Variables

| Variable             | Required | Description                                                   |
| -------------------- | -------- | ------------------------------------------------------------- |
| `LIVEKIT_URL`        | Yes      | LiveKit server URL (e.g., `wss://your-project.livekit.cloud`) |
| `LIVEKIT_API_KEY`    | Yes      | API key from LiveKit Cloud                                    |
| `LIVEKIT_API_SECRET` | Yes      | API secret from LiveKit Cloud                                 |
| `LIVEKIT_ROOM`       | No       | Room name (default: `edge-cv`)                                |

## Output

When the cloud processor detects objects, the client decodes the RLE masks and prints:

```
[edge-client] prompt='wheel' — 2 detection(s):
  [0] score=0.95 box=(0.100,0.200)-(0.500,0.800) mask=480x640, 12340px
  [1] score=0.87 box=(0.600,0.150)-(0.900,0.850) mask=480x640, 8920px
```

The decoded `mask` is a `numpy.ndarray` of shape `(H, W)` with `uint8` values `0` or `1`, available for downstream use (overlay, visualization, etc.).

## Configuration

| Setting  | Default | Description  |
| -------- | ------- | ------------ |
| `WIDTH`  | 640     | Video width  |
| `HEIGHT` | 480     | Video height |
