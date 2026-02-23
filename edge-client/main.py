import asyncio
import json
import logging
import os
import time
from collections import deque

import cv2
import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("edge-client")

WIDTH, HEIGHT = 640, 480
FRAME_DELAY_MS = 600  # Delay display to better match detection latency

# Colors for up to 8 detections (BGR)
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 255, 0),
    (255, 128, 0),
]
MASK_ALPHA = 0.4


def decode_mask_rle(rle: dict) -> np.ndarray:
    """Decode a COCO-style RLE mask to a binary numpy array (H, W)."""
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, count in enumerate(counts):
        if i % 2 == 1:
            flat[pos : pos + count] = 1
        pos += count
    return flat.reshape((h, w), order="F")


def draw_overlay(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw detection masks, bounding boxes, and scores on the frame."""
    if not detections:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        score = det["score"]
        box = det["box"]

        # Bounding box (normalized coords -> pixel coords)
        x1 = int(box["x1"] * w)
        y1 = int(box["y1"] * h)
        x2 = int(box["x2"] * w)
        y2 = int(box["y2"] * h)

        # Mask overlay
        if det.get("mask_rle"):
            mask = decode_mask_rle(det["mask_rle"])
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            colored = np.zeros_like(overlay)
            colored[:] = color
            overlay[mask == 1] = cv2.addWeighted(
                overlay[mask == 1], 1 - MASK_ALPHA, colored[mask == 1], MASK_ALPHA, 0
            )

        # Bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return overlay


# Shared state: latest detections from the cloud processor
_latest_detections: list[dict] = []


async def capture_and_display(source: rtc.VideoSource):
    """Capture webcam frames, publish to LiveKit, and display with overlays."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        logger.error("Cannot open webcam")
        return

    # Frame buffer for delayed display (to sync with detection latency)
    frame_buffer: deque[tuple[float, np.ndarray]] = deque()
    delay_sec = FRAME_DELAY_MS / 1000.0

    logger.info(f"Webcam streaming (display delay: {FRAME_DELAY_MS}ms)... Press 'q' to quit")
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            now = time.monotonic()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Publish to LiveKit immediately
            video_frame = rtc.VideoFrame(
                WIDTH, HEIGHT, rtc.VideoBufferType.RGB24, frame_rgb.tobytes()
            )
            source.capture_frame(video_frame)

            # Buffer the frame for delayed display
            frame_buffer.append((now, frame_bgr))

            # Display delayed frame (if available)
            while frame_buffer and (now - frame_buffer[0][0]) >= delay_sec:
                _, display_frame = frame_buffer.popleft()

            if frame_buffer:
                # Show the oldest frame that's ready
                display_frame = frame_buffer[0][1]
            else:
                display_frame = frame_bgr

            display = draw_overlay(display_frame, _latest_detections)
            cv2.imshow("Edge Client - SAM3", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            await asyncio.sleep(0)  # yield to event loop
    finally:
        cap.release()
        cv2.destroyAllWindows()


async def main():
    global _latest_detections

    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = os.environ.get("LIVEKIT_ROOM", "edge-cv")
    client_identity = os.environ.get("CLIENT_IDENTITY", "edge-client")
    track_name = f"sam3/{client_identity}"

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(client_identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = rtc.Room()


    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        global _latest_detections
        if data.topic == f"data/{track_name}":
            payload = json.loads(data.data.decode())
            prompt = payload.get("prompt", "")
            detections = payload.get("detections", [])
            _latest_detections = detections
            if detections:
                logger.info(f"prompt='{prompt}' — {len(detections)} detection(s)")

    logger.info(f"Connecting to room: {room_name}")
    await room.connect(url, token)
    logger.info("Connected, publishing video...")

    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track(track_name, source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    )

    try:
        await capture_and_display(source)
    except asyncio.CancelledError:
        pass
    finally:
        await room.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
