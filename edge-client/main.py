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

    # Paint all mask regions into a single color layer, then blend once with
    # one vectorized addWeighted call instead of N per-detection blends.
    color_layer = frame_bgr.copy()
    has_mask = False
    for i, det in enumerate(detections):
        if det.get("mask_rle"):
            mask = decode_mask_rle(det["mask_rle"])
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color_layer[mask == 1] = COLORS[i % len(COLORS)]
            has_mask = True

    if has_mask:
        overlay = cv2.addWeighted(frame_bgr, 1 - MASK_ALPHA, color_layer, MASK_ALPHA, 0)
    else:
        overlay = color_layer

    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        score = det["score"]
        box = det["box"]

        x1 = int(box["x1"] * w)
        y1 = int(box["y1"] * h)
        x2 = int(box["x2"] * w)
        y2 = int(box["y2"] * h)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            overlay, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

    return overlay


async def display_remote_track(track: rtc.Track, detections: list[dict]):
    """Read frames from a subscribed remote video track and display with detection overlays."""
    video_stream = rtc.VideoStream(track)

    frame_buffer: deque[tuple[float, np.ndarray]] = deque()
    delay_sec = FRAME_DELAY_MS / 1000.0

    logger.info(
        f"Displaying remote track '{track.name}' (display delay: {FRAME_DELAY_MS}ms)... "
        "Press 'q' to quit"
    )
    try:
        async for frame_event in video_stream:
            frame = frame_event.frame
            rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
            arr = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape(
                (rgb_frame.height, rgb_frame.width, 3)
            )
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            now = time.monotonic()
            frame_buffer.append((now, frame_bgr))

            while frame_buffer and (now - frame_buffer[0][0]) >= delay_sec:
                _, display_frame = frame_buffer.popleft()

            if frame_buffer:
                display_frame = frame_buffer[0][1]
            else:
                display_frame = frame_bgr

            display = draw_overlay(display_frame, detections)
            cv2.imshow("Edge Client - SAM 3.1", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception:
        logger.exception(f"Error rendering remote track '{track.name}'")
    finally:
        cv2.destroyAllWindows()


async def main():
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = os.environ.get("LIVEKIT_ROOM", "edge-cv")
    client_identity = os.environ.get("CLIENT_IDENTITY", "edge-client")

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(client_identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = rtc.Room()
    display_task: asyncio.Task | None = None
    subscribed_track_name: str | None = None
    detections: list[dict] = []

    @room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        if subscribed_track_name and data.topic == f"data/{subscribed_track_name}":
            payload = json.loads(data.data.decode())
            prompt = payload.get("prompt", "")
            new_detections = payload.get("detections", [])
            detections[:] = new_detections
            if detections:
                logger.info(f"prompt='{prompt}' — {len(detections)} detection(s)")

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        print("on_track_subscribed", track.name)
        nonlocal display_task, subscribed_track_name
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        if not track.name.startswith("sam3/"):
            return
        if display_task is not None:
            return

        subscribed_track_name = track.name
        logger.info(f"Subscribed to video track: {participant.identity}:{track.name}")
        display_task = asyncio.create_task(display_remote_track(track, detections))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        nonlocal display_task, subscribed_track_name
        if display_task is not None and track.name == subscribed_track_name:
            display_task.cancel()
            display_task = None
            subscribed_track_name = None
            logger.info(f"Stopped displaying {participant.identity}:{track.name}")

    logger.info(f"Connecting to room: {room_name}")
    await room.connect(url, token)
    logger.info("Connected, waiting for remote video tracks...")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        if display_task is not None:
            display_task.cancel()
        await room.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
