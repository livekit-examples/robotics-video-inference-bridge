import asyncio
import json
import logging
import os
import time

import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc
from PIL import Image

from sam3_utils import load_model, run_inference

load_dotenv(".env.local")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cloud-processor")

TARGET_FPS = 2
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Only one inference on the GPU at a time. When the lock is held,
# other tracks skip their current frame instead of queuing stale data.
gpu_lock = asyncio.Lock()


async def process_video_track(
    track: rtc.Track,
    participant_identity: str,
    processor,
    prompt: str,
    room: rtc.Room,
):
    """Process video frames from a track using SAM3 segmentation."""
    video_stream = rtc.VideoStream(track)
    last_frame_time = 0.0
    frames_received = 0
    frames_processed = 0
    frames_skipped_fps = 0
    frames_skipped_gpu = 0

    logger.info(f"[{participant_identity}] Starting frame processing loop (target {TARGET_FPS} fps)")

    async for frame_event in video_stream:
        frames_received += 1
        now = time.monotonic()

        if now - last_frame_time < FRAME_INTERVAL:
            frames_skipped_fps += 1
            continue

        # Drop frame if GPU is busy with another track
        if gpu_lock.locked():
            frames_skipped_gpu += 1
            if frames_skipped_gpu % 10 == 1:
                logger.debug(f"[{participant_identity}] GPU busy, skipping frame (total gpu skips: {frames_skipped_gpu})")
            continue

        last_frame_time = now

        try:
            frame = frame_event.frame
            rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
            arr = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape(
                (rgb_frame.height, rgb_frame.width, 3)
            )
            image = Image.fromarray(arr)

            if frames_processed == 0:
                logger.info(f"[{participant_identity}] First frame: {rgb_frame.width}x{rgb_frame.height}")

            t0 = time.monotonic()
            async with gpu_lock:
                detections = await asyncio.to_thread(
                    run_inference, processor, image, prompt
                )
            inference_ms = (time.monotonic() - t0) * 1000

            frames_processed += 1

            logger.info(
                f"[{participant_identity}] Frame #{frames_processed}: "
                f"{len(detections)} detection(s), "
                f"inference {inference_ms:.0f}ms "
                f"(received: {frames_received}, skipped fps: {frames_skipped_fps}, skipped gpu: {frames_skipped_gpu})"
            )

            await room.local_participant.publish_data(
                payload=json.dumps({
                    "source": participant_identity,
                    "timestamp": time.time(),
                    "frame_width": rgb_frame.width,
                    "frame_height": rgb_frame.height,
                    "prompt": prompt,
                    "detections": detections,
                }).encode(),
                reliable=False,
                topic="detections",
            )
        except Exception:
            logger.exception(f"[{participant_identity}] Error processing frame")


async def main():
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = os.environ.get("LIVEKIT_ROOM", "edge-cv")
    prompt = os.environ.get("SAM3_PROMPT", "object")

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("cloud-processor")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    logger.info("Loading SAM3 model...")
    processor = load_model()
    logger.info("Model ready")

    room = rtc.Room()
    video_tasks: dict[str, asyncio.Task] = {}

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        if participant.identity in video_tasks:
            return

        logger.info(f"Video track subscribed from {participant.identity}")
        video_tasks[participant.identity] = asyncio.create_task(
            process_video_track(track, participant.identity, processor, prompt, room)
        )

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(track: rtc.Track, publication, participant):
        task = video_tasks.pop(participant.identity, None)
        if task:
            task.cancel()
            logger.info(f"Stopped processing {participant.identity}")

    logger.info(f"Connecting to {url}")
    await room.connect(url, token)
    logger.info(f"Connected, waiting for video streams... (prompt: '{prompt}')")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Disconnecting...")
        await room.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
