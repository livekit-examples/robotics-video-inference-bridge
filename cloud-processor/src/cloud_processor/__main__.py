import asyncio
import os

from dotenv import load_dotenv

load_dotenv(".env.local")

from livekit import api, rtc

from cloud_processor.handlers import process_track
from cloud_processor.handlers.sam3 import load_model
from cloud_processor.state import set_prompt, logger


async def main():
    url = os.environ["LIVEKIT_URL"]
    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    room_name = os.environ.get("LIVEKIT_ROOM", "edge-cv")

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity("cloud-processor")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    logger.info("Loading SAM 3.1 model...")
    processor = load_model()
    logger.info("Model ready")

    room = rtc.Room()
    video_tasks: dict[str, asyncio.Task] = {}

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication, participant):
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return

        track_name = track.name
        task_key = f"{participant.identity}:{track_name}"
        if task_key in video_tasks:
            return

        logger.info(f"Video track subscribed: {task_key}")
        video_tasks[task_key] = asyncio.create_task(
            process_track(track, track_name, participant.identity, processor, room)
        )

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(track: rtc.Track, publication, participant):
        task_key = f"{participant.identity}:{track.name}"
        task = video_tasks.pop(task_key, None)
        if task:
            task.cancel()
            logger.info(f"Stopped processing {task_key}")

    logger.info(f"Connecting to {url}")
    await room.connect(url, token)

    room.local_participant.register_rpc_method("sam3.set_prompt", set_prompt)
    logger.info("Connected, RPC methods registered, waiting for video streams...")

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
