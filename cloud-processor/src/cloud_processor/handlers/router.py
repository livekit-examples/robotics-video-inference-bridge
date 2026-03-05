from livekit import rtc

from cloud_processor.handlers.sam3 import handle_sam3
from cloud_processor.state import logger


async def process_track(
    track: rtc.Track,
    track_name: str,
    participant_identity: str,
    processor,
    room: rtc.Room,
):
    """Dispatch to the appropriate handler based on track name prefix."""
    prefix = track_name.split("/")[0] if "/" in track_name else track_name

    match prefix:
        case "sam3":
            await handle_sam3(track, track_name, participant_identity, processor, room)
        case _:
            logger.debug(f"[{participant_identity}] No handler for track '{track_name}', ignoring")
