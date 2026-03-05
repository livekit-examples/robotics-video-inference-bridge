import asyncio
import json
import logging
import os

from livekit import rtc

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cloud-processor")

# Frame rate limiting
TARGET_FPS = 5
FRAME_INTERVAL = 1.0 / TARGET_FPS

# GPU access control - only one inference at a time
# When locked, other tracks skip frames instead of queuing stale data
gpu_lock = asyncio.Lock()

# Prompt storage: track_name -> prompt, None key = global
prompts: dict[str | None, str] = {}
default_prompt = os.environ.get("SAM3_PROMPT", "person")


def get_prompt(track_name: str) -> str:
    """Get effective prompt for a track: per-track -> global -> default."""
    return prompts.get(track_name) or prompts.get(None) or default_prompt


async def set_prompt(data: rtc.rpc.RpcInvocationData) -> str:
    """RPC handler for sam3.set_prompt method."""
    request = json.loads(data.payload)
    prompt = request["prompt"]
    track = request.get("track")  # None = global
    prompts[track] = prompt
    logger.info(f"[RPC] Prompt set: track={track or 'global'}, prompt='{prompt}'")
    return json.dumps({"success": True, "prompt": prompt})
