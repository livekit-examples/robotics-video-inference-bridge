"""LiveKit Voice Agent for Edge CV."""

import json
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import silero

load_dotenv(".env.local")

logger = logging.getLogger("voice-agent")


class Assistant(Agent):
    def __init__(self, room) -> None:
        super().__init__(
            instructions="You are a helpful voice assistant for an edge computer vision system. You can set what object the system should detect using the set_prompt tool.",
        )
        self._room = room

    @function_tool
    async def set_prompt(self, context: RunContext, prompt: str):
        """Set the object detection prompt for the SAM3 vision system.

        Args:
            prompt: The object to detect (e.g. "person", "car", "wheel")
        """
        logger.info(f"[Tool] Setting prompt to: {prompt}")

        # Find cloud-processor participant
        for participant in self._room.remote_participants.values():
            if "cloud-processor" in participant.identity:
                response = await self._room.local_participant.perform_rpc(
                    destination_identity=participant.identity,
                    method="sam3.set_prompt",
                    payload=json.dumps({"prompt": prompt}),
                )
                result = json.loads(response)
                if result.get("success"):
                    return f"Detection prompt set to '{prompt}'"
                return f"Failed to set prompt: {result}"

        return "Cloud processor not found in room"


server = AgentServer()


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(model="cartesia/sonic-2", voice="a0e99841-438c-4a64-b679-ae501e7d6091"),
        vad=silero.VAD.load(),
    )

    await session.start(agent=Assistant(ctx.room), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
