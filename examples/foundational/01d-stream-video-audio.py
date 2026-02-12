#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import asyncio
import os
import sys
import uuid
from urllib.parse import urlencode

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import (
    OutputImageRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    UserImageRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.stream_video.transport import StreamVideoParams, StreamVideoTransport
from pipecat.transports.stream_video.utils import StreamVideoRESTHelper

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class EchoProcessor(FrameProcessor):
    """
    A processor to emit the received text and video back to the call.
    """

    async def process_frame(self, frame, direction: FrameDirection):
        # Always call super
        await super().process_frame(frame, direction)

        # Echo video: convert input video to output video
        if isinstance(frame, UserImageRawFrame):
            await self.push_frame(
                OutputImageRawFrame(
                    image=frame.image,
                    size=frame.size,
                    format=frame.format,
                ),
                direction,
            )

        # Echo audio: when STT generates text, speak it back via TTS
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text
            await self.push_frame(TTSSpeakFrame(text), direction)

        # Pass the original frame downstream
        await self.push_frame(frame, direction)


async def main():
    stream_base_url = os.getenv("STREAM_BASE_URL")
    if not stream_base_url:
        raise ValueError("STREAM_BASE_URL environment variable not set.")

    stream_api_key = os.getenv("STREAM_API_KEY")
    stream_api_secret = os.getenv("STREAM_API_SECRET")
    stream_call_type = os.getenv("STREAM_CALL_TYPE", "default")
    stream_call_id = os.getenv("STREAM_CALL_ID", str(uuid.uuid4()))

    transport = StreamVideoTransport(
        api_key=stream_api_key,
        api_secret=stream_api_secret,
        call_type=stream_call_type,
        call_id=stream_call_id,
        user_id=os.getenv("STREAM_USER_ID", "pipecat-bot"),
        params=StreamVideoParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            video_out_enabled=True,
            video_in_enabled=True,
            video_out_is_live=True,
        ),
    )
    helper = StreamVideoRESTHelper(
        api_key=stream_api_key,
        api_secret=stream_api_secret,
    )

    stt = CartesiaSTTService(api_key=os.getenv("CARTESIA_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    runner = PipelineRunner()

    task = PipelineTask(
        Pipeline([transport.input(), stt, EchoProcessor(), tts, transport.output()])
    )

    @transport.event_handler("on_connected")
    async def on_connected(*_):
        """
        Create a demo call link once the agent joins the call.
        """
        user_id = "demo-user"
        token = helper.create_token(user_id=user_id, expiration=60)
        params = {
            "api_key": stream_api_key,
            "token": token,
            "skip_lobby": "true",
            "user_name": user_id,
            "video_encoder": "h264",
            "bitrate": 12000000,
            "w": 1920,
            "h": 1080,
        }
        call_url = f"{stream_base_url}/join/{stream_call_id}?{urlencode(params)}"
        logger.warning(f"Open this page to join the call: {call_url}")

    # Register an event handler so we can play the audio when the
    # participant joins.
    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(*_):
        await asyncio.sleep(1)
        await task.queue_frame(
            TTSSpeakFrame(
                "Hello there! How are you doing today? Would you like to talk about the weather?"
            )
        )

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
