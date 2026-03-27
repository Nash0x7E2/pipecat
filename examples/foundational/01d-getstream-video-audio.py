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

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.getstream.transport import GetstreamParams, GetstreamTransport
from pipecat.transports.getstream.utils import GetstreamRESTHelper

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    stream_base_url = os.getenv("STREAM_BASE_URL")
    if not stream_base_url:
        raise ValueError("STREAM_BASE_URL environment variable not set.")

    stream_api_key = os.getenv("STREAM_API_KEY")
    stream_api_secret = os.getenv("STREAM_API_SECRET")
    stream_call_type = os.getenv("STREAM_CALL_TYPE", "default")
    stream_call_id = os.getenv("STREAM_CALL_ID", str(uuid.uuid4()))

    transport = GetstreamTransport(
        api_key=stream_api_key,
        api_secret=stream_api_secret,
        call_type=stream_call_type,
        call_id=stream_call_id,
        user_id=os.getenv("STREAM_USER_ID", "pipecat-bot"),
        params=GetstreamParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            video_out_enabled=True,
            video_in_enabled=True,
            video_out_is_live=True,
        ),
    )
    helper = GetstreamRESTHelper(
        api_key=stream_api_key,
        api_secret=stream_api_secret,
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        settings=GoogleLLMService.Settings(
            system_instruction="You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the user said in a creative, helpful, and brief way.",
        ),
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-2-thalia-en",
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    runner = PipelineRunner()

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
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

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(*_):
        await asyncio.sleep(1)
        context.add_message(
            {
                "role": "user",
                "content": "Start by greeting the user and ask how you can help.",
            }
        )
        await task.queue_frames([LLMRunFrame()])

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
