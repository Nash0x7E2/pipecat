"""Tests for Getstream transport implementation.

Two focused tests:
1. Mock-based full participant lifecycle (join -> audio -> video -> leave)
2. Real integration: GetstreamTransportClient connects, sends audio+video,
   a raw SDK participant verifies reception and sends media back.
"""

import asyncio
import os
import time
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from dotenv import load_dotenv

from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams

load_dotenv(override=True)

try:
    from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType

    from pipecat.transports.getstream.transport import (
        GetstreamCallbacks,
        GetstreamParams,
        GetstreamTransportClient,
        PipecatVideoStreamTrack,
    )

    GETSTREAM_AVAILABLE = True
except Exception:
    GETSTREAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_callbacks() -> "GetstreamCallbacks":
    """Create GetstreamCallbacks with all-AsyncMock handlers."""
    return GetstreamCallbacks(
        on_connected=AsyncMock(),
        on_disconnected=AsyncMock(),
        on_before_disconnect=AsyncMock(),
        on_participant_joined=AsyncMock(),
        on_participant_left=AsyncMock(),
        on_audio_track_subscribed=AsyncMock(),
        on_audio_track_unsubscribed=AsyncMock(),
        on_video_track_subscribed=AsyncMock(),
        on_video_track_unsubscribed=AsyncMock(),
        on_data_received=AsyncMock(),
        on_first_participant_joined=AsyncMock(),
    )


async def _create_client(
    video_in_enabled: bool = False,
    audio_in_enabled: bool = True,
) -> "GetstreamTransportClient":
    """Create a GetstreamTransportClient with mocked internals."""
    params = GetstreamParams(
        video_in_enabled=video_in_enabled,
        audio_in_enabled=audio_in_enabled,
    )
    callbacks = _create_callbacks()
    client = GetstreamTransportClient(
        api_key="test-key",
        api_secret="test-secret",
        call_type="default",
        call_id="test-call",
        user_id="bot-user",
        params=params,
        callbacks=callbacks,
        transport_name="test-transport",
    )
    task_manager = TaskManager()
    task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
    client._task_manager = task_manager
    return client


def _make_participant(user_id: str, session_id: str = "session-1"):
    """Create a mock Participant protobuf."""
    p = MagicMock()
    p.user_id = user_id
    p.session_id = session_id
    return p


def _make_participant_event(user_id: str, session_id: str = "session-1"):
    """Create a mock ParticipantJoined/Left protobuf event."""
    event = MagicMock()
    event.participant = _make_participant(user_id, session_id)
    return event


def _make_track_published_event(user_id: str, session_id: str, track_type: int):
    """Create a mock TrackPublished protobuf event."""
    event = MagicMock()
    event.user_id = user_id
    event.session_id = session_id
    event.type = track_type
    event.participant = _make_participant(user_id, session_id)
    return event


def _make_track_unpublished_event(user_id: str, session_id: str, track_type: int):
    """Create a mock TrackUnpublished protobuf event."""
    event = MagicMock()
    event.user_id = user_id
    event.session_id = session_id
    event.type = track_type
    event.cause = 0
    event.participant = _make_participant(user_id, session_id)
    return event


def _make_pcm_data(user_id: str, session_id: str = "session-1"):
    """Create a mock PcmData with .participant attribute."""
    pcm = MagicMock()
    pcm.participant = _make_participant(user_id, session_id)
    return pcm


# ---------------------------------------------------------------------------
# Test 1: Full Participant Lifecycle (Mock)
# ---------------------------------------------------------------------------


@unittest.skipUnless(GETSTREAM_AVAILABLE, "getstream[webrtc] package not installed")
class TestGetstreamParticipantLifecycle(unittest.IsolatedAsyncioTestCase):
    """Mock-based test covering the full event lifecycle:
    join -> audio -> track add/publish -> track unpublish -> leave.
    """

    async def test_full_participant_session(self):
        """Simulate a complete participant session from join to leave."""
        client = await _create_client(video_in_enabled=True)
        user = _make_participant("user-A", "session-1")

        # 1. Participant joins
        join_event = _make_participant_event("user-A", "session-1")
        client._on_participant_joined(join_event)
        self.assertIn("user-A", client._participants)

        await client._async_on_participant_joined("user-A")
        client._callbacks.on_first_participant_joined.assert_called_once_with("user-A")

        # 2. Audio starts flowing
        mock_pcm = _make_pcm_data("user-A")
        client._on_audio(mock_pcm)
        self.assertEqual(client._audio_queue.qsize(), 1)
        self.assertIn("user-A", client._audio_subscribed_participants)

        # 3. Video track is added + published (two-phase resolution)
        client._on_track_added("video-track-1", "video", user)
        self.assertIn("video-track-1", client._pending_tracks)

        pub_event = _make_track_published_event("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)
        client._on_track_published(pub_event)
        self.assertNotIn("video-track-1", client._pending_tracks)
        self.assertIn(("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO), client._track_map)

        # 4. Track is unpublished
        mock_task = MagicMock()
        client._video_subscriber_tasks["user-A:video-track-1"] = mock_task
        client._video_subscribed_participants.add("user-A")

        unpub_event = _make_track_unpublished_event(
            "user-A", "session-1", TrackType.TRACK_TYPE_VIDEO
        )
        client._on_track_unpublished(unpub_event)
        mock_task.cancel.assert_called_once()
        self.assertNotIn("user-A:video-track-1", client._video_subscriber_tasks)
        self.assertNotIn("user-A", client._video_subscribed_participants)

        # 5. Participant leaves
        left_event = _make_participant_event("user-A", "session-1")
        client._on_participant_left(left_event)
        self.assertNotIn("user-A", client._participants)
        self.assertFalse(client._other_participant_has_joined)


# ---------------------------------------------------------------------------
# Test 2: Real Integration — GetstreamTransportClient sends/receives media
# ---------------------------------------------------------------------------

STREAM_API_KEY = os.environ.get("STREAM_API_KEY")
STREAM_API_SECRET = os.environ.get("STREAM_API_SECRET")
GETSTREAM_INTEGRATION_AVAILABLE = bool(GETSTREAM_AVAILABLE and STREAM_API_KEY and STREAM_API_SECRET)


@unittest.skipUnless(
    GETSTREAM_INTEGRATION_AVAILABLE,
    "Requires STREAM_API_KEY and STREAM_API_SECRET env vars and getstream[webrtc]",
)
class TestGetstreamBidirectionalMedia(unittest.IsolatedAsyncioTestCase):
    """Real integration test using GetstreamTransportClient.

    The bot connects via the actual transport client (connect/disconnect),
    publishes audio+video, and a raw SDK participant verifies reception.
    The raw participant also sends media back to verify the transport receives it.
    """

    async def test_simultaneous_audio_and_video_bidirectional(self):
        """GetstreamTransportClient exchanges audio+video with a real participant."""
        from getstream import AsyncStream
        from getstream.models import UserRequest
        from getstream.video import rtc
        from getstream.video.rtc import AudioStreamTrack, PcmData
        from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig

        # ── Setup: create call and users ─────────────────────────────
        call_id = f"integration-test-{uuid.uuid4().hex[:8]}"
        bot_user_id = f"bot-{uuid.uuid4().hex[:6]}"
        human_user_id = f"human-{uuid.uuid4().hex[:6]}"

        # ── Create the bot via GetstreamTransportClient ────────────
        bot_params = GetstreamParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_in_enabled=False,
            video_out_enabled=True,
            video_out_framerate=15,
        )
        bot_callbacks = _create_callbacks()
        bot_client = GetstreamTransportClient(
            api_key=STREAM_API_KEY,
            api_secret=STREAM_API_SECRET,
            call_type="default",
            call_id=call_id,
            user_id=bot_user_id,
            params=bot_params,
            callbacks=bot_callbacks,
            transport_name="integration-test-bot",
        )

        # Provide a real task manager
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        bot_client._task_manager = task_manager

        # Initialize the API client and upsert bot user (what setup() does)
        bot_client._client = AsyncStream(api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET)
        await bot_client._client.upsert_users(UserRequest(id=bot_user_id, name="Bot"))

        # Pre-create the human user
        await bot_client._client.upsert_users(
            UserRequest(id=human_user_id, name="Human"),
        )

        # Start the client (sets output sample rate)
        mock_start_frame = MagicMock()
        mock_start_frame.audio_out_sample_rate = 24000
        await bot_client.start(mock_start_frame)

        # ── Connect the bot to the call via the transport ────────────
        await bot_client.connect()

        try:
            self.assertTrue(bot_client._connected, "Bot should be connected")
            self.assertIsNotNone(bot_client._audio_track, "Bot should have an audio track")
            self.assertIsNotNone(bot_client._video_track, "Bot should have a video track")

            # ── Connect a raw SDK participant (the "human") ──────────
            api_client = AsyncStream(api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET)
            call = api_client.video.call("default", call_id)

            sub_config = SubscriptionConfig(default=TrackSubscriptionConfig(track_types=[1, 2]))
            cm_human = await rtc.join(
                call, user_id=human_user_id, create=False, subscription_config=sub_config
            )

            # Collectors for media the human receives from the bot
            human_received_audio = []
            human_received_video_tracks = []

            @cm_human.on("audio")
            def on_human_audio(pcm_data):
                human_received_audio.append(pcm_data)

            @cm_human.on("track_added")
            def on_human_track_added(track_id, kind, user):
                if kind == "video" and user and user.user_id != human_user_id:
                    human_received_video_tracks.append(track_id)

            async with cm_human:
                await asyncio.sleep(2)  # Let SFU settle

                # Human publishes audio so bot can receive it
                human_audio_track = AudioStreamTrack(sample_rate=24000, channels=1, format="s16")
                await cm_human.add_tracks(audio=human_audio_track)
                await cm_human.republish_tracks()

                # ── Bot sends audio (same path as write_audio_frame) ─
                for _ in range(100):
                    num_samples = 480  # 20ms at 24kHz
                    t = np.linspace(0, 0.020, num_samples, endpoint=False)
                    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
                    pcm = PcmData(
                        samples=samples,
                        sample_rate=24000,
                        channels=1,
                        format="s16",
                    )
                    await bot_client._audio_track.write(pcm)

                # ── Bot sends video (same path as write_video_frame) ─
                for i in range(15):
                    value = (i * 30 + 50) % 256
                    rgb = np.full((120, 160, 3), fill_value=value, dtype=np.uint8)
                    bot_client._video_track.write(rgb.tobytes(), (160, 120), "RGB")

                # ── Human sends audio back to the bot ────────────────
                for _ in range(50):
                    num_samples = 480
                    t = np.linspace(0, 0.020, num_samples, endpoint=False)
                    samples = (np.sin(2 * np.pi * 880 * t) * 16000).astype(np.int16)
                    pcm = PcmData(
                        samples=samples,
                        sample_rate=24000,
                        channels=1,
                        format="s16",
                    )
                    await human_audio_track.write(pcm)

                # ── Wait for media to propagate ──────────────────────
                deadline = time.time() + 20
                while time.time() < deadline:
                    human_got_audio = len(human_received_audio) > 0
                    human_got_video = len(human_received_video_tracks) > 0
                    bot_got_audio = bot_client._audio_queue.qsize() > 0
                    if human_got_audio and human_got_video and bot_got_audio:
                        break
                    await asyncio.sleep(0.5)

                # ── Assertions ───────────────────────────────────────

                # Human received audio from the bot
                self.assertGreater(
                    len(human_received_audio),
                    0,
                    "Human did not receive audio from the bot transport",
                )
                pcm_from_bot = human_received_audio[0]
                self.assertTrue(hasattr(pcm_from_bot, "samples"))
                self.assertGreater(len(pcm_from_bot.samples), 0)

                # Human received the bot's video track
                self.assertGreater(
                    len(human_received_video_tracks),
                    0,
                    "Human did not receive video track from the bot transport",
                )

                # Bot received audio from the human (via transport's _audio_queue)
                self.assertGreater(
                    bot_client._audio_queue.qsize(),
                    0,
                    "Bot transport did not receive audio from human",
                )

                # Bot saw the human as a participant
                self.assertIn(
                    human_user_id,
                    bot_client._participants,
                    "Bot transport did not register the human participant",
                )

        finally:
            # ── Disconnect the bot via the transport ─────────────────
            await bot_client.disconnect()
            self.assertFalse(bot_client._connected, "Bot should be disconnected")


if __name__ == "__main__":
    unittest.main()
