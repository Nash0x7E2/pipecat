"""Tests for Stream Video transport implementation.

Two focused tests:
1. Mock-based full participant lifecycle (join → audio → video → leave)
2. Real integration: bidirectional audio + video via Stream Video SFU
"""

import asyncio
import os
import time
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType

    from pipecat.transports.stream_video.transport import (
        PipecatVideoStreamTrack,
        StreamVideoCallbacks,
        StreamVideoParams,
        StreamVideoTransport,
        StreamVideoTransportClient,
    )

    STREAM_VIDEO_AVAILABLE = True
except Exception:
    STREAM_VIDEO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_callbacks() -> "StreamVideoCallbacks":
    """Create StreamVideoCallbacks with all-AsyncMock handlers."""
    return StreamVideoCallbacks(
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


def _create_client(
    video_in_enabled: bool = False,
    audio_in_enabled: bool = True,
) -> "StreamVideoTransportClient":
    """Create a StreamVideoTransportClient with mocked internals."""
    params = StreamVideoParams(
        video_in_enabled=video_in_enabled,
        audio_in_enabled=audio_in_enabled,
    )
    callbacks = _create_callbacks()
    client = StreamVideoTransportClient(
        api_key="test-key",
        api_secret="test-secret",
        call_type="default",
        call_id="test-call",
        user_id="bot-user",
        params=params,
        callbacks=callbacks,
        transport_name="test-transport",
    )
    task_manager = MagicMock()
    task_manager.create_task.return_value = MagicMock()
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


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestStreamVideoParticipantLifecycle(unittest.IsolatedAsyncioTestCase):
    """Mock-based test covering the full event lifecycle:
    join → audio → track add/publish → track unpublish → leave.
    """

    async def test_full_participant_session(self):
        """Simulate a complete participant session from join to leave."""
        client = _create_client(video_in_enabled=True)
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
# Test 2: Real Integration — Bidirectional Audio + Video via Stream API
# ---------------------------------------------------------------------------

STREAM_API_KEY = os.environ.get("STREAM_API_KEY")
STREAM_API_SECRET = os.environ.get("STREAM_API_SECRET")
STREAM_INTEGRATION_AVAILABLE = bool(STREAM_VIDEO_AVAILABLE and STREAM_API_KEY and STREAM_API_SECRET)


@unittest.skipUnless(
    STREAM_INTEGRATION_AVAILABLE,
    "Requires STREAM_API_KEY and STREAM_API_SECRET env vars and getstream[webrtc]",
)
class TestStreamVideoBidirectionalMedia(unittest.IsolatedAsyncioTestCase):
    """Real integration test: two participants exchange audio + video via Stream Video.

    Connects two ConnectionManagers to a real Stream Video call, publishes
    audio and video from participant-B, and verifies participant-A receives them.
    Then publishes from participant-A and verifies participant-B receives them.
    """

    async def test_simultaneous_audio_and_video_bidirectional(self):
        """Two real participants exchange audio and video over Stream Video SFU."""
        from getstream import AsyncStream
        from getstream.models import UserRequest
        from getstream.video import rtc
        from getstream.video.rtc import AudioStreamTrack, PcmData
        from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig

        from pipecat.transports.stream_video.transport import PipecatVideoStreamTrack

        # ── Setup: create client, users, and call ──────────────────────
        api_client = AsyncStream(api_key=STREAM_API_KEY, api_secret=STREAM_API_SECRET)
        call_id = f"integration-test-{uuid.uuid4().hex[:8]}"
        user_a_id = f"user-a-{uuid.uuid4().hex[:6]}"
        user_b_id = f"user-b-{uuid.uuid4().hex[:6]}"

        await api_client.upsert_users(
            UserRequest(id=user_a_id, name="User A"),
            UserRequest(id=user_b_id, name="User B"),
        )

        call = api_client.video.call("default", call_id)
        await call.get_or_create(data={"created_by_id": user_a_id})

        sub_config = SubscriptionConfig(
            default=TrackSubscriptionConfig(
                track_types=[1, 2],  # AUDIO=1, VIDEO=2
            )
        )

        # ── Collectors for received media ──────────────────────────────
        a_received_audio = []
        a_received_video_tracks = []
        b_received_audio = []
        b_received_video_tracks = []

        # ── Connect participant A first ────────────────────────────────
        cm_a = await rtc.join(call, user_id=user_a_id, create=False, subscription_config=sub_config)

        # Register listeners BEFORE entering context manager
        @cm_a.on("audio")
        def on_a_audio(pcm_data):
            a_received_audio.append(pcm_data)

        @cm_a.on("track_added")
        def on_a_track_added(track_id, kind, user):
            if kind == "video" and user and user.user_id != user_a_id:
                a_received_video_tracks.append(track_id)

        try:
            async with cm_a:
                # ── Connect participant B ──────────────────────────────
                cm_b = await rtc.join(
                    call, user_id=user_b_id, create=False, subscription_config=sub_config
                )

                @cm_b.on("audio")
                def on_b_audio(pcm_data):
                    b_received_audio.append(pcm_data)

                @cm_b.on("track_added")
                def on_b_track_added(track_id, kind, user):
                    if kind == "video" and user and user.user_id != user_b_id:
                        b_received_video_tracks.append(track_id)

                async with cm_b:
                    # Brief settle for SFU to register both participants
                    await asyncio.sleep(2)

                    # ── Publish audio + video from BOTH participants ───

                    # Participant A: audio first, then video separately
                    audio_track_a = AudioStreamTrack(sample_rate=48000, channels=1, format="s16")
                    await cm_a.add_tracks(audio=audio_track_a)

                    # Participant B: audio first, then video separately
                    audio_track_b = AudioStreamTrack(sample_rate=48000, channels=1, format="s16")
                    await cm_b.add_tracks(audio=audio_track_b)

                    # Add video tracks in a second negotiation
                    video_track_a = PipecatVideoStreamTrack(framerate=15)
                    await cm_a.add_tracks(video=video_track_a)

                    video_track_b = PipecatVideoStreamTrack(framerate=15)
                    await cm_b.add_tracks(video=video_track_b)

                    # Republish tracks so each side gets track_published
                    # events for the other's already-published tracks
                    await cm_a.republish_tracks()
                    await cm_b.republish_tracks()

                    # ── Send audio from both sides ─────────────────────
                    # Generate 2s of 440Hz tone (100 x 20ms frames)
                    for _ in range(100):
                        num_samples = 960  # 20ms at 48kHz
                        t = np.linspace(0, 0.020, num_samples, endpoint=False)
                        samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
                        pcm = PcmData(
                            samples=samples,
                            sample_rate=48000,
                            channels=1,
                            format="s16",
                        )
                        await audio_track_a.write(pcm)
                        await audio_track_b.write(pcm)

                    # ── Send video from both sides ─────────────────────
                    for i in range(10):
                        value_a = (i * 50 + 10) % 256
                        value_b = (i * 50 + 130) % 256
                        rgb_a = np.full((120, 160, 3), fill_value=value_a, dtype=np.uint8)
                        rgb_b = np.full((120, 160, 3), fill_value=value_b, dtype=np.uint8)
                        video_track_a.write(rgb_a.tobytes(), (160, 120), "RGB")
                        video_track_b.write(rgb_b.tobytes(), (160, 120), "RGB")

                    # ── Wait for media to propagate through the SFU ────
                    deadline = time.time() + 20
                    while time.time() < deadline:
                        audio_ok = a_received_audio and b_received_audio
                        video_ok = a_received_video_tracks and b_received_video_tracks
                        if audio_ok and video_ok:
                            break
                        await asyncio.sleep(0.5)

                    # ── Assertions ─────────────────────────────────────

                    # Audio: A received audio from B and vice versa
                    self.assertGreater(
                        len(a_received_audio),
                        0,
                        "Participant A did not receive any audio from B",
                    )
                    self.assertGreater(
                        len(b_received_audio),
                        0,
                        "Participant B did not receive any audio from A",
                    )

                    # Verify received audio has valid PcmData properties
                    pcm_from_b = a_received_audio[0]
                    self.assertTrue(hasattr(pcm_from_b, "samples"))
                    self.assertTrue(hasattr(pcm_from_b, "sample_rate"))
                    self.assertGreater(len(pcm_from_b.samples), 0)

                    pcm_from_a = b_received_audio[0]
                    self.assertTrue(hasattr(pcm_from_a, "samples"))
                    self.assertGreater(len(pcm_from_a.samples), 0)

                    # Video: Both sides received the other's video track
                    self.assertGreater(
                        len(a_received_video_tracks),
                        0,
                        "Participant A did not receive video track from B",
                    )
                    self.assertGreater(
                        len(b_received_video_tracks),
                        0,
                        "Participant B did not receive video track from A",
                    )

        finally:
            # ── Cleanup: delete the call ───────────────────────────────
            try:
                await call.delete(hard=True)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
