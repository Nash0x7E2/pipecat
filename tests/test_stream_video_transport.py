"""Tests for Stream Video transport implementation.

Unit tests covering two-phase track resolution, participant lifecycle,
audio routing, video gating, track cleanup, and basic transport wiring.
All tests use mocked SDK objects — no real network connections needed.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock

import numpy as np

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
    callbacks: "StreamVideoCallbacks | None" = None,
) -> "StreamVideoTransportClient":
    """Create a StreamVideoTransportClient with mocked internals.

    Args:
        video_in_enabled: Whether video input is enabled.
        audio_in_enabled: Whether audio input is enabled.
        callbacks: Optional pre-built callbacks; creates fresh AsyncMocks if None.

    Returns:
        A StreamVideoTransportClient ready for handler testing.
    """
    params = StreamVideoParams(
        video_in_enabled=video_in_enabled,
        audio_in_enabled=audio_in_enabled,
    )
    if callbacks is None:
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
    # Make create_task return a MagicMock that can be awaited (for tasks that store the result)
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
# Test Class 1: Two-Phase Track Resolution
# ---------------------------------------------------------------------------


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestStreamVideoTrackResolution(unittest.IsolatedAsyncioTestCase):
    """Tests for the two-phase track resolution logic.

    Stream Video fires separate events from the WebRTC layer (track_added)
    and the SFU (track_published). The client must match them together.
    Supports both orderings: track_added first or track_published first.
    """

    def test_track_added_then_published_resolves_video(self):
        """track_added followed by track_published should resolve and start subscriber."""
        client = _create_client(video_in_enabled=True)
        user = _make_participant("user-A", "session-1")

        # Phase 1: WebRTC fires track_added
        client._on_track_added("track-1", "video", user)
        self.assertIn("track-1", client._pending_tracks)

        # Phase 2: SFU fires track_published
        event = _make_track_published_event("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)
        client._on_track_published(event)

        # Should be resolved
        self.assertNotIn("track-1", client._pending_tracks)
        self.assertEqual(len(client._pending_publications), 0)
        self.assertIn(("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO), client._track_map)

        # A video subscriber task should have been created
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        video_tasks = [n for n in task_names if "video_subscriber" in n.lower()]
        self.assertEqual(len(video_tasks), 1)

    def test_track_published_then_added_resolves_video(self):
        """track_published followed by track_added should also resolve correctly."""
        client = _create_client(video_in_enabled=True)
        user = _make_participant("user-A", "session-1")

        # Phase 1: SFU fires track_published first (e.g. via republish_tracks)
        event = _make_track_published_event("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)
        client._on_track_published(event)
        self.assertEqual(len(client._pending_publications), 1)

        # Phase 2: WebRTC fires track_added
        client._on_track_added("track-1", "video", user)

        # Should be resolved
        self.assertEqual(len(client._pending_publications), 0)
        self.assertEqual(len(client._pending_tracks), 0)
        self.assertIn(("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO), client._track_map)

        # A video subscriber task should have been created
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        video_tasks = [n for n in task_names if "video_subscriber" in n.lower()]
        self.assertEqual(len(video_tasks), 1)

    def test_track_published_resolves_audio_track(self):
        """Audio track resolution should not start a video subscriber."""
        client = _create_client()
        user = _make_participant("user-A", "session-1")

        client._on_track_added("track-2", "audio", user)
        event = _make_track_published_event("user-A", "session-1", TrackType.TRACK_TYPE_AUDIO)
        client._on_track_published(event)

        # Resolved
        self.assertNotIn("track-2", client._pending_tracks)
        self.assertIn(("user-A", "session-1", TrackType.TRACK_TYPE_AUDIO), client._track_map)

        # No video subscriber task for audio tracks
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        video_tasks = [n for n in task_names if "video_subscriber" in n.lower()]
        self.assertEqual(video_tasks, [])

    def test_track_published_ignores_screenshare(self):
        """Screenshare tracks should be resolved but not start a video subscriber."""
        client = _create_client(video_in_enabled=True)
        user = _make_participant("user-A", "session-1")

        client._on_track_added("track-3", "video", user)
        event = _make_track_published_event(
            "user-A", "session-1", TrackType.TRACK_TYPE_SCREEN_SHARE
        )
        client._on_track_published(event)

        # Track resolved
        self.assertNotIn("track-3", client._pending_tracks)
        self.assertIn(("user-A", "session-1", TrackType.TRACK_TYPE_SCREEN_SHARE), client._track_map)

        # No video subscriber started (screenshare is excluded)
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        video_tasks = [n for n in task_names if "video_subscriber" in n.lower()]
        self.assertEqual(video_tasks, [])

    def test_track_published_no_match_stores_pending(self):
        """track_published with no matching track_added should store as pending publication."""
        client = _create_client()

        event = _make_track_published_event("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)
        client._on_track_published(event)

        self.assertEqual(len(client._pending_publications), 1)
        self.assertIn(
            ("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO), client._pending_publications
        )

    def test_track_added_ignores_bot_own_tracks(self):
        """_on_track_added should ignore tracks from the bot's own user ID."""
        client = _create_client()
        user = _make_participant("bot-user", "session-bot")

        client._on_track_added("track-self", "audio", user)

        self.assertNotIn("track-self", client._pending_tracks)

    def test_track_added_ignores_none_user(self):
        """_on_track_added should ignore tracks with no user."""
        client = _create_client()

        client._on_track_added("track-1", "audio", None)

        self.assertNotIn("track-1", client._pending_tracks)


# ---------------------------------------------------------------------------
# Test Class 2: Participant Lifecycle
# ---------------------------------------------------------------------------


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestStreamVideoParticipantLifecycle(unittest.IsolatedAsyncioTestCase):
    """Tests for participant join/leave event handling."""

    def test_participant_joined_fires_callback(self):
        """_on_participant_joined should track participant and schedule callback."""
        client = _create_client()
        event = _make_participant_event("user-A", "session-1")

        client._on_participant_joined(event)

        self.assertIn("user-A", client._participants)
        self.assertEqual(client._participants["user-A"]["session_id"], "session-1")

        # Async callback should have been scheduled
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        join_tasks = [n for n in task_names if "participant_joined" in n.lower()]
        self.assertEqual(len(join_tasks), 1)

    async def test_first_participant_joined_fires_once(self):
        """on_first_participant_joined should fire only for the very first participant."""
        client = _create_client()

        # First participant
        await client._async_on_participant_joined("user-A")
        client._callbacks.on_first_participant_joined.assert_called_once_with("user-A")
        client._callbacks.on_participant_joined.assert_called_once_with("user-A")

        # Second participant — first_joined should NOT fire again
        client._callbacks.on_first_participant_joined.reset_mock()
        await client._async_on_participant_joined("user-B")
        client._callbacks.on_first_participant_joined.assert_not_called()

    def test_participant_left_cleans_up_state(self):
        """_on_participant_left should remove participant and fire callbacks."""
        client = _create_client()

        # Simulate join first
        client._participants["user-A"] = {"session_id": "session-1"}
        client._audio_subscribed_participants.add("user-A")
        client._other_participant_has_joined = True

        event = _make_participant_event("user-A", "session-1")
        client._on_participant_left(event)

        self.assertNotIn("user-A", client._participants)
        self.assertNotIn("user-A", client._audio_subscribed_participants)

        # Callbacks should have been scheduled
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        left_tasks = [n for n in task_names if "participant_left" in n.lower()]
        unsub_tasks = [n for n in task_names if "audio_track_unsubscribed" in n.lower()]
        self.assertGreaterEqual(len(left_tasks), 1)
        self.assertGreaterEqual(len(unsub_tasks), 1)

    def test_participant_left_resets_first_join_when_empty(self):
        """After all participants leave, _other_participant_has_joined resets."""
        client = _create_client()

        client._participants["user-A"] = {"session_id": "session-1"}
        client._other_participant_has_joined = True

        event = _make_participant_event("user-A", "session-1")
        client._on_participant_left(event)

        self.assertFalse(client._other_participant_has_joined)

    def test_participant_joined_ignores_bot_user(self):
        """_on_participant_joined should ignore the bot's own user ID."""
        client = _create_client()
        event = _make_participant_event("bot-user", "session-bot")

        client._on_participant_joined(event)

        self.assertNotIn("bot-user", client._participants)
        client._task_manager.create_task.assert_not_called()


# ---------------------------------------------------------------------------
# Test Class 3: Audio Routing
# ---------------------------------------------------------------------------


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestStreamVideoAudioRouting(unittest.IsolatedAsyncioTestCase):
    """Tests for audio event handling and queue routing."""

    def test_audio_event_queues_data(self):
        """_on_audio should enqueue (pcm_data, user_id) tuples."""
        client = _create_client()
        mock_pcm = _make_pcm_data("user-A")

        client._on_audio(mock_pcm)

        self.assertEqual(client._audio_queue.qsize(), 1)
        pcm_data, uid = client._audio_queue.get_nowait()
        self.assertIs(pcm_data, mock_pcm)
        self.assertEqual(uid, "user-A")

    def test_audio_event_ignores_bot_own_audio(self):
        """_on_audio should not enqueue audio from the bot itself."""
        client = _create_client()
        mock_pcm = _make_pcm_data("bot-user")

        client._on_audio(mock_pcm)

        self.assertEqual(client._audio_queue.qsize(), 0)

    def test_audio_event_ignores_no_participant(self):
        """_on_audio should not enqueue audio without a participant."""
        client = _create_client()
        mock_pcm = MagicMock(spec=[])  # No .participant attribute

        client._on_audio(mock_pcm)

        self.assertEqual(client._audio_queue.qsize(), 0)

    def test_first_audio_fires_subscription_callback(self):
        """First audio from a participant should fire on_audio_track_subscribed."""
        client = _create_client()
        mock_pcm = _make_pcm_data("user-A")

        client._on_audio(mock_pcm)

        self.assertIn("user-A", client._audio_subscribed_participants)
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        sub_tasks = [n for n in task_names if "audio_track_subscribed" in n.lower()]
        self.assertEqual(len(sub_tasks), 1)

    def test_subsequent_audio_does_not_refire_subscription(self):
        """Subsequent audio from the same participant should not re-fire subscription."""
        client = _create_client()

        client._on_audio(_make_pcm_data("user-A"))
        client._on_audio(_make_pcm_data("user-A"))
        client._on_audio(_make_pcm_data("user-A"))

        # Should have 3 audio items queued
        self.assertEqual(client._audio_queue.qsize(), 3)

        # But subscription callback only fired once
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        sub_tasks = [n for n in task_names if "audio_track_subscribed" in n.lower()]
        self.assertEqual(len(sub_tasks), 1)


# ---------------------------------------------------------------------------
# Test Class 4: Video Gating
# ---------------------------------------------------------------------------


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestStreamVideoVideoGating(unittest.IsolatedAsyncioTestCase):
    """Tests for video subscriber gating based on video_in_enabled."""

    async def test_video_disabled_does_not_start_receive_loop(self):
        """When video_in_enabled=False, no video receive loop task should start."""
        client = _create_client(video_in_enabled=False)
        client._connection = None

        await client._start_video_subscriber("track-1", "user-A")

        # No video receive loop task should have been created
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        loop_tasks = [n for n in task_names if "video_receive_loop" in n.lower()]
        self.assertEqual(loop_tasks, [])

    async def test_video_disabled_still_fires_callback(self):
        """Even when video input is disabled, on_video_track_subscribed should fire."""
        client = _create_client(video_in_enabled=False)
        client._connection = None

        await client._start_video_subscriber("track-1", "user-A")

        self.assertIn("user-A", client._video_subscribed_participants)
        client._callbacks.on_video_track_subscribed.assert_called_once_with("user-A")

    async def test_video_enabled_starts_receive_loop(self):
        """When video_in_enabled=True and connection exists, receive loop should start."""
        client = _create_client(video_in_enabled=True)

        # Mock the connection's subscriber_pc
        mock_video_track = MagicMock()
        mock_subscriber_pc = MagicMock()
        mock_subscriber_pc.add_track_subscriber.return_value = mock_video_track
        mock_connection = MagicMock()
        mock_connection.subscriber_pc = mock_subscriber_pc
        client._connection = mock_connection

        await client._start_video_subscriber("track-1", "user-A")

        # Callback should fire
        self.assertIn("user-A", client._video_subscribed_participants)
        client._callbacks.on_video_track_subscribed.assert_called_once_with("user-A")

        # Video receive loop task should have been created
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        loop_tasks = [n for n in task_names if "video_receive_loop" in n.lower()]
        self.assertEqual(len(loop_tasks), 1)

        # Subscriber task should be stored
        self.assertIn("user-A:track-1", client._video_subscriber_tasks)


# ---------------------------------------------------------------------------
# Test Class 5: Track Cleanup
# ---------------------------------------------------------------------------


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestStreamVideoTrackCleanup(unittest.IsolatedAsyncioTestCase):
    """Tests for track unpublish and participant leave cleanup."""

    def test_track_unpublished_removes_track_map_entry(self):
        """_on_track_unpublished should remove the entry from _track_map."""
        client = _create_client()
        client._track_map[("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)] = "track-1"

        event = _make_track_unpublished_event("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)
        client._on_track_unpublished(event)

        self.assertNotIn(("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO), client._track_map)

    def test_track_unpublished_cancels_video_task(self):
        """_on_track_unpublished should cancel the video subscriber task."""
        client = _create_client()
        client._track_map[("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)] = "track-1"
        mock_task = MagicMock()
        client._video_subscriber_tasks["user-A:track-1"] = mock_task
        client._video_subscribed_participants.add("user-A")

        event = _make_track_unpublished_event("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)
        client._on_track_unpublished(event)

        mock_task.cancel.assert_called_once()
        self.assertNotIn("user-A:track-1", client._video_subscriber_tasks)

    def test_track_unpublished_fires_unsubscribe_callback(self):
        """_on_track_unpublished should fire video unsubscribe callback for video tracks."""
        client = _create_client()
        client._track_map[("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)] = "track-1"
        client._video_subscribed_participants.add("user-A")

        event = _make_track_unpublished_event("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)
        client._on_track_unpublished(event)

        self.assertNotIn("user-A", client._video_subscribed_participants)
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        unsub_tasks = [n for n in task_names if "video_track_unsubscribed" in n.lower()]
        self.assertGreaterEqual(len(unsub_tasks), 1)

    def test_track_unpublished_fires_audio_unsubscribe_for_audio(self):
        """_on_track_unpublished should fire audio unsubscribe callback for audio tracks."""
        client = _create_client()
        client._track_map[("user-A", "session-1", TrackType.TRACK_TYPE_AUDIO)] = "track-2"
        client._audio_subscribed_participants.add("user-A")

        event = _make_track_unpublished_event("user-A", "session-1", TrackType.TRACK_TYPE_AUDIO)
        client._on_track_unpublished(event)

        self.assertNotIn("user-A", client._audio_subscribed_participants)
        task_names = [call[0][1] for call in client._task_manager.create_task.call_args_list]
        unsub_tasks = [n for n in task_names if "audio_track_unsubscribed" in n.lower()]
        self.assertGreaterEqual(len(unsub_tasks), 1)

    def test_participant_left_cancels_video_subscriber_tasks(self):
        """_on_participant_left should cancel video subscriber tasks for that participant."""
        client = _create_client()
        client._participants["user-A"] = {"session_id": "session-1"}
        mock_task = MagicMock()
        client._video_subscriber_tasks["user-A:track-1"] = mock_task
        client._video_subscribed_participants.add("user-A")

        event = _make_participant_event("user-A", "session-1")
        client._on_participant_left(event)

        mock_task.cancel.assert_called_once()
        self.assertNotIn("user-A:track-1", client._video_subscriber_tasks)


# ---------------------------------------------------------------------------
# Test Class 6: Transport Wiring (Integration)
# ---------------------------------------------------------------------------


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestStreamVideoTransportWiring(unittest.IsolatedAsyncioTestCase):
    """Integration tests verifying the facade correctly wires client, input, and output."""

    def test_transport_creates_client_with_credentials(self):
        """StreamVideoTransport should create a client with the given credentials."""
        transport = StreamVideoTransport(
            api_key="test-key",
            api_secret="test-secret",
            call_type="default",
            call_id="test-call-123",
            user_id="bot-user",
        )

        self.assertEqual(transport._client._api_key, "test-key")
        self.assertEqual(transport._client._api_secret, "test-secret")
        self.assertEqual(transport._client._call_type, "default")
        self.assertEqual(transport._client._call_id, "test-call-123")
        self.assertEqual(transport._client._user_id, "bot-user")

    def test_transport_input_returns_input_transport(self):
        """transport.input() should return a StreamVideoInputTransport."""
        from pipecat.transports.stream_video.transport import StreamVideoInputTransport

        transport = StreamVideoTransport(
            api_key="k", api_secret="s", call_type="default", call_id="c", user_id="u"
        )

        inp = transport.input()
        self.assertIsInstance(inp, StreamVideoInputTransport)
        # Calling input() again should return the same instance
        self.assertIs(transport.input(), inp)

    def test_transport_output_returns_output_transport(self):
        """transport.output() should return a StreamVideoOutputTransport."""
        from pipecat.transports.stream_video.transport import StreamVideoOutputTransport

        transport = StreamVideoTransport(
            api_key="k", api_secret="s", call_type="default", call_id="c", user_id="u"
        )

        out = transport.output()
        self.assertIsInstance(out, StreamVideoOutputTransport)
        self.assertIs(transport.output(), out)

    def test_transport_participant_id(self):
        """transport.participant_id should return the bot's user ID."""
        transport = StreamVideoTransport(
            api_key="k", api_secret="s", call_type="default", call_id="c", user_id="my-bot"
        )

        self.assertEqual(transport.participant_id, "my-bot")

    def test_transport_get_participants_excludes_bot(self):
        """get_participants() should not include the bot's own user ID."""
        transport = StreamVideoTransport(
            api_key="k", api_secret="s", call_type="default", call_id="c", user_id="bot-user"
        )

        # Simulate participants
        transport._client._participants["user-A"] = {"session_id": "s1"}
        transport._client._participants["user-B"] = {"session_id": "s2"}
        transport._client._participants["bot-user"] = {"session_id": "s3"}

        participants = transport.get_participants()
        self.assertIn("user-A", participants)
        self.assertIn("user-B", participants)
        self.assertNotIn("bot-user", participants)
        self.assertEqual(len(participants), 2)

    def test_transport_default_params(self):
        """StreamVideoTransport should use default StreamVideoParams when none provided."""
        transport = StreamVideoTransport(
            api_key="k", api_secret="s", call_type="default", call_id="c", user_id="u"
        )

        self.assertIsInstance(transport._params, StreamVideoParams)

    def test_transport_custom_params(self):
        """StreamVideoTransport should use provided custom params."""
        params = StreamVideoParams(audio_in_enabled=False, video_in_enabled=True)
        transport = StreamVideoTransport(
            api_key="k",
            api_secret="s",
            call_type="default",
            call_id="c",
            user_id="u",
            params=params,
        )

        self.assertFalse(transport._params.audio_in_enabled)
        self.assertTrue(transport._params.video_in_enabled)

    def test_client_shared_between_input_and_output(self):
        """Input and output transports should share the same client instance."""
        transport = StreamVideoTransport(
            api_key="k", api_secret="s", call_type="default", call_id="c", user_id="u"
        )

        inp = transport.input()
        out = transport.output()

        self.assertIs(inp._client, out._client)
        self.assertIs(inp._client, transport._client)


# ---------------------------------------------------------------------------
# Test Class 7: PipecatVideoStreamTrack
# ---------------------------------------------------------------------------


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestPipecatVideoStreamTrack(unittest.IsolatedAsyncioTestCase):
    """Tests for the custom video output track used for WebRTC publishing."""

    def test_track_kind_is_video(self):
        """Track kind should be 'video'."""
        track = PipecatVideoStreamTrack(framerate=30)
        self.assertEqual(track.kind, "video")

    def test_write_enqueues_frame(self):
        """write() should convert image bytes and enqueue an av.VideoFrame."""
        track = PipecatVideoStreamTrack(framerate=30)
        # Create a 4x4 RGB image (48 bytes)
        image = np.zeros((4, 4, 3), dtype=np.uint8).tobytes()

        track.write(image, (4, 4), "RGB")

        self.assertEqual(track._queue.qsize(), 1)

    def test_write_increments_pts(self):
        """Each write() should increment PTS by time_base_den / framerate."""
        track = PipecatVideoStreamTrack(framerate=30)
        image = np.zeros((4, 4, 3), dtype=np.uint8).tobytes()

        self.assertEqual(track._pts, 0)
        track.write(image, (4, 4), "RGB")
        self.assertEqual(track._pts, 3000)  # 90000 / 30
        track.write(image, (4, 4), "RGB")
        self.assertEqual(track._pts, 6000)

    async def test_recv_returns_written_frame(self):
        """recv() should return a frame that was written."""
        track = PipecatVideoStreamTrack(framerate=30)
        image = np.ones((4, 4, 3), dtype=np.uint8).tobytes()

        track.write(image, (4, 4), "RGB")
        frame = await track.recv()

        arr = frame.to_ndarray(format="rgb24")
        self.assertEqual(arr.shape, (4, 4, 3))
        np.testing.assert_array_equal(arr, np.ones((4, 4, 3), dtype=np.uint8))

    async def test_recv_returns_black_frame_when_empty(self):
        """recv() should return a black frame when queue is empty and no last frame."""
        track = PipecatVideoStreamTrack(framerate=30)
        frame = await track.recv()

        arr = frame.to_ndarray(format="rgb24")
        self.assertEqual(arr.shape, (480, 640, 3))
        np.testing.assert_array_equal(arr, np.zeros((480, 640, 3), dtype=np.uint8))

    async def test_recv_holds_last_frame_when_empty(self):
        """recv() should repeat the last frame when queue empties."""
        track = PipecatVideoStreamTrack(framerate=30)
        image = np.full((4, 4, 3), 128, dtype=np.uint8).tobytes()

        track.write(image, (4, 4), "RGB")
        first = await track.recv()  # Consumes the queued frame
        second = await track.recv()  # Queue empty, should hold last

        arr = second.to_ndarray(format="rgb24")
        np.testing.assert_array_equal(arr, np.full((4, 4, 3), 128, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Test Class 8: End-to-End Event Flow (Integration)
# ---------------------------------------------------------------------------


@unittest.skipUnless(STREAM_VIDEO_AVAILABLE, "getstream[webrtc] package not installed")
class TestStreamVideoEndToEndEventFlow(unittest.IsolatedAsyncioTestCase):
    """Integration test simulating a complete participant session lifecycle.

    Verifies that a sequence of events (join -> audio -> track add/publish ->
    track unpublish -> leave) flows through the client correctly.
    """

    async def test_full_participant_session(self):
        """Simulate a complete participant session from join to leave."""
        client = _create_client(video_in_enabled=True)
        user = _make_participant("user-A", "session-1")

        # 1. Participant joins
        join_event = _make_participant_event("user-A", "session-1")
        client._on_participant_joined(join_event)
        self.assertIn("user-A", client._participants)

        # Fire the async part directly
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

    async def test_multiple_participants_concurrent(self):
        """Multiple participants can join, have tracks, and leave independently."""
        client = _create_client(video_in_enabled=True)

        # Two participants join
        client._on_participant_joined(_make_participant_event("user-A", "session-1"))
        client._on_participant_joined(_make_participant_event("user-B", "session-2"))
        self.assertEqual(len(client.get_participants()), 2)

        # Both send audio
        client._on_audio(_make_pcm_data("user-A"))
        client._on_audio(_make_pcm_data("user-B"))
        self.assertEqual(client._audio_queue.qsize(), 2)

        # Both have video tracks
        user_a = _make_participant("user-A", "session-1")
        user_b = _make_participant("user-B", "session-2")
        client._on_track_added("vt-A", "video", user_a)
        client._on_track_added("vt-B", "video", user_b)
        client._on_track_published(
            _make_track_published_event("user-A", "session-1", TrackType.TRACK_TYPE_VIDEO)
        )
        client._on_track_published(
            _make_track_published_event("user-B", "session-2", TrackType.TRACK_TYPE_VIDEO)
        )

        self.assertEqual(len(client._track_map), 2)
        self.assertEqual(len(client._pending_tracks), 0)

        # User A leaves — user B should still be tracked
        client._on_participant_left(_make_participant_event("user-A", "session-1"))
        self.assertNotIn("user-A", client._participants)
        self.assertIn("user-B", client._participants)
        self.assertEqual(len(client.get_participants()), 1)

        # Verify user-B is still there
        self.assertIn("user-B", client._participants)


if __name__ == "__main__":
    unittest.main()
