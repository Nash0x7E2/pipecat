"""Stream Video transport implementation for Pipecat.

This module provides Stream Video real-time communication integration
including audio streaming, video publishing/subscribing, participant management,
and call event handling for conversational AI applications.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    UserAudioRawFrame,
    UserImageRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.asyncio.task_manager import BaseTaskManager

try:
    import av
    from aiortc import MediaStreamTrack
    from getstream import AsyncStream
    from getstream.models import UserRequest
    from getstream.video import rtc
    from getstream.video.rtc import AudioStreamTrack, PcmData
    from getstream.video.rtc.pb.stream.video.sfu.models.models_pb2 import TrackType
    from getstream.video.rtc.tracks import SubscriptionConfig, TrackSubscriptionConfig
except ModuleNotFoundError as _e:
    logger.error(f"Exception: {_e}")
    logger.error(
        "In order to use Stream Video, you need to `pip install pipecat-ai[stream-video]`."
    )
    raise Exception(f"Missing module: {_e}")


@dataclass
class GetstreamOutputTransportMessageFrame(OutputTransportMessageFrame):
    """Frame for transport messages in Stream Video calls.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


@dataclass
class GetstreamOutputTransportMessageUrgentFrame(OutputTransportMessageUrgentFrame):
    """Frame for urgent transport messages in Stream Video calls.

    Parameters:
        participant_id: Optional ID of the participant this message is for/from.
    """

    participant_id: Optional[str] = None


class GetstreamParams(TransportParams):
    """Configuration parameters for Stream Video transport.

    Inherits all parameters from TransportParams without additional configuration.
    """

    pass


class GetstreamCallbacks(BaseModel):
    """Callback handlers for Stream Video events.

    Parameters:
        on_connected: Called when connected to the Stream Video call.
        on_disconnected: Called when disconnected from the call.
        on_before_disconnect: Called just before disconnecting from the call.
        on_participant_joined: Called when a participant joins the call.
        on_participant_left: Called when a participant leaves the call.
        on_audio_track_subscribed: Called when an audio track is subscribed.
        on_audio_track_unsubscribed: Called when an audio track is unsubscribed.
        on_video_track_subscribed: Called when a video track is subscribed.
        on_video_track_unsubscribed: Called when a video track is unsubscribed.
        on_data_received: Called when data is received from a participant.
        on_first_participant_joined: Called when the first participant joins.
    """

    on_connected: Callable[[], Awaitable[None]]
    on_disconnected: Callable[[], Awaitable[None]]
    on_before_disconnect: Callable[[], Awaitable[None]]
    on_participant_joined: Callable[[str], Awaitable[None]]
    on_participant_left: Callable[[str], Awaitable[None]]
    on_audio_track_subscribed: Callable[[str], Awaitable[None]]
    on_audio_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_video_track_subscribed: Callable[[str], Awaitable[None]]
    on_video_track_unsubscribed: Callable[[str], Awaitable[None]]
    on_data_received: Callable[[bytes, str], Awaitable[None]]
    on_first_participant_joined: Callable[[str], Awaitable[None]]


class PipecatVideoStreamTrack(MediaStreamTrack):
    """Custom aiortc MediaStreamTrack for publishing video from Pipecat pipeline.

    Bridges Pipecat's OutputImageRawFrame into the WebRTC video publishing
    mechanism used by Stream Video's ConnectionManager.
    """

    kind = "video"

    def __init__(self, framerate: int = 30):
        """Initialize the video stream track.

        Args:
            framerate: Target video framerate in FPS.
        """
        super().__init__()
        self._framerate = framerate
        self._queue: asyncio.Queue = asyncio.Queue()
        self._start_time: Optional[float] = None
        self._frame_count = 0
        self._last_frame: Optional[av.VideoFrame] = None
        self._pts = 0
        self._time_base_den = 90000  # Standard WebRTC clock rate

    def write(self, image: bytes, size: tuple, format: Optional[str]):
        """Write an image frame to the track for WebRTC publishing.

        Args:
            image: Raw image bytes from Pipecat's OutputImageRawFrame.
            size: Tuple of (width, height) in pixels.
            format: Image format string (e.g. "RGB").
        """
        width, height = size
        try:
            array = np.frombuffer(image, dtype=np.uint8).reshape(height, width, 3)
            frame = av.VideoFrame.from_ndarray(array, format=format)
            frame.pts = self._pts
            frame.time_base = Fraction(1, self._time_base_den)
            self._pts += int(self._time_base_den / self._framerate)
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass
        except Exception:
            logger.exception(f"Error converting image to video frame")

    async def recv(self) -> av.VideoFrame:
        """Receive the next video frame for WebRTC publishing.

        Called by aiortc's internals to pull frames. Maintains proper timing
        using the 90kHz WebRTC clock.

        Returns:
            The next av.VideoFrame to publish.
        """
        if self._start_time is None:
            self._start_time = time.time()

        # Calculate target time for this frame
        target_time = self._start_time + (self._frame_count / self._framerate)
        now = time.time()
        delay = target_time - now
        if delay > 0:
            await asyncio.sleep(delay)

        self._frame_count += 1

        try:
            frame = self._queue.get_nowait()
            self._last_frame = frame
            return frame
        except asyncio.QueueEmpty:
            # Hold last frame or send black frame
            if self._last_frame is not None:
                held = av.VideoFrame.from_ndarray(
                    self._last_frame.to_ndarray(format="rgb24"), format="rgb24"
                )
                held.pts = self._pts
                held.time_base = Fraction(1, self._time_base_den)
                self._pts += int(self._time_base_den / self._framerate)
                return held
            else:
                # Send a small black frame as placeholder
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(black, format="rgb24")
                frame.pts = self._pts
                frame.time_base = Fraction(1, self._time_base_den)
                self._pts += int(self._time_base_den / self._framerate)
                return frame


class GetstreamTransportClient:
    """Core client for interacting with Stream Video calls.

    Manages the WebRTC connection to Stream Video's SFU and handles all low-level
    interactions including audio/video streaming, track management, and event handling.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        call_type: str,
        call_id: str,
        user_id: str,
        params: GetstreamParams,
        callbacks: GetstreamCallbacks,
        transport_name: str,
    ):
        """Initialize the Stream Video transport client.

        Args:
            api_key: Stream Video API key.
            api_secret: Stream Video API secret.
            call_type: The Stream call type (e.g. "default").
            call_id: Unique call identifier.
            user_id: The bot/agent user ID.
            params: Configuration parameters for the transport.
            callbacks: Event callback handlers.
            transport_name: Name identifier for the transport.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._call_type = call_type
        self._call_id = call_id
        self._user_id = user_id
        self._params = params
        self._callbacks = callbacks
        self._transport_name = transport_name

        self._client: Optional[AsyncStream] = None
        self._call = None
        self._connection = None
        self._audio_track: Optional[AudioStreamTrack] = None
        self._video_track: Optional[PipecatVideoStreamTrack] = None
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._video_queue: asyncio.Queue = asyncio.Queue()
        self._connected = False
        self._disconnect_counter = 0
        self._other_participant_has_joined = False
        self._task_manager: Optional[BaseTaskManager] = None
        self._async_lock = asyncio.Lock()

        # Two-phase track resolution state (bidirectional matching)
        self._pending_tracks: Dict[str, dict] = {}  # track_added arrived, awaiting track_published
        self._pending_publications: Dict[
            tuple, dict
        ] = {}  # track_published arrived, awaiting track_added
        self._track_map: Dict[tuple, str] = {}
        self._video_subscriber_tasks: Dict[str, asyncio.Task] = {}
        self._participants: Dict[str, dict] = {}
        self._audio_subscribed_participants: set = set()
        self._video_subscribed_participants: set = set()

        self._out_sample_rate = self._params.audio_out_sample_rate

    @property
    def participant_id(self) -> str:
        """Get the bot's user ID.

        Returns:
            The user ID used by this client.
        """
        return self._user_id

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the client with task manager and API client initialization.

        Args:
            setup: The frame processor setup configuration.
        """
        if self._task_manager:
            return

        self._task_manager = setup.task_manager
        self._client = AsyncStream(api_key=self._api_key, api_secret=self._api_secret)

        # Ensure the bot user exists
        try:
            await self._client.upsert_users(UserRequest(id=self._user_id, name=self._user_id))
        except Exception as exc:
            logger.warning(f"Could not create user {self._user_id}: {exc}")

    async def cleanup(self):
        """Clean up client resources."""
        await self.disconnect()

    async def start(self, frame: StartFrame):
        """Start the client and store output sample rate.

        Args:
            frame: The start frame containing initialization parameters.
        """
        self._out_sample_rate = self._out_sample_rate or frame.audio_out_sample_rate

    async def connect(self):
        """Connect to the Stream Video call.

        Creates the call, joins the SFU, registers event handlers,
        and publishes audio/video tracks.
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Was setup() called?")

        async with self._async_lock:
            if self._connected:
                self._disconnect_counter += 1
                return

            logger.info(f"Connecting to Stream Video call {self._call_type}:{self._call_id}")

            try:
                # Create or get the call
                self._call = self._client.video.call(self._call_type, self._call_id)
                await self._call.get_or_create(data={"created_by_id": self._user_id})

                # Configure subscription to receive audio and video tracks
                subscription_config = SubscriptionConfig(
                    default=TrackSubscriptionConfig(
                        track_types=[
                            TrackType.TRACK_TYPE_AUDIO,
                            TrackType.TRACK_TYPE_VIDEO,
                        ]
                    )
                )

                # Join the call via WebRTC
                self._connection = await rtc.join(
                    self._call,
                    self._user_id,
                    subscription_config=subscription_config,
                )

                # Register event handlers before connecting
                self._connection.on("audio")(self._on_audio)
                self._connection.on("track_added")(self._on_track_added)
                self._connection.on("participant_joined")(self._on_participant_joined)
                self._connection.on("participant_left")(self._on_participant_left)
                self._connection.on("track_published")(self._on_track_published)
                self._connection.on("track_unpublished")(self._on_track_unpublished)
                self._connection.on("call_ended")(self._on_call_ended)

                # Establish the WebRTC connection
                await self._connection.__aenter__()

                # Republish any existing tracks
                await self._connection.republish_tracks()

                # Create and publish audio track
                in_sample_rate = self._params.audio_in_sample_rate or 24000
                self._audio_track = AudioStreamTrack(sample_rate=in_sample_rate)

                # Create video track if video output is enabled
                if self._params.video_out_enabled:
                    self._video_track = PipecatVideoStreamTrack(
                        framerate=self._params.video_out_framerate
                    )
                    await self._connection.add_tracks(
                        audio=self._audio_track, video=self._video_track
                    )
                else:
                    await self._connection.add_tracks(audio=self._audio_track)

                self._connected = True
                self._disconnect_counter += 1

                logger.info(f"Connected to Stream Video call {self._call_type}:{self._call_id}")

                await self._callbacks.on_connected()

            except Exception:
                logger.exception(
                    f"Error connecting to Stream Video call {self._call_type}:{self._call_id}"
                )
                raise

    async def disconnect(self):
        """Disconnect from the Stream Video call."""
        async with self._async_lock:
            self._disconnect_counter -= 1

            if not self._connected or self._disconnect_counter > 0:
                return

            logger.info(f"Disconnecting from Stream Video call {self._call_type}:{self._call_id}")

            await self._callbacks.on_before_disconnect()

            # Cancel all video subscriber tasks
            for task_id, task in self._video_subscriber_tasks.items():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self._video_subscriber_tasks.clear()

            # Leave the connection
            if self._connection:
                try:
                    await asyncio.wait_for(self._connection.leave(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout leaving Stream Video call, forcing disconnect")
                except Exception as exc:
                    logger.warning(f"Error leaving Stream Video call: {exc}")

            self._connected = False
            self._connection = None
            self._audio_track = None
            self._video_track = None
            self._pending_tracks.clear()
            self._pending_publications.clear()
            self._track_map.clear()
            self._participants.clear()
            self._audio_subscribed_participants.clear()
            self._video_subscribed_participants.clear()

            logger.info(f"Disconnected from Stream Video call {self._call_type}:{self._call_id}")

            await self._callbacks.on_disconnected()

    async def send_data(self, data: bytes, participant_id: Optional[str] = None):
        """Send custom event data to participants in the call.

        Args:
            data: The data bytes to send.
            participant_id: Optional specific participant to target.
        """
        if not self._connected or not self._call:
            return

        try:
            custom_data = json.loads(data.decode()) if isinstance(data, bytes) else data
            await self._call.send_call_event(user_id=self._user_id, custom=custom_data)
        except Exception:
            logger.exception("Error sending data")

    def get_participants(self) -> List[str]:
        """Get list of participant IDs in the call.

        Returns:
            List of participant user IDs (excluding the bot).
        """
        return [uid for uid in self._participants.keys() if uid != self._user_id]

    async def get_next_audio_frame(self):
        """Get the next audio frame from the queue.

        Yields:
            Tuple of (PcmData, participant_id) for each received audio frame.
        """
        while True:
            pcm_data, participant_id = await self._audio_queue.get()
            yield pcm_data, participant_id

    async def get_next_video_frame(self):
        """Get the next video frame from the queue.

        Yields:
            Tuple of (rgb_ndarray, participant_id) for each received video frame.
        """
        while True:
            rgb_array, participant_id = await self._video_queue.get()
            yield rgb_array, participant_id

    async def flush_audio(self):
        """Flush the audio track buffer (e.g. on interruption)."""
        if self._connected and self._audio_track:
            await self._audio_track.flush()

    async def write_audio(self, pcm_data: PcmData) -> bool:
        """Write PCM audio data to the audio track.

        Args:
            pcm_data: The PCM audio data to write.

        Returns:
            True if written, False if not connected or no track.
        """
        if not self._connected or not self._audio_track:
            return False
        await self._audio_track.write(pcm_data)
        return True

    def write_video(self, image: bytes, size: tuple, format: Optional[str]) -> bool:
        """Write a video frame to the video track.

        Args:
            image: Raw image bytes.
            size: Tuple of (width, height) in pixels.
            format: Image format string (e.g. "RGB"), or None.

        Returns:
            True if written, False if not connected or no track.
        """
        if not self._connected or not self._video_track:
            return False
        self._video_track.write(image, size, format)
        return True

    # Event handlers

    def _on_audio(self, pcm_data: PcmData):
        """Handle incoming audio from a participant.

        Args:
            pcm_data: The PCM audio data with .participant attribute set by the SDK.
        """
        participant = getattr(pcm_data, "participant", None)
        if participant is None:
            return
        user_id = participant.user_id
        if user_id == self._user_id:
            return
        try:
            self._audio_queue.put_nowait((pcm_data, user_id))
        except asyncio.QueueFull:
            pass

        # Track audio subscription on first audio from this participant
        if user_id not in self._audio_subscribed_participants:
            self._audio_subscribed_participants.add(user_id)
            self._create_task(
                self._callbacks.on_audio_track_subscribed(user_id),
                f"{self}::on_audio_track_subscribed",
            )

    def _on_track_added(self, track_source_id: str, kind: str, user):
        """Handle WebRTC track added event (phase 1 of track resolution).

        Checks for a matching pending publication (if track_published arrived first)
        or stores the track in pending state awaiting SFU type confirmation.

        Args:
            track_source_id: The original track source identifier.
            kind: The WebRTC track kind ("audio" or "video").
            user: The Participant protobuf object (or None).
        """
        if user is None or user.user_id == self._user_id:
            return

        user_id = user.user_id
        session_id = user.session_id

        # Check for matching pending publication (track_published arrived first)
        matched_pub_key = None
        for pub_key in self._pending_publications:
            pub_user_id, pub_session_id, track_type = pub_key
            if pub_user_id == user_id and pub_session_id == session_id:
                pub_kind = (
                    "video"
                    if track_type in (TrackType.TRACK_TYPE_VIDEO, TrackType.TRACK_TYPE_SCREEN_SHARE)
                    else "audio"
                )
                if pub_kind == kind:
                    matched_pub_key = pub_key
                    break

        if matched_pub_key:
            self._pending_publications.pop(matched_pub_key)
            track_type = matched_pub_key[2]
            self._resolve_track(track_source_id, user_id, session_id, track_type)
        else:
            self._pending_tracks[track_source_id] = {
                "user_id": user_id,
                "session_id": session_id,
                "kind": kind,
            }
            logger.debug(f"Track added (pending): {track_source_id} from {user_id} kind={kind}")

    def _on_participant_joined(self, event):
        """Handle participant joined event.

        Args:
            event: The ParticipantJoined protobuf event from the SFU.
        """
        participant = event.participant
        user_id = participant.user_id
        session_id = participant.session_id
        if user_id == self._user_id:
            return

        logger.info(f"Participant joined: {user_id}")
        self._participants[user_id] = {"session_id": session_id}
        self._create_task(
            self._async_on_participant_joined(user_id),
            f"{self}::_async_on_participant_joined",
        )

    async def _async_on_participant_joined(self, user_id: str):
        """Async handler for participant joined event.

        Args:
            user_id: The participant's user ID.
        """
        await self._callbacks.on_participant_joined(user_id)
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._callbacks.on_first_participant_joined(user_id)

    def _on_participant_left(self, event):
        """Handle participant left event.

        Args:
            event: The ParticipantLeft protobuf event from the SFU.
        """
        participant = event.participant
        user_id = participant.user_id
        if user_id == self._user_id:
            return

        logger.info(f"Participant left: {user_id}")
        self._participants.pop(user_id, None)

        # Clean up subscriptions for this participant
        if user_id in self._audio_subscribed_participants:
            self._audio_subscribed_participants.discard(user_id)
            self._create_task(
                self._callbacks.on_audio_track_unsubscribed(user_id),
                f"{self}::on_audio_track_unsubscribed",
            )

        if user_id in self._video_subscribed_participants:
            self._video_subscribed_participants.discard(user_id)
            self._create_task(
                self._callbacks.on_video_track_unsubscribed(user_id),
                f"{self}::on_video_track_unsubscribed",
            )

        # Cancel any video subscriber tasks for this participant
        tasks_to_cancel = [
            (tid, task)
            for tid, task in self._video_subscriber_tasks.items()
            if tid.startswith(f"{user_id}:")
        ]
        for tid, task in tasks_to_cancel:
            task.cancel()
            self._video_subscriber_tasks.pop(tid, None)

        self._create_task(
            self._callbacks.on_participant_left(user_id),
            f"{self}::on_participant_left",
        )

        if len(self.get_participants()) == 0:
            self._other_participant_has_joined = False

    def _on_track_published(self, event):
        """Handle SFU track published event (phase 2 of track resolution).

        Checks for a matching pending track (if track_added arrived first)
        or stores as pending publication awaiting track_added.

        Args:
            event: The TrackPublished protobuf event from the SFU.
        """
        user_id = event.user_id
        session_id = event.session_id
        track_type = event.type  # int TrackType enum

        if user_id == self._user_id:
            return

        expected_kind = (
            "video"
            if track_type in (TrackType.TRACK_TYPE_VIDEO, TrackType.TRACK_TYPE_SCREEN_SHARE)
            else "audio"
        )

        # Find matching pending track (track_added arrived first)
        matched_track_id = None
        for track_id, info in self._pending_tracks.items():
            if (
                info["user_id"] == user_id
                and info["session_id"] == session_id
                and info["kind"] == expected_kind
            ):
                matched_track_id = track_id
                break

        if matched_track_id:
            self._pending_tracks.pop(matched_track_id)
            self._resolve_track(matched_track_id, user_id, session_id, track_type)
        else:
            # Store as pending publication, waiting for track_added
            pub_key = (user_id, session_id, track_type)
            self._pending_publications[pub_key] = {"track_type": track_type}
            logger.debug(f"Track published (pending): {user_id}/{session_id}/{track_type}")

    def _resolve_track(self, track_source_id: str, user_id: str, session_id: str, track_type: int):
        """Resolve a track after both track_added and track_published have been received.

        Args:
            track_source_id: The original track source identifier.
            user_id: The participant's user ID.
            session_id: The participant's session ID.
            track_type: The TrackType int enum value.
        """
        self._track_map[(user_id, session_id, track_type)] = track_source_id
        logger.debug(f"Track resolved: {track_source_id} from {user_id} type={track_type}")

        # Start video subscriber only for TRACK_TYPE_VIDEO (not screenshare)
        if track_type == TrackType.TRACK_TYPE_VIDEO:
            self._create_task(
                self._start_video_subscriber(track_source_id, user_id),
                f"{self}::_start_video_subscriber",
            )

    async def _start_video_subscriber(self, track_id: str, user_id: str):
        """Start receiving video frames from a track.

        Args:
            track_id: The WebRTC track ID to subscribe to.
            user_id: The participant's user ID.
        """
        if not self._connection or not self._params.video_in_enabled:
            # Still fire the callback even if we don't consume frames
            self._video_subscribed_participants.add(user_id)
            await self._callbacks.on_video_track_subscribed(user_id)
            return

        try:
            video_track = self._connection.subscriber_pc.add_track_subscriber(track_id)
            self._video_subscribed_participants.add(user_id)
            await self._callbacks.on_video_track_subscribed(user_id)

            task_key = f"{user_id}:{track_id}"
            task = self._create_task(
                self._video_receive_loop(video_track, user_id),
                f"{self}::_video_receive_loop:{user_id}",
            )
            self._video_subscriber_tasks[task_key] = task
        except Exception:
            logger.exception(f"Error subscribing to video track {track_id}")

    async def _video_receive_loop(self, video_track, user_id: str):
        """Receive video frames from a subscribed track and queue them.

        Args:
            video_track: The aiortc video track to receive from.
            user_id: The participant's user ID.
        """
        try:
            while True:
                frame = await video_track.recv()
                rgb_array = frame.to_ndarray(format="rgb24")
                await self._video_queue.put((rgb_array, user_id))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug(f"Video receive loop ended for {user_id}: {exc}")

    def _on_track_unpublished(self, event):
        """Handle track unpublished event.

        Args:
            event: The TrackUnpublished protobuf event from the SFU.
        """
        user_id = event.user_id
        session_id = event.session_id
        track_type = event.type  # int TrackType enum

        if user_id == self._user_id:
            return

        track_key = (user_id, session_id, track_type)
        track_id = self._track_map.pop(track_key, None)

        # Also clean up any pending publication for this track
        self._pending_publications.pop(track_key, None)

        if track_id:
            # Cancel video subscriber if it was a video track
            task_key = f"{user_id}:{track_id}"
            task = self._video_subscriber_tasks.pop(task_key, None)
            if task:
                task.cancel()

        if track_type == TrackType.TRACK_TYPE_VIDEO:
            if user_id in self._video_subscribed_participants:
                self._video_subscribed_participants.discard(user_id)
                self._create_task(
                    self._callbacks.on_video_track_unsubscribed(user_id),
                    f"{self}::on_video_track_unsubscribed",
                )
        elif track_type in (TrackType.TRACK_TYPE_AUDIO, TrackType.TRACK_TYPE_SCREEN_SHARE_AUDIO):
            if user_id in self._audio_subscribed_participants:
                self._audio_subscribed_participants.discard(user_id)
                self._create_task(
                    self._callbacks.on_audio_track_unsubscribed(user_id),
                    f"{self}::on_audio_track_unsubscribed",
                )

    def _on_call_ended(self, *args):
        """Handle call ended event."""
        logger.info("Stream Video call ended")
        if self._connected:
            self._create_task(
                self.disconnect(),
                f"{self}::disconnect_on_call_ended",
            )

    def _create_task(self, coroutine: Coroutine | Awaitable, name: str) -> asyncio.Task:
        """Create an asyncio task via the task manager.

        Raises:
            RuntimeError: If the task manager has not been initialized via setup().
        """
        if self._task_manager is None:
            raise RuntimeError("Task manager not initialized. Was setup() called?")
        return self._task_manager.create_task(coroutine, name)

    def __str__(self):
        """String representation of the Stream Video transport client."""
        return f"{self._transport_name}::GetstreamTransportClient"


class GetstreamInputTransport(BaseInputTransport):
    """Handles incoming media streams and events from Stream Video calls.

    Processes incoming audio and video from call participants and forwards them
    as Pipecat frames, including audio format conversion and resampling.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: GetstreamTransportClient,
        params: GetstreamParams,
        **kwargs,
    ):
        """Initialize the Stream Video input transport.

        Args:
            transport: The parent transport instance.
            client: GetstreamTransportClient instance.
            params: Configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client

        self._audio_in_task = None
        self._video_in_task = None
        self._resampler = create_stream_resampler()

        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the input transport and connect to the Stream Video call.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.start(frame)
        await self._client.connect()
        if not self._audio_in_task and self._params.audio_in_enabled:
            self._audio_in_task = self.create_task(self._audio_in_task_handler())
        if not self._video_in_task and self._params.video_in_enabled:
            self._video_in_task = self.create_task(self._video_in_task_handler())
        await self.set_transport_ready(frame)
        logger.info("GetstreamInputTransport started")

    async def stop(self, frame: EndFrame):
        """Stop the input transport and disconnect.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.disconnect()
        if self._audio_in_task:
            await self.cancel_task(self._audio_in_task)
        if self._video_in_task:
            await self.cancel_task(self._video_in_task)
        logger.info("GetstreamInputTransport stopped")

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and disconnect.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.disconnect()
        if self._audio_in_task and self._params.audio_in_enabled:
            await self.cancel_task(self._audio_in_task)
        if self._video_in_task and self._params.video_in_enabled:
            await self.cancel_task(self._video_in_task)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the input transport with shared client setup.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        """Clean up input transport and shared resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def push_app_message(self, message: Any, sender: str):
        """Push an application message as an urgent transport frame.

        Args:
            message: The message data to send.
            sender: ID of the message sender.
        """
        frame = GetstreamOutputTransportMessageUrgentFrame(message=message, participant_id=sender)
        await self.push_frame(frame)

    async def _audio_in_task_handler(self):
        """Handle incoming audio frames from participants."""
        logger.info("Stream Video audio input task started")
        audio_iterator = self._client.get_next_audio_frame()
        async for audio_data in audio_iterator:
            if audio_data:
                pcm_data, participant_id = audio_data
                pipecat_audio_frame = await self._convert_stream_audio_to_pipecat(pcm_data)

                if len(pipecat_audio_frame.audio) == 0:
                    continue

                input_audio_frame = UserAudioRawFrame(
                    user_id=participant_id,
                    audio=pipecat_audio_frame.audio,
                    sample_rate=pipecat_audio_frame.sample_rate,
                    num_channels=pipecat_audio_frame.num_channels,
                )
                await self.push_audio_frame(input_audio_frame)

    async def _video_in_task_handler(self):
        """Handle incoming video frames from participants."""
        logger.info("Stream Video video input task started")
        video_iterator = self._client.get_next_video_frame()
        async for video_data in video_iterator:
            if video_data:
                rgb_array, participant_id = video_data
                height, width = rgb_array.shape[:2]
                image_bytes = rgb_array.tobytes()

                input_video_frame = UserImageRawFrame(
                    user_id=participant_id,
                    image=image_bytes,
                    size=(width, height),
                    format="RGB",
                )
                await self.push_video_frame(input_video_frame)

    async def _convert_stream_audio_to_pipecat(self, pcm_data: PcmData) -> AudioRawFrame:
        """Convert Stream Video PcmData to Pipecat AudioRawFrame.

        Handles int16/float32 conversion and resampling.

        Args:
            pcm_data: The PcmData from Stream Video SDK.

        Returns:
            Converted AudioRawFrame for the pipeline.
        """
        samples = pcm_data.samples

        # Convert float32 to int16 if needed
        if samples.dtype == np.float32:
            samples = (samples * 32767).astype(np.int16)
        elif samples.dtype != np.int16:
            samples = samples.astype(np.int16)

        raw_bytes = samples.tobytes()

        # Resample to transport input sample rate
        audio_data = await self._resampler.resample(
            raw_bytes, pcm_data.sample_rate, self.sample_rate
        )

        return AudioRawFrame(
            audio=audio_data,
            sample_rate=self.sample_rate,
            num_channels=pcm_data.channels if hasattr(pcm_data, "channels") else 1,
        )


class GetstreamOutputTransport(BaseOutputTransport):
    """Handles outgoing media streams and events to Stream Video calls.

    Manages sending audio frames, video frames, and data messages to
    Stream Video call participants.
    """

    def __init__(
        self,
        transport: BaseTransport,
        client: GetstreamTransportClient,
        params: GetstreamParams,
        **kwargs,
    ):
        """Initialize the Stream Video output transport.

        Args:
            transport: The parent transport instance.
            client: GetstreamTransportClient instance.
            params: Configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._transport = transport
        self._client = client

        self._initialized = False

        # Clock-based audio pacing to avoid drift from asyncio.sleep() inaccuracy.
        self._audio_clock: float = 0.0
        self._audio_clock_total: float = 0.0

    async def start(self, frame: StartFrame):
        """Start the output transport and connect to the Stream Video call.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.start(frame)
        await self._client.connect()
        await self.set_transport_ready(frame)
        logger.info("GetstreamOutputTransport started")

    async def stop(self, frame: EndFrame):
        """Stop the output transport and disconnect.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.disconnect()
        logger.info("GetstreamOutputTransport stopped")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames with Stream Video-specific interruption handling.

        On interruption, flushes the SDK's internal audio buffer so previously
        buffered TTS audio stops playing immediately.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        if isinstance(frame, InterruptionFrame) and self._allow_interruptions:
            await self._client.flush_audio()
            self._audio_clock = 0.0
            self._audio_clock_total = 0.0

        await super().process_frame(frame, direction)

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and disconnect.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.disconnect()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the output transport with shared client setup.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)
        await self._client.setup(setup)

    async def cleanup(self):
        """Clean up output transport and shared resources."""
        await super().cleanup()
        await self._transport.cleanup()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message to participants.

        Args:
            frame: The transport message frame to send.
        """
        message = frame.message
        if isinstance(message, dict):
            message = json.dumps(message, ensure_ascii=False)
        if isinstance(
            frame,
            (
                GetstreamOutputTransportMessageFrame,
                GetstreamOutputTransportMessageUrgentFrame,
            ),
        ):
            await self._client.send_data(message.encode(), frame.participant_id)
        else:
            await self._client.send_data(message.encode())

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the Stream Video call.

        Uses clock-based pacing to output audio at real-time rate. The Stream
        Video SDK's AudioStreamTrack.write() buffers internally without
        backpressure, so we track a monotonic clock to avoid both buffer
        overflow (writing too fast) and audio breakup from asyncio.sleep()
        drift accumulation.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        try:
            now = time.monotonic()

            bytes_per_sample = 2  # s16 format
            duration = len(frame.audio) / (
                self.sample_rate * bytes_per_sample * self._params.audio_out_channels
            )

            # Reset clock on first frame or after a gap (e.g. interruption,
            # silence between utterances). A negative delay means we fell
            # behind, which indicates a discontinuity.
            target = self._audio_clock + self._audio_clock_total + duration
            if self._audio_clock == 0.0 or (target - now) < -0.1:
                self._audio_clock = now
                self._audio_clock_total = 0.0
                target = now + duration

            self._audio_clock_total += duration

            # Sleep until the wall-clock catches up to where this chunk
            # should be delivered, absorbing prior sleep overshoot.
            delay = target - now
            if delay > 0:
                await asyncio.sleep(delay)

            pcm_data = PcmData.from_bytes(
                frame.audio,
                sample_rate=self.sample_rate,
                format="s16",
                channels=self._params.audio_out_channels,
            )
            return await self._client.write_audio(pcm_data)
        except Exception:
            logger.exception(f"Error writing audio frame")
            return False

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the Stream Video call.

        Args:
            frame: The output video frame to write.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        try:
            return self._client.write_video(frame.image, frame.size, frame.format)
        except Exception:
            logger.exception(f"Error writing video frame")
            return False


class GetstreamTransport(BaseTransport):
    """Transport implementation for Stream Video real-time communication.

    Provides comprehensive Stream Video integration including audio/video streaming,
    participant management, and call event handling for conversational AI applications.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        call_type: str,
        call_id: str,
        user_id: str,
        params: Optional[GetstreamParams] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the Stream Video transport.

        Args:
            api_key: Stream Video API key.
            api_secret: Stream Video API secret.
            call_type: The Stream call type (e.g. "default").
            call_id: Unique call identifier.
            user_id: The bot/agent user ID.
            params: Configuration parameters for the transport.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        super().__init__(input_name=input_name, output_name=output_name)

        callbacks = GetstreamCallbacks(
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_before_disconnect=self._on_before_disconnect,
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
            on_audio_track_subscribed=self._on_audio_track_subscribed,
            on_audio_track_unsubscribed=self._on_audio_track_unsubscribed,
            on_video_track_subscribed=self._on_video_track_subscribed,
            on_video_track_unsubscribed=self._on_video_track_unsubscribed,
            on_data_received=self._on_data_received,
            on_first_participant_joined=self._on_first_participant_joined,
        )
        self._params = params or GetstreamParams()

        self._client = GetstreamTransportClient(
            api_key,
            api_secret,
            call_type,
            call_id,
            user_id,
            self._params,
            callbacks,
            self.name,
        )
        self._input: Optional[GetstreamInputTransport] = None
        self._output: Optional[GetstreamOutputTransport] = None

        self._register_event_handler("on_connected")
        self._register_event_handler("on_disconnected")
        self._register_event_handler("on_participant_connected")
        self._register_event_handler("on_participant_disconnected")
        self._register_event_handler("on_audio_track_subscribed")
        self._register_event_handler("on_audio_track_unsubscribed")
        self._register_event_handler("on_video_track_subscribed")
        self._register_event_handler("on_video_track_unsubscribed")
        self._register_event_handler("on_data_received")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_before_disconnect", sync=True)

    def input(self) -> GetstreamInputTransport:
        """Get the input transport for receiving media and events.

        Returns:
            The Stream Video input transport instance.
        """
        if not self._input:
            self._input = GetstreamInputTransport(
                self, self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> GetstreamOutputTransport:
        """Get the output transport for sending media and events.

        Returns:
            The Stream Video output transport instance.
        """
        if not self._output:
            self._output = GetstreamOutputTransport(
                self, self._client, self._params, name=self._output_name
            )
        return self._output

    @property
    def participant_id(self) -> str:
        """Get the participant ID for this transport.

        Returns:
            The user ID assigned to this transport.
        """
        return self._client.participant_id

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send an audio frame to the Stream Video call.

        Args:
            frame: The audio frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    def get_participants(self) -> List[str]:
        """Get list of participant IDs in the call.

        Returns:
            List of participant user IDs.
        """
        return self._client.get_participants()

    async def send_custom_event(self, data: dict):
        """Send a custom event to call participants.

        Args:
            data: Dictionary of custom event data to send.
        """
        await self._client.send_data(json.dumps(data).encode())

    async def _on_connected(self):
        """Handle call connected events."""
        await self._call_event_handler("on_connected")

    async def _on_disconnected(self):
        """Handle call disconnected events."""
        await self._call_event_handler("on_disconnected")

    async def _on_before_disconnect(self):
        """Handle before disconnection events."""
        await self._call_event_handler("on_before_disconnect")

    async def _on_participant_joined(self, participant_id: str):
        """Handle participant joined events."""
        await self._call_event_handler("on_participant_connected", participant_id)

    async def _on_participant_left(self, participant_id: str):
        """Handle participant left events."""
        await self._call_event_handler("on_participant_disconnected", participant_id)
        await self._call_event_handler("on_participant_left", participant_id, "left")

    async def _on_audio_track_subscribed(self, participant_id: str):
        """Handle audio track subscribed events."""
        await self._call_event_handler("on_audio_track_subscribed", participant_id)

    async def _on_audio_track_unsubscribed(self, participant_id: str):
        """Handle audio track unsubscribed events."""
        await self._call_event_handler("on_audio_track_unsubscribed", participant_id)

    async def _on_video_track_subscribed(self, participant_id: str):
        """Handle video track subscribed events."""
        await self._call_event_handler("on_video_track_subscribed", participant_id)

    async def _on_video_track_unsubscribed(self, participant_id: str):
        """Handle video track unsubscribed events."""
        await self._call_event_handler("on_video_track_unsubscribed", participant_id)

    async def _on_data_received(self, data: bytes, participant_id: str):
        """Handle data received events."""
        if self._input:
            await self._input.push_app_message(data.decode(), participant_id)
        await self._call_event_handler("on_data_received", data, participant_id)

    async def send_message(self, message: str, participant_id: Optional[str] = None):
        """Send a message to participants in the call.

        Args:
            message: The message string to send.
            participant_id: Optional specific participant to send to.
        """
        if self._output:
            frame = GetstreamOutputTransportMessageFrame(
                message=message, participant_id=participant_id
            )
            await self._output.send_message(frame)

    async def send_message_urgent(self, message: str, participant_id: Optional[str] = None):
        """Send an urgent message to participants in the call.

        Args:
            message: The urgent message string to send.
            participant_id: Optional specific participant to send to.
        """
        if self._output:
            frame = GetstreamOutputTransportMessageUrgentFrame(
                message=message, participant_id=participant_id
            )
            await self._output.send_message(frame)

    async def _on_first_participant_joined(self, participant_id: str):
        """Handle first participant joined events."""
        await self._call_event_handler("on_first_participant_joined", participant_id)
