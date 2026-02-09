"""Stream Video REST Helpers.

Methods that wrap the Stream Video API for call and user management.
"""

from loguru import logger

try:
    from getstream import AsyncStream
    from getstream.models import UserRequest
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Stream Video, you need to `pip install pipecat-ai[stream-video]`."
    )
    raise Exception(f"Missing module: {e}")


class StreamVideoRESTHelper:
    """Helper class for interacting with Stream Video's REST API.

    Provides methods for managing Stream Video calls, users, and tokens.
    """

    def __init__(self, *, api_key: str, api_secret: str):
        """Initialize the Stream Video REST helper.

        Args:
            api_key: Your Stream Video API key.
            api_secret: Your Stream Video API secret.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._client = AsyncStream(api_key=api_key, api_secret=api_secret)

    async def create_user(self, user_id: str, name: str | None = None) -> dict:
        """Create or update a user in Stream.

        Args:
            user_id: Unique identifier for the user.
            name: Optional display name for the user.

        Returns:
            Dictionary containing user data from Stream API.
        """
        user_request = UserRequest(id=user_id, name=name or user_id)
        response = await self._client.upsert_users(user_request)
        return response

    async def create_call(self, call_type: str, call_id: str, created_by_id: str) -> dict:
        """Create or retrieve a Stream Video call.

        Args:
            call_type: The type of call (e.g. "default").
            call_id: Unique identifier for the call.
            created_by_id: User ID of the call creator.

        Returns:
            Dictionary containing call data from Stream API.
        """
        call = self._client.video.call(call_type, call_id)
        response = await call.get_or_create(data={"created_by_id": created_by_id})
        return response

    def create_token(self, user_id: str, expiration: int | None = None) -> str:
        """Generate a JWT token for a user.

        Args:
            user_id: User ID to generate the token for.
            expiration: Optional token expiration time in seconds.

        Returns:
            Signed JWT token string.
        """
        kwargs = {"user_id": user_id}
        if expiration is not None:
            kwargs["expiration"] = expiration
        return self._client.create_token(**kwargs)

    async def delete_call(self, call_type: str, call_id: str) -> bool:
        """Delete a Stream Video call.

        Args:
            call_type: The type of call.
            call_id: The call identifier to delete.

        Returns:
            True if the call was deleted successfully.

        Raises:
            Exception: If the deletion fails.
        """
        call = self._client.video.call(call_type, call_id)
        await call.delete()
        return True
