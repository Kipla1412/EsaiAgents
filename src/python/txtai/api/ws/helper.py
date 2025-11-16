
from starlette.websockets import WebSocketState

class WebSocketResponder:
    """
    Helper class for safe WebSocket responses and TTS streaming.
    """

    def __init__(self, tts_service=None):
        self.tts_service = tts_service

    async def respond_with_voice(self, text: str, websocket):
        """
        Sends text + TTS audio to client.
        """
        if websocket.client_state != WebSocketState.CONNECTED:
            return

        # Send plain text message
        await websocket.send_text(text)

        # If TTS service exists → stream audio
        if self.tts_service:
            await self.tts_service.handle_message(text, ws=websocket)

    async def safe_close(self, websocket):
        """
        Safely closes the WebSocket without raising RuntimeError.
        """
        try:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except Exception:
            pass

# async def respond_with_voice(text, websocket, tts_service):
#     """
#     Sends text to client AND converts it into speech using TTS.
#     """
#     # Send text
#     await websocket.send_text(text)

#     # Convert text → speech (TTS)
#     await tts_service.handle_message(text, ws=websocket)

# async def safe_close(websocket):
#     """
#     Close WS only if safe. Prevents RuntimeError: Cannot call "send" once a close message has been sent.
#     """
#     try:
#         if websocket.application_state != WebSocketState.DISCONNECTED:
#             await websocket.close()
#     except Exception:
#         pass
