import asyncio
from starlette.websockets import WebSocketState

async def stream_audio(ws, pcm_bytes: bytes, chunk_size: int = 32000, delay: float = 0.01, end_signal: str = "__finish_speech__"):
    """
    Streams PCM16 audio bytes to a connected WebSocket client in chunks.
    Can be reused for TTS, voice cloning, or other audio generation pipelines.

    Args:
        ws (WebSocket): Active WebSocket connection.
        pcm_bytes (bytes): Raw PCM16 audio data to stream.
        chunk_size (int): Number of bytes per chunk to send.
        delay (float): Delay between chunks for smoother streaming.
        end_signal (str): Marker sent when streaming is complete.
    """
    try:
        total_bytes = len(pcm_bytes)
        print(f"[TTS] Starting stream: {total_bytes} bytes of audio")

        for i in range(0, total_bytes, chunk_size):
            if ws.client_state != WebSocketState.CONNECTED:
                break
            await ws.send_bytes(pcm_bytes[i:i + chunk_size])
            await asyncio.sleep(delay)

        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(end_signal)

        print("[TTS] Done streaming.")

    except Exception as e:
        print(f"[TTS STREAM ERROR] {e}")
