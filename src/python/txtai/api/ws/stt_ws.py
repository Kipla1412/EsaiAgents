from fastapi import WebSocket, WebSocketDisconnect
from ...customagents.configloader import ConfigLoader
from ...customagents.factory import AgentFactory
from ...customagents.agentservice import AgentService
from ...txtailogging.logger import get_logger
from dotenv import load_dotenv
from .helper import WebSocketResponder
import os
import mlflow
import numpy as np
import asyncio
import time

import importlib.resources as resources
from ...customagents import chatagent  


def register_stt_ws(app):

    """
    Registers the Speech-to-Text (STT) WebSocket endpoint inside lifespan.
    """

    #load_dotenv(dotenv_path=r"txtai\src\python\txtai\.env")
    logger = get_logger("STTWebSocketServer")

    # config_path = r"txtai\src\python\txtai\agentconfig\medical.yml"
    # config = ConfigLoader.load(config_path)
    with resources.open_text(chatagent, "medical.yml") as cfg:
        config = ConfigLoader.load(cfg.name)

    mlflow.set_experiment("speech_to_text_agent")

    stt_agent = AgentFactory.create_agent("speech_to_text", config)
    stt_service = AgentService(stt_agent)
    helper =WebSocketResponder()

    @app.websocket("/ws/stt")
    async def websocket_stt(websocket:WebSocket):
        """
        Speech-to-Text WebSocket endpoint.
        Streams audio from client and returns transcribed text in real time.
        """
        await websocket.accept()
        logger.info(f"Client connected to /ws/stt at {time.strftime('%H:%M:%S')}")
        initial_msg =stt_agent.get_initial_message()
        await websocket.send_text(initial_msg)

        audio_buffer = []

        try:
            while True:

                msg =await websocket.receive()

                if "text" in msg:

                    user_text = msg["text"].strip()
                    if user_text.lower() == "__end__":
                        if audio_buffer:
                            combined_audio = np.concatenate(audio_buffer)
                            audio_buffer = []
                            logger.info(f"Received final audio of {len(combined_audio)} samples — transcribing...")

                            await websocket.send_text("Transcribing... please wait")
                            text = await stt_service.handle_message(combined_audio)
                            logger.info(f"Transcribed: {text}")
                            await websocket.send_text(f"{text}")
                        else:
                            await websocket.send_text("No audio data received.")
                    else:
                        await websocket.send_text("Send raw audio bytes or '__end__' to process.")
                    continue

                elif "bytes" in msg:
                    audio_bytes = msg["bytes"]
                    if not audio_bytes:
                        continue

                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_buffer.append(audio_np)
                    logger.debug(f"Received {len(audio_np)} audio samples (total buffer={len(audio_buffer)})")

                    await websocket.send_text(f"Received {len(audio_np)} samples...")

        except WebSocketDisconnect:
            logger.warning("Client disconnected.")
            stt_service.end_session(status="disconnected")

        except Exception as e:
            logger.error(f"STT WebSocket error: {e}", exc_info=True)
            await websocket.send_text("Error during transcription.")
            stt_service.end_session(status="error")

        finally:
            await helper.safe_close(websocket)
            logger.info("STT WebSocket connection closed.")





