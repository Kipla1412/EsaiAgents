from fastapi import WebSocket, WebSocketDisconnect
from ...customagents.configloader import ConfigLoader
from ...customagents.factory import AgentFactory
from ...customagents.agentservice import AgentService
from ...txtailogging.logger import get_logger

from dotenv import load_dotenv
import os
import mlflow
import numpy as np
import asyncio
import time
from .helper import WebSocketResponder
from .validator import MessageValidator

import importlib.resources as resources
from ...customagents import chatagent  


validator = MessageValidator()

def register_tts_ws(app):
    """
    Registers the Text-to-Speech (TTS) WebSocket endpoint inside lifespan.
    """

    #load_dotenv(dotenv_path=r"txtai\src\python\txtai\.env")
    logger =get_logger("TTSWebSocketServer")

    with resources.open_text(chatagent, "medical.yml") as cfg:
        config = ConfigLoader.load(cfg.name)

    mlflow.set_experiment("tts_agent")
    tts_agent = AgentFactory.create_agent("text_to_speech", config)
    tts_service = AgentService(tts_agent)
    helper =WebSocketResponder(tts_service)

    @app.websocket("/ws/tts")
    async def websocket_tts(websocket: WebSocket):
        """
        Text-to-Speech WebSocket endpoint.
        Receives text → streams back audio bytes (or text).
        """
        await websocket.accept()
        logger.info(f"Client connected to /ws/tts at {time.strftime('%H:%M:%S')}")
        #initial_msg = getattr(tts_agent, "initial_message", "Hello! how can i assist you? ")
        #initial_msg = tts_agent.get_initial_message()
        # await websocket.send_text(initial_msg)
        # await tts_service.handle_message(initial_msg, ws=websocket)
        greeting = tts_agent.get_initial_message()
        await helper.respond_with_voice(greeting, websocket)
        try:
            while True:

                user_input = await websocket.receive_text()
                if not user_input.strip():
                    continue
                
                # if user_input.lower() in ["undefined", "null", "__start__", "start"]:
                #     logger.info(f"Ignored auto-message: {user_input}")
                #     continue
                logger.info(f"Received text for TTS: {user_input}")
                #await tts_service.handle_message(user_input, ws=websocket)
                await helper.respond_with_voice(user_input, websocket)

        except WebSocketDisconnect:
            logger.warning("Client disconnected.")
            tts_service.end_session(status="disconnected")

        except Exception as e:
            logger.error(f"TTS WebSocket error: {e}", exc_info=True)
            await websocket.send_text("TTS processing failed.")
            tts_service.end_session(status="error")

        finally:
            await helper.safe_close(websocket)
            logger.info("TTS WebSocket connection closed.")
