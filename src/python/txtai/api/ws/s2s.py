from fastapi import WebSocket, WebSocketDisconnect
from ...customagents.configloader import ConfigLoader
from ...customagents.factory import AgentFactory
from ...customagents.agentservice import AgentService
from ...logging.logger import get_logger
from dotenv import load_dotenv
import os

from .helper import WebSocketResponder
from .validator import MessageValidator
import mlflow
import numpy as np
import asyncio
import time

import importlib.resources as resources
from ...customagents import chatagent  


def register_s2s_ws(app):
    """
    Registers the Speech-to-Speech (STT) WebSocket endpoint inside lifespan.
    """

    load_dotenv(dotenv_path=r"txtai\src\python\txtai\.env")
    logger = get_logger("SpeechToSpeechWebSocket")

    # config_path = r"txtai\src\python\txtai\agentconfig\medical.yml"
    # config = ConfigLoader.load(config_path)

    with resources.open_text(chatagent, "medical.yml") as cfg:
        config = ConfigLoader.load(cfg.name)

    mlflow.set_experiment("speech_to_speech_agent")

    stt_agent = AgentFactory.create_agent("speech_to_text", config)
    chat_agent = AgentFactory.create_agent("conversational", config)
    tts_agent = AgentFactory.create_agent("text_to_speech", config)

    stt_service = AgentService(stt_agent)
    chat_service = AgentService(chat_agent)
    tts_service = AgentService(tts_agent)

    helper =WebSocketResponder(tts_service)
    validator = MessageValidator()

    @app.websocket("/ws/s2s")
    async def websocket_s2s(websocket: WebSocket):

        await websocket.accept()
        logger.info("Client connected to /ws/s2s")

        
        greeting = chat_agent.get_initial_message()
        #await helper.respond_with_voice(greeting, websocket, tts_service)
        await helper.respond_with_voice(greeting, websocket)
        if hasattr(chat_agent, "reset"):
            chat_agent.reset()
        # chat_service.turn_count = 0
        audio_buffer = []

        try:
            while True:
                msg = await websocket.receive()

               
                if "bytes" in msg:
                    audio_np = np.frombuffer(
                        msg["bytes"], dtype=np.int16
                    ).astype(np.float32) / 32768.0

                    audio_buffer.append(audio_np)
                    continue

               
                if "text" in msg :
                    text=msg["text"].strip().lower() 
                    
                    if text == "__end__":
                    
                        if not audio_buffer:
                            await websocket.send_text("No audio received.")
                            continue

                    audio_data = np.concatenate(audio_buffer)
                    audio_buffer.clear()

                    user_text = await stt_service.handle_message(audio_data)
                    await websocket.send_text(f"User: {user_text}")

                    reply_text = await chat_service.handle_message(user_text)

                    await helper.respond_with_voice(reply_text, websocket)

                    logger.info("Speech-to-Speech turn completed.")
                    continue
                
                if not validator.is_valid(text):
                        logger.info(f"Ignored browser message: {text}")
                        continue
                await websocket.send_text("Send audio chunks or '__end__'.")

        except WebSocketDisconnect:
            logger.info("Client disconnected from S2S.")
            for s in [stt_service, chat_service, tts_service]:
                s.end_session("disconnect")

        except Exception as e:
            logger.error(f"S2S Error: {e}", exc_info=True)
            await websocket.send_text("Error occurred in S2S pipeline.")
            for s in [stt_service, chat_service, tts_service]:
                s.end_session("error")

        finally:
            await helper.safe_close(websocket)
            logger.info("S2S WebSocket closed.")


    # @app.websocket("/ws/s2s")
    # async def websocket_s2s(websocket: WebSocket):

    #     await websocket.accept()
    #     logger.info(f"Client connected to /ws/s2s at {time.strftime('%H:%M:%S')}")

    #     initial_msg = "Hello! how can i assist you today?"
    #     await websocket.send_text(initial_msg)
    #     await tts_service.handle_message(initial_msg, ws=websocket)
        
    #     audio_buffer = []

    #     try:
    #         while True:
    #             msg = await websocket.receive()

    #             if "text" in msg:
    #                 text = msg["text"].strip()

    #                 if text.lower() == "__end__":
    #                     if not audio_buffer:
    #                         await websocket.send_text("No audio received.")
    #                         continue

    #                     combined_audio = np.concatenate(audio_buffer)
    #                     audio_buffer.clear()

    #                     text_input = await stt_service.handle_message(combined_audio)
    #                     await websocket.send_text(f"user: {text_input}")

    #                     reply_text = await chat_service.handle_message(text_input)
    #                     await websocket.send_text(f"Agent: {reply_text}")

    #                     await tts_service.handle_message(reply_text, ws=websocket)
    #                     logger.info("Speech-to-Speech completed.")
    #                     continue

    #                 else:
    #                     await websocket.send_text("send audio chunks")
    #                 continue

    #             elif "bytes" in msg:
    #                 audio_bytes = msg["bytes"]
    #                 # if not audio_bytes:
    #                 #     continue
    #                 if audio_bytes:
    #                     audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    #                     audio_buffer.append(audio_np)

    #                     await websocket.send_text(f"Received {len(audio_np)} samples")
        
    #     except WebSocketDisconnect:
    #         logger.warning("Client disconnected.")
    #         for s in [stt_service, chat_service, tts_service]:
    #             s.end_session("disconnected")

    #     except Exception as e:
    #         logger.error(f"S2S Orchestrator Error: {e}", exc_info=True)
    #         await websocket.send_text("Speech-to-Speech error occurred.")
    #         for s in [stt_service, chat_service, tts_service]:
    #             s.end_session("error")

    #     finally:
    #         await websocket.close()
    #         logger.info("/ws/s2s WebSocket closed.")



