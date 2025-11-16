from fastapi import WebSocket, WebSocketDisconnect
from ...customagents.configloader import ConfigLoader
from ...customagents.factory import AgentFactory
from ...customagents.agentservice import AgentService
from ...logging.logger import get_logger
import importlib.resources as resources
from ...customagents import chatagent  
from ...customagents.resourceloader import ConfigResourceLoader
import mlflow
import os


def register_chat_ws(app):
    """
    Registers the Chat WebSocket endpoint inside lifespan.
    """

    logger = get_logger("ChatWebSocket")

    # Load config (only once)
    # config_path = r"txtai\src\python\txtai\agentconfig\medical.yml"
    # config = ConfigLoader.load(config_path)
    with resources.open_text(chatagent, "medical.yml") as cfg:
        config = ConfigLoader.load(cfg.name)

    mlflow.set_experiment("chat_agent")

    # Create agent + service ONCE (not inside websocket function)
    chat_agent = AgentFactory.create_agent("conversational", config)
    chat_service = AgentService(chat_agent)

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):

        await websocket.accept()
        logger.info("Client connected to /ws/chat")

        # Send initial message
        await websocket.send_text(chat_agent.get_initial_message())

        try:
            while True:
                # Receive text
                user_message = await websocket.receive_text()
                logger.info(f"User: {user_message}")

                # Process using ChatAgent
                response = await chat_service.handle_message(user_message)

                # Send response
                await websocket.send_text(response)
                logger.info(f"Agent: {response}")

        except WebSocketDisconnect:
            logger.warning("Client disconnected.")
            chat_service.end_session("disconnected")

        except Exception as e:
            logger.error(f"Chat WebSocket error: {e}", exc_info=True)
            await websocket.send_text("Sorry, something went wrong.")
            chat_service.end_session("error")

        finally:
            await websocket.close()
            logger.info("Chat WebSocket closed.")
