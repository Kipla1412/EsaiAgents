from fastapi import WebSocket, WebSocketDisconnect
from ...customagents.configloader import ConfigLoader
from ...customagents.factory import AgentFactory
from ...customagents.agentservice import AgentService
from ...txtailogging.logger import get_logger
import importlib.resources as resources
from ...customagents import phiaagent
import mlflow
import json
import os

def register_phia_ws(app):
    """
    Registers the PHIA WebSocket endpoint inside lifespan.
    """

    logger = get_logger("PHIAWebSocket")

    with resources.open_text(phiaagent, "phia_config.yml") as cfg:
        config = ConfigLoader.load(cfg.name)

    mlflow.set_experiment("phia_agent")

    phia_agent = AgentFactory.create_agent("phia", config)
    phia_service = AgentService(phia_agent)

    @app.websocket("/ws/phia")
    async def websocket_phia(websocket: WebSocket):

        await websocket.accept()
        logger.info("Client connected to /ws/phia")

        await websocket.send_text("PHIA Agent ready! Send: {\"question\": \"...\"}")

        try:
            while True:
                raw_msg = await websocket.receive_text()

                # Expect JSON input
                try:
                    payload = json.loads(raw_msg)
                    question = payload.get("question", "").strip()

                except Exception:
                    await websocket.send_text("Invalid JSON. Use: {\"question\":\"...\"}")
                    continue

                if not question:
                    await websocket.send_text("Missing `question` in request.")
                    continue

                logger.info(f"PHIA Question: {question}")
                result = await phia_service.handle_message(question)

                await websocket.send_text(result)

        except WebSocketDisconnect:
            logger.warning("PHIA client disconnected.")
            phia_service.end_session("disconnected")

        except Exception as e:
            logger.error(f"PHIA WebSocket error: {e}", exc_info=True)
            await websocket.send_text("PHIA error.")
            phia_service.end_session("error")

        finally:
            await websocket.close()
            logger.info("PHIA WebSocket connection closed.")