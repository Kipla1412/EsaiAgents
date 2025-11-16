import time 
import asyncio
from ..txtailogging.logger import get_logger
from ..eval.tracker import Tracker
import mlflow
from .utils import stream_audio

class AgentService:

    def __init__(self, agent, name: str =None):

        self.agent = agent
        self.tracker = getattr(agent, "tracker", None)  # logs to MLflow
        #self.agent_name = name or agent.__class__.__name__
        self.logger = get_logger(name or agent.__class__.__name__)
        self.session_start = time.time()
        self.turn_count = 0

        if self.tracker:
            self.tracker.log_static(
                agent.config.get("llm", agent.config.get("tts", agent.config.get("stt", {}))),
                agent.config.get("agent", {"description": "Generic session"})
            )
            #mlflow.set_experiment(self.agent_name)
            self.logger.info(f"Initialized {name or agent.__class__.__name__}")

    async def handle_message(self, user_input, ws=None):

        self.turn_count += 1 
        self.logger.info(f"[Turn {self.turn_count}] User: {user_input}")

        start_time = time.time()
        try:
            
            if isinstance(user_input, str) and not user_input.strip():
                self.logger.info("Ignored empty user message (no TTS).")
                return None
            
            #response =await self.agent.generate_response(user_input)
            if isinstance(user_input, dict) and "text" in user_input:
                response = await self.agent.generate_response(**user_input)
            else:
                response = await self.agent.generate_response(user_input)
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            response = "Error during response generation."

        duration = round(time.time() - start_time, 3)

        #self.tracker.log_turn(user_input, response, duration)
        #self.logger.info(f"[Turn {self.turn_count}] Agent: {response}")
        self.logger.debug(f"Turn duration: {duration}s")

        if isinstance(response, tuple) and isinstance(response[0], (bytes, bytearray)) and ws:
            pcm_bytes, sr = response
            self.logger.info(f"Streaming TTS output ({len(pcm_bytes)} bytes @ {sr}Hz)")
            await stream_audio(ws, pcm_bytes)
        
        if self.tracker:
            summary = self._summarize_output(response)
            self.tracker.log_turn(str(user_input), summary, duration)

        self.logger.info(f"[Turn {self.turn_count}] Response ready.")
        return response
    
    def end_session(self, status="completed"):

        """Close MLflow run and log summary metrics."""
        session_duration = round(time.time() - self.session_start, 2)
        self.logger.info(f"Session ended ({status}) - Duration: {session_duration}s")

       
        mlflow.log_metric("session_duration_sec", session_duration)
        mlflow.log_metric("total_turns", self.turn_count)
        mlflow.set_tag("session_status", status)

        if self.tracker:
            self.tracker.end_run()
        else:
            mlflow.end_run()
    
    def _summarize_output(self, response):
        
        """Detect response type and create summary for MLflow."""
        if not response:
            return "No output"
            
        if isinstance(response, (bytes, bytearray)):
            return f"Audio({len(response)} bytes)"
        elif isinstance(response, tuple):
            if response[0] is None:
                return "Audio(None)"
            return f"Audio({len(response[0])} bytes, {response[1]} Hz)"
        return str(response)[:100]

