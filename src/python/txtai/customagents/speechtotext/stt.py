import os
from ..base import BaseAgent
from ...logging.logger import get_logger
from ...pipeline.audio.cloudtranscription import CloudTranscription
import asyncio
import numpy as np
import time
import mlflow
from mlflow.entities import SpanType
from dotenv import load_dotenv

#mlflow.set_experiment("speech_text") 
load_dotenv()
print("HUG_API_KEY:", os.getenv("HUG_API_KEY"))
class SpeechToText(BaseAgent): 

    def __init__(self, config, tracker, logger=None):
        super().__init__(config, tracker, logger)# or get_logger("SpeechToTextAgent")

    
        stt_config = config.get("stt", {})
        self.model = stt_config.get("model")
        self.endpoint = stt_config.get("endpoint_url")
        self.api_key = os.getenv("HUG_API_KEY")
        self.provider = stt_config.get("provider", "huggingface")
        self.chunk = stt_config.get("chunk", 10)
        self.target_rate = stt_config.get("target_rate", 16000)

        self.transcriber = CloudTranscription(
            endpoint_url=self.endpoint,
            api_key=self.api_key,
            model=self.model,
            provider=self.provider,
            chunk=self.chunk,
            target_rate=self.target_rate
        )
        
        self.initial_message = "Hello! I am your medical assistant. How can I help you today?"
        self.logger.info(f"Initialized STT model '{self.model}' on {self.provider}")
    
    def initial_message(self):
        return self.initial_message
    
    @mlflow.trace(name="stt_generate_response", span_type=SpanType.AGENT)
    async def generate_response(self, audio_input: np.ndarray):
        """
        Transcribes an audio file into text using the configured cloud provider.
        Args:
            audio_path (str): Path to an audio file (wav/mp3/flac)
        Returns:
            str: Transcribed text
        """
        try:
            self.logger.info(f"Transcribing audio via {self.provider}: {audio_input}")
            text = await asyncio.to_thread(
                self.transcriber.transcribe, audio_input, self.target_rate
            )
            self.tracker.log_turn(f"Audio({audio_input})", text)
            self.logger.info(f" Transcription complete: {text}")
            return text

        except Exception as e:
            self.logger.error(f"Error during cloud transcription: {e}")
            return f"Error during transcription: {e}"