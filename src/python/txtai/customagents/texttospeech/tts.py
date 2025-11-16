from ..base import BaseAgent
from ...pipeline.audio.cloudtts import CloudTextToSpeech
from ...logging.logger import get_logger
import asyncio
import os
from mlflow.entities import SpanType
from dotenv import load_dotenv
import mlflow
import numpy as np

#mlflow.set_experiment("text_speech")
load_dotenv()
#load_dotenv(dotenv_path=r"txtai\src\python\txtai\.env")
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

class TextToSpeechAgent(BaseAgent):

    def __init__(self, config, tracker, logger=None):
        super().__init__(config, tracker, logger or get_logger("TextToSpeechAgent"))

        tts_config= config.get("tts", {})
        
        self.engine = CloudTextToSpeech(
            endpoint_url=tts_config.get("endpoint_url"),
            api_key=os.getenv("GROQ_API_KEY"),
            model=tts_config.get("model"),
            provider=tts_config.get("provider", "groq"),
            target_rate=tts_config.get("target_rate", 22050),
            voice=tts_config.get("voice", "Aaliyah-PlayAI"),
        )
        self.initial_message ="Hello! I am your medical assistant. How can I help you today?"
        self.logger.info(f"TTS Agent initialized ({self.engine.provider})")
    
    def get_initial_message(self):
        return self.initial_message
    
    @mlflow.trace(name="tts_generate_response", span_type=SpanType.AGENT)
    async def generate_response(self, text:str):
        try:
            if not isinstance(text, str) or not text.strip():#if not text.strip():
                raise ValueError("Empty text received for TTS conversion.")
            audio, sr = await asyncio.to_thread(self.engine, text)
            
            if not isinstance(audio, np.ndarray):
                raise TypeError(f"TTS engine returned invalid audio type: {type(audio)}")
        
            pcm_bytes = self.engine.float_to_pcm16(audio)
            wav_bytes = self.engine.array_to_wav_bytes(audio, sr)
            if self.tracker:
                self.tracker.log_turn(text, f"{len(wav_bytes)} bytes")#{len(pcm_bytes)} bytes (PCM)
            self.logger.info(f"Generated speech: {len(wav_bytes)} bytes @ {sr}Hz")

            # if self.tracker:
            #     self.tracker.log_turn(text, f"{len(wav_bytes)} bytes")#{len(pcm_bytes)} bytes (PCM)
            # self.logger.info(f"Generated speech: {len(wav_bytes)} bytes @ {sr}Hz")
            
            return pcm_bytes, sr
            #return wav_bytes,sr # "pcm_bytes": pcm_bytes,
        except Exception as e:
            self.logger.error(f"TTS Error: {e}")
            return None, None