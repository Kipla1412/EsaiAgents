import requests
import base64
import json
import io
import soundfile as sf
import numpy as np
from .signal import Signal  # Your resample helper

class CloudTextToSpeech:
    """
    Cloud-based Text to Speech Wrapper.
    Supports Hugging Face, OpenAI/OpenRouter, and Groq.
    """

    def __init__(self, endpoint_url, api_key=None, model=None, provider="huggingface", target_rate=22050, voice=None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.target_rate = target_rate
        self.voice = voice or "Aaliyah-PlayAI"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def __call__(self, text, as_wav=True):
        return self.synthesize(text, as_wav=as_wav)

    def synthesize(self, text, as_wav=True):
        text = text.strip()
        if not text:
            raise ValueError("Input text must be non-empty")

        audio_bytes = None

        if self.provider == "huggingface":
            url = f"{self.endpoint_url}/{self.model}"
            payload = {"inputs": text}
            response = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(payload))
            if response.status_code != 200:
                raise RuntimeError(f"HuggingFace TTS failed: {response.status_code} {response.text}")
            audio_bytes = response.content

        elif self.provider == "groq":
            url = f"{self.endpoint_url}/audio/speech"
            payload = {"model": self.model, "input": text, "voice": self.voice, "response_format": "wav"}
            response = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(payload))
            if response.status_code != 200:
                raise RuntimeError(f"Groq TTS failed: {response.status_code} {response.text}")
            audio_bytes = response.content

        elif self.provider in ["openai", "openrouter"]:
            url = self.endpoint_url
            payload = {"model": self.model, "input": text}
            response = requests.post(url, headers={**self.headers, "Content-Type": "application/json"}, data=json.dumps(payload))
            if response.status_code != 200:
                raise RuntimeError(f"{self.provider} TTS failed: {response.status_code} {response.text}")
            try:
                result = response.json()
                audio_bytes = base64.b64decode(result.get("audio", result.get("data", "")))
            except:
                audio_bytes = response.content

        if not audio_bytes:
            raise RuntimeError("TTS returned empty audio.")

        
        return self.bytes_to_array(audio_bytes) 

    def bytes_to_array(self, audio_bytes, make_mono =True):

        buffer = io.BytesIO(audio_bytes)
        audio, sr = sf.read(buffer, dtype='float32')
        
        if audio.size == 0:
            raise RuntimeError("Audio read successfully but it's empty")
        
        if sr != self.target_rate:
            audio = Signal.resample(audio, sr, self.target_rate)
            sr = self.target_rate
        ## additionally i add this make mono , audio size method
        if make_mono and audio.ndim > 1:
            audio = Signal.mono(audio)
        return audio, sr
    
    def array_to_wav_bytes(self, audio: np.ndarray, sr: int):
        """
        Converts a NumPy float array to valid WAV bytes.
        Used when sending audio to browsers or WebSocket clients.
        """
        import io
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def float_to_pcm16(self, audio: np.ndarray):
        """Convert float32 audio (-1 to 1) to PCM16 bytes for WebSocket."""
        audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        return audio_int16.tobytes()


