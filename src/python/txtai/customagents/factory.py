from ..agent.base import Agent
from .chatagent.chat import ConversationalAgent
from ..mlflow.tracker import Tracker
from ..logging.logger import get_logger
from .speechtotext.stt import SpeechToText
from .texttospeech.tts import TextToSpeechAgent
class AgentFactory:

    @staticmethod
    def create_agent(agent_type: str, config):
        llm_config = config.get("llm", {})
        agent_config = config.get("agent", {})

        base_agent = Agent(**{
            **agent_config,
            "model": llm_config,
            "tools": []
        })

        tracker = Tracker()
        tracker.log_static(llm_config, agent_config)
        logger = get_logger(agent_type)

        if agent_type == "conversational":
            return ConversationalAgent(base_agent, config, tracker, logger)
        
        elif agent_type == "speech_to_text":
            return SpeechToText(config, tracker, logger)
        
        elif agent_type == "text_to_speech":
            return TextToSpeechAgent(config, tracker, logger)
        
        raise ValueError(f"Unknown agent type: {agent_type}")
