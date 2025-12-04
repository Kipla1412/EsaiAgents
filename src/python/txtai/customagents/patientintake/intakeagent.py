import mlflow
from mlflow.entities import SpanType
from ..base import BaseAgent
from .extractjson import extract_json
from .summarybuilder import build_summary
import importlib.resources as resources
from ..resourceloader import ConfigResourceLoader

class MedicalIntakeAgent(BaseAgent):

    def __init__(self, agent_model, config, tracker, logger):
        super().__init__(config, tracker, logger)

        self.agent_model = agent_model
        self.tracker = tracker
        self.logger = logger
        
        # Load system prompt
        with resources.open_text("txtai.customagents.patientintake", "systemprompt.txt") as f:
            self.system_prompt = f.read()

        # Keep conversation history
        self.conversation_history = f"System: {self.system_prompt}\n"
        
        self.initial_message = (
            "Hello! I'm your intake assistant. "
            "Let's start with your full name and date of birth."
        )

    def get_initial_message(self):
        return self.initial_message

    def reset(self):
        self.conversation_history = f"System: {self.system_prompt}\n"

    @mlflow.trace(name="generate_response", span_type=SpanType.AGENT)
    async def generate_response(self, user_input: str):

        try:
            # Add user message
            self.conversation_history += f"User: {user_input}\n"

            # Send to model
            response = self.agent_model(text=self.conversation_history)

            # Save model reply
            self.conversation_history += f"Agent: {response}\n"

            # Track turn
            self.tracker.log_turn(user_input, response)

            return response

        except Exception as e:
            self.logger.error(f"Error during agent response: {e}")
            return "Sorry, something went wrong."

    async def extract_structured_json(self):
        """
        Extract structured JSON using your JSON tool.
        """
        return extract_json(self.conversation_history)

    async def build_previsit_summary(self):
        """
        Call JSON tool → Summary builder → Return formatted text summary.
        """
        json_data = extract_json(self.conversation_history)
        summary = build_summary(json_data)
        return summary
