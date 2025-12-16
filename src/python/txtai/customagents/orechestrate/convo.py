import asyncio
import mlflow
from mlflow.entities import SpanType
from ..base import BaseAgent


class ConversationalAgent(BaseAgent):

    def __init__(self, agent_model, config, tracker, logger):
        super().__init__(config, tracker, logger)
        self.agent_model = agent_model
        self.initial_message = (
            "Hello! I am your medical assistant. "
            "How can I help you today?"
        )

    def get_initial_message(self):
        return self.initial_message

    def add_message(self, role: str, text: str):
        self.conversation_history += f"{role}: {text}\n"

    def reset(self):
        self.conversation_history = ""

    @mlflow.trace(name="generate_response", span_type=SpanType.AGENT)
    async def generate_response(self, user_input: str) -> str:

        try:
            # Add user message to history
            self.add_message("User", user_input)

            # Run model (sync → async safe)
            response = await asyncio.to_thread(
                self.agent_model,
                self.conversation_history.strip()     # cleaner
            )

            # Save assistant response
            self.add_message("Agent", response)

            # Track the turn
            self.tracker.log_turn(user_input, response)

            return response

        except Exception as e:
            self.logger.error(f"ConversationalAgent error: {e}", exc_info=True)
            return "Sorry, I encountered an error while processing your request."
