import mlflow
from mlflow.entities import SpanType
from ..base import BaseAgent

class ConversationalAgent(BaseAgent):

    def __init__(self, agent_model, config, tracker, logger):

        super().__init__(config, tracker, logger)
        self.agent_model = agent_model
        self.initial_message ="Hello! I am your medical assistant. How can I help you today?"

    def get_initial_message(self):

        return self.initial_message

    def reset(self):
        self.conversation_history = ""

    @mlflow.trace(name="generate_response", span_type=SpanType.AGENT)
    async def generate_response(self, user_input: str):
        
        try:
            self.conversation_history += f"User: {user_input}\n"
            response = self.agent_model(text=self.conversation_history)
            self.conversation_history += f"Agent: {response}\n"
            self.tracker.log_turn(user_input, response)
            return response
        
        except Exception as e:
            self.logger.error(f"Error during agent response generation: {e}")
            return "Sorry, I encountered an error processing your request."

