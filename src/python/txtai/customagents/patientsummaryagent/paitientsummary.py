import asyncio
import mlflow
from mlflow.entities import SpanType
from ..base import BaseAgent

class SimpleSummaryAgent(BaseAgent):

    def __init__(self, base_agent, config, tracker, logger):
        super().__init__(config, tracker, logger)

        self.agent_model = base_agent
        self.conversation_history = ""
        self.logger = logger
        self.tracker = tracker

    def add_message(self, role: str, text: str):
        self.conversation_history += f"{role}: {text}\n"
    
    async def generate_response(self, user_input):
        raise NotImplementedError("SimpleSummaryAgent does not support chat responses.")
    
    @mlflow.trace(name="simple_summary", span_type=SpanType.AGENT)
    async def generate_summary(self):

        prompt = f"""
        Summarize the following patient conversation into one clear,
        human-readable paragraph.

        Conversation:
        {self.conversation_history}

        Rules:
        • Use simple, natural language
        • No bullet points
        • No diagnoses
        • No assumptions
        • Just describe the main symptoms and context

        Write ONE short paragraph only.
        """

        try:
            response = await asyncio.to_thread(
                self.agent_model,
                prompt
            )

            return response if isinstance(response, str) else str(response)

        except Exception as e:
            self.logger.error(f"SimpleSummaryAgent error: {e}", exc_info=True)
            return "Summary generation failed."
