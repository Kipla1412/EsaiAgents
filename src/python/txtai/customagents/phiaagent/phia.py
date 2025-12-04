
import asyncio
import mlflow
from mlflow.entities import SpanType

from ..base import BaseAgent

class PHIAAgent(BaseAgent):

    def __init__(self, base_agent, tools, system_prompt, config, tracker, logger):
        """
        base_agent      : Agent object
        tools           : [{"name":..., "tool":...}]
        system_prompt   : prompt text
        config          : agent config
        tracker         : Tracker object
        logger          : logger
        """
        super().__init__(config, tracker, logger)

        # EXACTLY same naming as ConversationalAgent
        self.agent_model = base_agent

        # Attach tools to model
        self.agent_model.process.model.tools = [t["tool"] for t in tools]

        # Inject system prompt
        try:
            self.agent_model.process.model.llm.system = system_prompt
        except Exception:
            logger.warning("Failed to set system prompt into txtai model")

        self.conversation_history = ""
        self.initial_message = "PHIA Agent ready!"

    def reset(self):
        self.conversation_history = ""
        self.logger.info("PHIAAgent conversation reset.")

    @mlflow.trace(name="phia_generate", span_type=SpanType.AGENT)
    async def generate_response(self, question: str):
        """
        EXACT SAME PATTERN as ConversationalAgent
        but runs the txtai SQL reasoning agent.
        """
        try:
            # Keep history like ConversationalAgent
            self.conversation_history += f"User: {question}\n"

            # txtai Agent is synchronous; wrap in thread executor
            response = await asyncio.to_thread(self.agent_model, question)

            # Normalise string
            response = response if isinstance(response, str) else str(response)

            # Add to history
            self.conversation_history += f"Agent: {response}\n"

            # Tracker logging
            self.tracker.log_turn(question, response)

            return response
        
        except Exception as e:
            self.logger.error(f"PHIA error: {e}", exc_info=True)
            return "PHIA processing error"
