# import mlflow
# from mlflow.entities import SpanType
# from ..base import BaseAgent
# from .tools.extractjson import ExtractJSONTool
# from .tools.askquestion import AskQuestionTool
# from .tools.buildersummary import SummaryTool
# import importlib.resources as resources
# from ..resourceloader import ConfigResourceLoader

# class MedicalIntakeAgent(BaseAgent):

#     def __init__(self, agent_model, config, tracker, logger, tools):
#         super().__init__(config, tracker, logger)

#         self.agent_model = agent_model
#         self.tracker = tracker
#         self.logger = logger
#         #self.tools = tools
#         # Load system prompt
#         with resources.open_text("txtai.customagents.patientintake", "systemprompt.txt") as f:
#             self.system_prompt = f.read()
#         self.agent_model.process.model.tools = tools
#         # Keep conversation history
#         self.conversation_history = f"System: {self.system_prompt}\n"
        
#         self.initial_message = (
#             "Hello! I'm your intake assistant. "
#             "Let's start with your full name and date of birth."
#         )

#     def get_initial_message(self):
#         return self.initial_message

#     def reset(self):
#         self.conversation_history = f"System: {self.system_prompt}\n"

#     @mlflow.trace(name="generate_response", span_type=SpanType.AGENT)
#     async def generate_response(self, user_input: str):

#         try:
#             # Add user message
#             self.conversation_history += f"User: {user_input}\n"

#             # Send to model
#             response = self.agent_model(text=self.conversation_history)

#             # Save model reply
#             self.conversation_history += f"Agent: {response}\n"

#             # Track turn
#             self.tracker.log_turn(user_input, response)

#             return response

#         except Exception as e:
#             self.logger.error(f"Error during agent response: {e}")
#             return "Sorry, something went wrong."

    
#     # async def extract_structured_json(self):
#     #     jsontool = self.tools[0]
#     #     return jsontool.forward(self.conversation_history)

#     async def build_previsit_summary(self):
#         #jsontool = self.tools[0]
#         summarytool = self.tools[0]

        
#         return summarytool.forward(self.conversation_history)

import asyncio
import mlflow
from mlflow.entities import SpanType
from ..base import BaseAgent


class MedicalConversationAgent(BaseAgent):

    def __init__(self, base_agent, tools, system_prompt, config, tracker, logger):
        super().__init__(config, tracker, logger)

        # attach txtai Agent model
        self.agent_model = base_agent

        # attach tools to model 
        self.agent_model.process.model.tools = [t["tool"] for t in tools]

        # inject system prompt into model
        try:
            self.agent_model.process.model.llm.system = system_prompt
        except Exception:
            logger.warning("Could not set system prompt in model")

        # conversation memory
        self.conversation_history = ""
        self.logger = logger
        self.tracker = tracker

        self.initial_message = "Assistant: Hello! How can I help you today?"

    def reset(self):
        self.conversation_history = ""
        self.logger.info("MedicalConversationAgent reset")
    
    def get_initial_message(self):
        return self.initial_message

    @mlflow.trace(name="medical_generate", span_type=SpanType.AGENT)
    async def generate_response(self, user_input: str):

        try:
            # save user input
            self.conversation_history += f"User: {user_input}\n"

            # run model (txtai is sync → thread)
            response = await asyncio.to_thread(self.agent_model, self.conversation_history)

            # normalize
            if not isinstance(response, str):
                response = str(response)

            # save agent output
            self.conversation_history += f"Assistant: {response}\n"
            
            # log to tracker
            self.tracker.log_turn(user_input, response)

            # return text
            return response

        except Exception as e:
            self.logger.error(f"MedicalConversationAgent error: {e}", exc_info=True)
            return "Assistant: Sorry, something went wrong."

    async def generate_summary(self):
        # Find the summary tool by name
        summary_tool = None
        for t in self.tools:
            if t["name"] == "soap_note_generator":
                summary_tool = t["tool"]
                break

        if summary_tool is None:
            return "No summary tool found."

        # return summary_tool.forward(self.conversation_history)
        # Step 1: Create SOAP prompt using Conversation
        prompt = summary_tool.forward(self.conversation_history)

    # Step 2: Send prompt to LLM to generate full SOAP Note
        response = await asyncio.to_thread(self.agent_model, prompt)

        return response

