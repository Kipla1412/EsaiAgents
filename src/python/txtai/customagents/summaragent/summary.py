# import asyncio
# import mlflow
# from mlflow.entities import SpanType
# from ..base import BaseAgent


# class SOAPNoteAgent(BaseAgent):

#     def __init__(self, base_agent, config, tracker, logger):
#         super().__init__(config, tracker, logger)

#         self.agent_model = base_agent
#         self.conversation_history = ""
#         self.logger = logger
#         self.tracker = tracker

#     def add_message(self, role: str, text: str):
#         self.conversation_history += f"{role}: {text}\n"

#     @mlflow.trace(name="soap_generate", span_type=SpanType.AGENT)
#     async def generate_soap_note(self):
#         """
#         Sends the conversation history to the LLM so it writes
#         a structured SOAP Note automatically.
#         """

#         prompt = f"""
#         Convert the following patient conversation into a complete, professional SOAP Note.

#         Conversation:
#         {self.conversation_history}

#         Follow this exact structure:

#         SOAP Note

#         Subjective
#         • List all symptoms and complaints mentioned by the patient.
#         • Include patient's own descriptions and concerns.

#         Objective
#         • Include only observable facts or self-reported measurable details.
#         • Do NOT invent or assume anything not explicitly stated.

#         Assessment
#         • Provide 2-4 possible clinical impressions (no definitive diagnosis).
#         • Rank each as High / Moderate / Low likelihood with 1–2 lines of reasoning.

#         Plan
#         • Supportive care recommendations.
#         • Home care instructions.
#         • Monitoring advice.
#         • Red-flag symptoms that require urgent medical evaluation.
#         • Follow-up suggestions.

#         Output ONLY the SOAP Note. Do NOT add explanations, disclaimers, or extra text.
#         """

#         try:
#             response = await asyncio.to_thread(
#                 self.agent_model,
#                 prompt
#             )

#             if not isinstance(response, str):
#                 response = str(response)

#             return response

#         except Exception as e:
#             self.logger.error(f"SOAPNoteAgent error: {e}", exc_info=True)
#             return "SOAP Note generation failed."

import asyncio
import mlflow
from mlflow.entities import SpanType
from ..base import BaseAgent


class SOAPNoteAgent(BaseAgent):

    def __init__(self, base_agent, config, tracker, logger):
        super().__init__(config, tracker, logger)
        self.agent_model = base_agent

    def add_message(self, role, text):
        self.conversation_history += f"{role}: {text}\n"

    async def generate_response(self, user_input):
        raise NotImplementedError("SOAPNoteAgent does not support chat responses.")
    
    @mlflow.trace(name="soap_generate", span_type=SpanType.AGENT)
    async def generate_soap_note(self):

        prompt = f"""
Convert the following patient conversation into a structured SOAP Note.

Conversation:
{self.conversation_history}

Follow this exact format:

SOAP Note

Subjective
• List all symptoms reported by the patient.

Objective
• Include only measurable or observable details mentioned.

Assessment
• Provide 2-4 possible clinical impressions.
• Label each as High / Moderate / Low likelihood.
• Add 1-2 lines of reasoning.

Plan
• Supportive care steps
• Home instructions
• Monitoring advice
• Red flags for urgent care
• Follow-up guidance
"""

        try:
            response = await asyncio.to_thread(
                self.agent_model,
                prompt
            )

            return response if isinstance(response, str) else str(response)

        except Exception as e:
            self.logger.error(f"SOAPNoteAgent error: {e}", exc_info=True)
            return "SOAP Note generation failed."
