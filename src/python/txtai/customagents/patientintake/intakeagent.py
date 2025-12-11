

import asyncio
import mlflow
from mlflow.entities import SpanType
from ..base import BaseAgent
from datetime import datetime


from ..util import parse_soap_note, SOAPReportGenerator


class MedicalConversationAgent(BaseAgent):

    def __init__(self, base_agent, tools, system_prompt, config, tracker, logger, patient_info):
        super().__init__(config, tracker, logger)

        # attach txtai Agent model
        self.agent_model = base_agent
        self.tools = tools 
        self.patient_info = patient_info

        # attach tools to model 
        self.agent_model.process.model.tools = [t["tool"] for t in tools if t["name"] == "final_answer"]

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

    async def generate_summary(self)  -> str:

        """
        SUMMARY:
        - Called by backend when session ends (exit, timeout, disconnect, etc).
        - Does NOT rely on LLM tool selection.
        - Uses the internal summary tool, then asks LLM to generate SOAP Note.
        """
        # Find the summary tool by name
        summary_tool = None
        for t in self.tools:
            if t["name"] == "soap_note_generator":
                summary_tool = t["tool"]
                break

        if summary_tool is None:
            self.logger.error("Summary tool 'soap_note_generator' not found.")
            return "No summary tool found."
        
        prompt = summary_tool.forward(self.conversation_history)
        try:
        # Step 2: Send prompt to LLM to generate full SOAP Note
            summary = await asyncio.to_thread(self.agent_model, prompt)

            if not isinstance(summary, str):
                summary = str(summary)

            self.tracker.log_turn("[AUTO_SUMMARY_TRIGGER]", summary)

            return summary
        except Exception as e:
            self.logger.error(f"Error during auto summary generation: {e}", exc_info=True)
            return "Unable to generate summary at this time."
    
    async def generate_pdf_report(self, pdf_path="soap_report.pdf"):
        """
        Full pipeline: summary → parse SOAP → generate PDF
        """
        soap_text = await self.generate_summary()

        sections = parse_soap_note(soap_text)
        sections["date"] = datetime.now().strftime("%d-%m-%Y")


        pdf = SOAPReportGenerator(self.patient_info)
        output = pdf.generate(sections, pdf_path)

        return output
