
# from smolagents import Tool
# from ..summarybuilder import build_summary
# import json

# class BuildSummaryTool(Tool):
#     name = "build_summary"
#     description = "Build a formatted Pre-Visit Summary from extracted JSON."
#     inputs = {
#         "json_text": {
#             "type": "string",
#             "description": "A JSON string containing structured medical data."
#         }
#     }
#     output_type = "string"

#     def forward(self, json_text: str) -> str:
#         try:
#             data = json.loads(json_text)
#         except:
#             return "Invalid JSON input."

#         summary = build_summary(data)
#         return summary

# from smolagents import Tool
# import textwrap

# class SummaryTool(Tool):
#     name = "summary_tool"
#     description = "Summarizes the entire medical conversation."
#     inputs = {
#         "text": {
#             "type": "string",
#             "description": "Full conversation text"
#         }
#     }
#     output_type = "string"

#     def forward(self, text: str) -> str:
#         # Simple clean summarization (you can improve this)
       
#         summary = f"Conversation Summary:\n\n{textwrap.fill(text, 80)}"
#         return summary

# from smolagents import Tool
# import re 

# class SummaryTool(Tool):
#     name = "summary_tool"
#     description = "Generate a structured medical summary from conversation."
#     inputs = {
#         "text": {
#             "type": "string",
#             "description": "Full conversation transcript"
#         }
#     }
#     output_type = "string"
    
#     def forward(self, text: str) -> str:
#         # Extract from full conversation_history
#         patient_lines = [line for line in text.split('\n') if 'User:' in line]
#         concerns = [line.split(':', 1)[1].strip() for line in patient_lines]
        
#         summary = f"""DETAILED STRUCTURED SUMMARY

#         ----------------------------------------
#         Extract a structured summary from the following patient–assistant conversation.

#         Conversation:
#         {text}

#         Produce this EXACT format:

#         DETAILED STRUCTURED SUMMARY
#         ----------------------------------------
#         1. Presenting Concerns:
#         - <list all symptoms user mentioned>

#         2. Symptom Details:
#         - Onset: <if user said>
#         - Severity: <if user said>
#         - Associated symptoms: <list>

#         3. Assistant Guidance Given:
#         - <summaries of assistant messages>

#         4. Red Flags:
#         - <any emergency symptoms>

#         5. Recommended Actions:
#         - <doctor, urgent care, hydration, rest, etc.>

#         6. Additional Notes:
#         - <anything medically relevant>

#         Don't add anything outside this template.
#         """
#         return summary
    
# from smolagents import Tool

# class SummaryTool(Tool):
#     name = "summarize_conversation"
#     description = "Summarize the entire conversation into a medical-style summary."
#     inputs = {
#         "text": {
#             "type": "string",
#             "description": "Full conversation history"
#         }
#     }
#     output_type = "string"

#     def forward(self, text: str) -> str:
#         """
#         Very simple summarizer: you can improve this later or call an LLM inside.
#         For now it returns a clean paragraph summary.
#         """

#         # Here you can use rules or simple extraction.
#         # For now: return a plain summary.
#         return f"Summary of conversation:\n\n{text}"

# from smolagents import Tool

# class SummaryTool(Tool):
#     name = "soap_note_generator"
#     description = "Convert conversation history into a medical SOAP Note."
#     inputs = {
#         "text": {
#             "type": "string",
#             "description": "Full conversation history"
#         }
#     }
#     output_type = "string"

#     def forward(self, text: str) -> str:
#         """
#         Returns the instruction to generate a SOAP Note.
#         The agent LLM will finalize it.
#         """

from smolagents import Tool

class SummaryTool(Tool):
    name = "soap_note_generator"
    description = "ALWAYS use this tool when the patient ends the conversation or when a summary is required."
    inputs = {
        "text": {"type": "string", "description": "Full conversation history"}
    }
    output_type = "string"

    def forward(self, text: str) -> str:

        return f"""
        Convert the following conversation into a structured, professional medical SOAP Note.

        Conversation:
        {text}

        Follow this EXACT format:

        SOAP Note

        Subjective
        • List all symptoms and complaints clearly.
        • List any patient concerns or questions.

        Objective
        • Include only observable or self-reported facts.
        • Do NOT assume anything not stated by the patient.

        Assessment
        • Provide 2-4 likely diagnoses with reasoning.
        • Rank them (High / Moderate / Low likelihood).

        Plan
        • Supportive care recommendations.
        • Any initial medications (if appropriate).
        • Home monitoring instructions.
        • Red-flag symptoms requiring urgent care.
        • Follow-up advice.

        Write the SOAP Note cleanly with bullet points EXACTLY like this:

        SOAP Note
        Subjective
        • ...
        • ...
        Objective
        • ...
        Assessment
        • ...
        Plan
        • ...

        Do NOT add anything outside the SOAP Note. Only return the note.
        """

        # soap_prompt = f"""
        # Generate a detailed medical SOAP Note from the following conversation.

        # Conversation:
        # {text}

        # Use this exact structure:

        # SOAP Note

        # Subjective:
        # • List patient symptoms and complaints clearly.

        # Objective:
        # • List observable/self-reported factual details (no assumptions).

        # Assessment:
        # • Provide top 2-4 likely diagnoses with brief reasoning.

        # Plan:
        # • Recommend tests, supportive care, medications (if appropriate),
        # red flags, follow-up instructions.

        # Make the note clean, clinical, and professional.
        # """

        # # IMPORTANT: Only return prompt.
        # # The agent's LLM will handle generation.
        # return soap_prompt
