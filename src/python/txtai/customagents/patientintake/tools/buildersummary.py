
from smolagents import Tool

class SummaryTool(Tool):
    name = "soap_note_generator"
    description = "Generate a medical SOAP Note summary from the full conversation history."
    inputs = {
        "text": {"type": "string", "description": "Full conversation history"}
    }
    output_type = "string"

    def forward(self, text: str) -> str:

        """
        Build a prompt that asks the LLM to generate a SOAP note.
        The MedicalConversationAgent will send this prompt to the LLM.
        """

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

        
