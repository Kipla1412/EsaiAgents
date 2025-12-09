from smolagents import Tool

class AskQuestionTool(Tool):
    name = "ask_question"
    description = "Ask the patient a medical intake question. Input must be the question string."

    inputs = {
        "question": {
            "type": "string",
            "description": "The question the agent should ask the patient."
        }
    }

    output_type = "string"

    def forward(self, question: str):
        return question
