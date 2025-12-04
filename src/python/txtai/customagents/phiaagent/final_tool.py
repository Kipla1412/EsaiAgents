from smolagents import Tool

class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Returns the final natural-language answer to the user."

    inputs = {
        "answer": {
            "type": "string",
            "description": "Human friendly final answer produced after SQL execution."
        }
    }

    output_type = "string"

    def forward(self, answer: str) -> str:
        """
        Just returns the final human-readable answer.
        """
        return answer
