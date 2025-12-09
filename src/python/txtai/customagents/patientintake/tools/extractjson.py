# from smolagents import Tool
# import json, re
# from typing import Dict, Any

# class ExtractJSONTool(Tool):
#     name = "extract_json"
#     description = "Extracts structured JSON from the full conversation history."

#     inputs = {
#         "conversation": {
#             "type": "string",
#             "description": "The full conversation transcript."
#         }
#     }

#     output_type = "json"

#     def forward(self, conversation: str) -> Dict[str, Any]:
#         matches = re.findall(r"\{.*\}", conversation, flags=re.DOTALL)
#         if not matches:
#             return {"error": "No JSON object found"}

#         candidate = max(matches, key=len)

#         try:
#             return json.loads(candidate)
#         except:
#             cleaned = candidate.replace("'", '"')
#             cleaned = re.sub(r",\s*}", "}", cleaned)
#             cleaned = re.sub(r",\s*]", "]", cleaned)
#             try:
#                 return json.loads(cleaned)
#             except:
#                 return {"error": "Invalid JSON", "raw": candidate[:400]}
import json
import re
from smolagents import Tool

def extract_json_func(text: str):
    """
    Pure JSON extraction logic.
    """
    matches = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    if not matches:
        return json.dumps({"error": "No JSON found"})

    candidate = max(matches, key=len)

    try:
        return json.dumps(json.loads(candidate))
    except:
        cleaned = candidate.replace("'", "\"")
        cleaned = re.sub(r",\s*}", "}", cleaned)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            return json.dumps(json.loads(cleaned))
        except:
            return json.dumps({"error": "JSON parse failed"})
    

class ExtractJSONTool(Tool):
    name = "extract_json"
    description = "Extract clean JSON from the conversation history text."
    inputs = {
        "text": {
            "type": "string",
            "description": "Full conversation log containing the patient's dialogue."
        }
    }
    output_type = "string"

    def forward(self, text: str) -> str:
        return extract_json_func(text)
