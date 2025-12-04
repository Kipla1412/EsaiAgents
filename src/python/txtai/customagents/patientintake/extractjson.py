import json
import re
from typing import Any, Dict

def extract_json(model_text: str) -> Dict[str, Any]:
    """
    Extracts a single JSON object from LLM text output.
    Cleans common formatting issues and returns parsed JSON.
    If invalid, returns {"error": "..."}.
    """

    # Extract potential JSON blocks
    matches = re.findall(r"\{.*\}", model_text, flags=re.DOTALL)
    if not matches:
        return {"error": "No JSON object found in model output."}

    # Heuristic: largest block = most likely valid JSON
    candidate = max(matches, key=len)

    # Helper for safe JSON load
    def try_load(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    # First attempt: direct parse
    parsed = try_load(candidate)
    if parsed is not None:
        return parsed

    # 3Try cleaning common LLM noise
    cleaned = candidate

    # Remove newlines
    cleaned = cleaned.replace("\n", " ")

    # Replace single quotes → double quotes
    cleaned = cleaned.replace("'", '"')

    # Remove trailing commas in objects + arrays
    cleaned = re.sub(r",\s*}", "}", cleaned)
    cleaned = re.sub(r",\s*]", "]", cleaned)

    # Try parsing cleaned text
    parsed = try_load(cleaned)
    if parsed is not None:
        return parsed

    # If still invalid, return error
    return {
        "error": "JSON parsing failed",
        "raw_candidate": candidate[:500]  # return snippet for debugging
    }
