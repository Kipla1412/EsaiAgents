import re

def parse_soap_note(text: str):
    """
    Convert full SOAP Note text into a dictionary:
    { subjective: "...", objective: "...", assessment: "...", plan: "..." }
    """

    sections = {
        "subjective": "",
        "objective": "",
        "assessment": "",
        "plan": ""
    }

    current = None

    for line in text.split("\n"):
        line = line.strip()

        # Detect sections
        if re.match(r"^Subjective$", line, re.IGNORECASE):
            current = "subjective"
            continue

        if re.match(r"^Objective$", line, re.IGNORECASE):
            current = "objective"
            continue

        if re.match(r"^Assessment$", line, re.IGNORECASE):
            current = "assessment"
            continue

        if re.match(r"^Plan$", line, re.IGNORECASE):
            current = "plan"
            continue

        # Append content to active section
        if current:
            sections[current] += line + "\n"

    return sections
