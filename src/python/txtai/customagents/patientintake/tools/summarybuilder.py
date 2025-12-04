from smolagents import Tool
from typing import Dict, Any

class SummaryBuilderTool(Tool):
    name = "build_summary"
    description = "Builds a clinician-ready pre-visit summary from structured intake JSON."

    inputs = {
        "data": {
            "type": "json",
            "description": "Structured JSON intake data."
        }
    }

    output_type = "string"

    def forward(self, data: Dict[str, Any]) -> str:
        pv = data.get("patient_verification", {})
        name = pv.get("full_name", "Information not provided.")
        dob = pv.get("dob", "Information not provided.")
        appt = pv.get("appointment_time", "Information not provided.")
        visit = pv.get("visit_type", "Information not provided.")

        cc = data.get("chief_complaint", {})
        verbatim = cc.get("verbatim_cc", "Information not provided.")

        hpi = cc.get("hpi_details", {})
        onset = hpi.get("onset", "Information not provided.")
        location = hpi.get("location", "Information not provided.")
        duration = hpi.get("duration", "Information not provided.")
        character = hpi.get("character", "Information not provided.")
        severity = hpi.get("severity_rating")
        severity_text = str(severity) if severity is not None else "Information not provided."
        aggr = hpi.get("aggravating_factors", "Information not provided.")
        allev = hpi.get("alleviating_factors", "Information not provided.")
        assoc = hpi.get("associated_symptoms", "Information not provided.")

        summary = f"""
Patient Name: {name}
DOB: {dob}
Appointment Time: {appt}
Visit Type: {visit}

1. CHIEF COMPLAINT
    - {verbatim}

2. HISTORY OF PRESENT ILLNESS
    - Onset: {onset}
    - Location: {location}
    - Duration: {duration}
    - Character: {character}
    - Severity: {severity_text}
    - Aggravating: {aggr}
    - Alleviating: {allev}
    - Associated Symptoms: {assoc}

Thank you for sharing this information.
"""
        return summary
