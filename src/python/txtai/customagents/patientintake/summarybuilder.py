
from typing import Dict, Any

def build_summary(data: Dict[str, Any]) -> str:
    """
    Converts extracted JSON into a doctor-ready
    Pre-Visit Summary exactly matching the required format.
    """

    #  Patient Verification 
    pv = data.get("patient_verification", {})
    name = pv.get("full_name") or "Information not provided."
    dob = pv.get("dob") or "Information not provided."
    appt = pv.get("appointment_time") or "Information not provided."
    visit = pv.get("visit_type") or "Information not provided."

    # Chief Complaint
    cc = data.get("chief_complaint", {})
    verbatim = cc.get("verbatim_cc") or "Information not provided."

    # HPI Details 
    hpi = cc.get("hpi_details", {})
    onset = hpi.get("onset") or "Information not provided."
    location = hpi.get("location") or "Information not provided."
    duration = hpi.get("duration") or "Information not provided."
    character = hpi.get("character") or "Information not provided."
    severity = hpi.get("severity_rating")
    severity_text = str(severity) if severity is not None else "Information not provided."
    aggr = hpi.get("aggravating_factors") or "Information not provided."
    allev = hpi.get("alleviating_factors") or "Information not provided."
    radiation = hpi.get("radiation", "Information not provided.")
    timing = hpi.get("timing", "Information not provided.")
    assoc = hpi.get("associated_symptoms", "Information not provided.")

    # PMH / PSH / Meds / Allergies / FH 
    pmh = data.get("past_medical_history", {}).get("conditions", [])
    psh = data.get("past_surgical_history", {}).get("surgeries", [])
    meds = data.get("medications", [])
    allergies = data.get("allergies", [])
    fh = data.get("family_history", {}).get("conditions", [])

    # Social History
    sh = data.get("social_history", {})
    tobacco = sh.get("tobacco_use") or "Information not provided."
    alcohol = sh.get("alcohol_use") or "Information not provided."
    substance = sh.get("substance_use") or "Information not provided."
    occupation = sh.get("occupation") or "Information not provided."
    living = sh.get("living_situation") or "Information not provided."

    # --- ROS ---
    ros = data.get("review_of_systems", {})
    cons = ros.get("constitutional", {})

    fever = cons.get("fever")
    chills = cons.get("chills")
    weight_change = cons.get("weight_change_unexplained")
    night_sweats = cons.get("night_sweats")

    other_ros = ros.get("other_pertinent_positives") or "Information not provided."

    # Helper for list formatting
    def list_items(items, formatter):
        if not items:
            return "None."
        return "\n    ".join(formatter(i) for i in items)

    pmh_text = list_items(pmh, lambda x: f"{x.get('condition_name','')} ({x.get('diagnosis_year_approx','')})")
    psh_text = list_items(psh, lambda x: f"{x.get('procedure_name','')} ({x.get('approx_year','')})")
    meds_text = list_items(meds, lambda x: f"{x.get('name','')} ({x.get('dosage') or ''}, {x.get('frequency') or ''})")
    allergies_text = list_items(allergies, lambda x: f"{x.get('allergen','')} → {x.get('reaction_type','')}")
    fh_text = list_items(fh, lambda x: f"{x.get('condition_name','')} — {x.get('relative','')}")

    # Build final summary
    summary = f"""
Patient Name: {name}
DOB: {dob}
Appointment Time: {appt}
Visit Type: {visit}

1. CHIEF COMPLAINT (CC)
    - {verbatim}

2. HISTORY OF PRESENT ILLNESS (HPI)
    - Onset: {onset}
    - Location: {location}
    - Duration: {duration}
    - Character: {character}
    - Severity (1–10): {severity_text}
    - Aggravating/Alleviating Factors: {aggr} / {allev}
    - Radiation: {radiation}
    - Timing: {timing}
    - Associated Symptoms: {assoc}

3. PAST MEDICAL HISTORY (PMH)
    - {pmh_text}

4. PAST SURGICAL HISTORY (PSH)
    - {psh_text}

5. MEDICATIONS
    - {meds_text}

6. ALLERGIES
    - {allergies_text}

7. SOCIAL HISTORY (SH)
    - Tobacco: {tobacco}
    - Alcohol: {alcohol}
    - Substance: {substance}
    - Occupation: {occupation}
    - Living Situation: {living}

8. FAMILY HISTORY (FH)
    - {fh_text}

9. REVIEW OF SYSTEMS (ROS)
    - Constitutional:
        Fever: {fever}
        Chills: {chills}
        Weight Change (Unexplained): {weight_change}
        Night Sweats: {night_sweats}

    - Other Pertinent Positives/Negatives:
        {other_ros}

Thank you for sharing all this information. I’ll prepare your pre-visit summary for your doctor.
"""

    return summary
