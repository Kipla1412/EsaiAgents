import asyncio
import mlflow
from mlflow.entities import SpanType
from ..base import BaseAgent


class AssessmentPlanAgent(BaseAgent):
    """
    Doctor-facing agent.
    Generates clinical Assessment & Plan from full conversation history.
    """

    def __init__(self, base_agent, config, tracker, logger):
        super().__init__(config, tracker, logger)
        self.agent_model = base_agent

    def add_message(self, role: str, text: str):
        """
        Store conversation turns for later clinical reasoning.
        """
        self.conversation_history += f"{role}: {text}\n"

    @mlflow.trace(name="assessment_plan_generate", span_type=SpanType.AGENT)
    async def generate_response(self):
        """
        Generate Assessment & Plan document.
        """

        prompt = f"""
        You are a clinical Assessment & Plan generation agent.

        Analyze the following patient conversation and generate a
        doctor-facing Assessment & Plan.

        Conversation:
        {self.conversation_history}

        --------------------------------------------------
        Assessment & Plan
        --------------------------------------------------

        A. Clinical Overview
        Briefly summarize key symptoms, duration, severity, and risk factors.

        B. Differential Diagnosis
        List the TOP 3-5 diagnoses in descending likelihood.

        For each diagnosis include:
        - Diagnosis name
        - Estimated likelihood percentage (total ~100%)
        - Clear clinical rationale

        Use this exact format:

        1. Diagnosis Name - ~XX% Likelihood
        • Rationale: ...

        C. Diagnostic Plan
        Group tests under:
        - Laboratory Tests
        - Imaging Studies (only if indicated)
        - Other Diagnostics

        For each test include:
        - Test name
        - Sample/source
        - Purpose

        D. Treatment Plan (Conditional)
        ONLY conditional recommendations.

        Use format:
        - If Diagnosis X is confirmed:
        • Medication
        • Example adult dose
        • Route
        • Duration
        • Rationale

        E. Procedures / Interventions
        Clearly state if none are indicated.

        F. Risk & Urgency Assessment
        - Urgency level: LOW / MODERATE / HIGH
        - Red-flag symptoms requiring escalation

        --------------------------------------------------

        STRICT RULES:
        - Plain text only
        - No markdown symbols
        - No patient-facing language
        - No definitive diagnosis
        - Output ONLY the Assessment & Plan document
        """

        try:
            response = await asyncio.to_thread(
                self.agent_model,
                prompt
            )

            return response if isinstance(response, str) else str(response)

        except Exception as e:
            self.logger.error(
                f"AssessmentPlanAgent error: {e}",
                exc_info=True
            )
            return "Assessment & Plan generation failed."
