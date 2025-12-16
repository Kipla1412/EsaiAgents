class MedicalAgentOrchestrator:

    def __init__(self, chat_agent, summary_agent, soap_agent, assessment_agent):
        self.chat_agent = chat_agent
        self.summary_agent = summary_agent
        self.soap_agent = soap_agent
        self.assessment_agent = assessment_agent

    async def handle_user_message(self, text: str) -> str:
        """
        Main entry point for user messages.
        1. Sends message to ConversationalAgent
        2. Syncs conversation into Summary + SOAP agents
        """

        # Get chat agent response
        reply = await self.chat_agent.generate_response(text)

        # Sync conversation into other agents
        self._sync_to_summary(text, reply)
        self._sync_to_soap(text, reply)
        self._sync_to_assessment(text, reply)

        return reply

    def _sync_to_summary(self, user_msg: str, bot_msg: str):
        """Mirror conversation into SimpleSummaryAgent."""
        self.summary_agent.add_message("User", user_msg)
        self.summary_agent.add_message("Agent", bot_msg)

    def _sync_to_soap(self, user_msg: str, bot_msg: str):
        """Mirror conversation into SOAPNoteAgent."""
        self.soap_agent.add_message("User", user_msg)
        self.soap_agent.add_message("Agent", bot_msg)

    def _sync_to_assessment(self, user_msg: str, bot_msg: str):
        self.assessment_agent.add_message("User", user_msg)
        self.assessment_agent.add_message("Agent", bot_msg)

    async def generate_summary(self) -> str:
        return await self.summary_agent.generate_summary()

    async def generate_soap(self) -> str:
        return await self.soap_agent.generate_soap_note()
    
    async def generate_assessment_plan(self) -> str:
        return await self.assessment_agent.generate_response()

    def reset(self):
        """Clears history across all agents (new patient)."""
        self.chat_agent.reset()
        self.summary_agent.conversation_history = ""
        self.soap_agent.conversation_history = ""
