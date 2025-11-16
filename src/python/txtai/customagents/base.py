from abc import ABC, abstractmethod

class BaseAgent(ABC):

    """Base template for all agents"""

    def __init__(self, config, tracker, logger):

        self.config = config  # credinals ,dtabase connection, api manage
        self.tracker = tracker
        self.logger = logger
        self.conversation_history = ""

    @abstractmethod
    async def generate_response(self, user_input): 
        """ Every agent must implement its own response logic"""

        pass

    def reset_conversation(self):

        self.conversation_history = ""
        self.logger.info("conversation reset...(-_-)")
        