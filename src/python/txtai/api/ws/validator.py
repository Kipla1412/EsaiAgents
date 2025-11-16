
class MessageValidator:
    """
    Smart WebSocket message validator.
    Filters browser noise and ensures real human text.
    """

    IGNORE_SET = {
        "",
        " ",
        "\n",
        "undefined",
        "null",
        "__start__",
        "start"
    }

    def __init__(self):
        self.seen_first_real = False   # Track first real message

    def is_valid(self, msg: str) -> bool:
        """
        Returns True only for meaningful human messages.
        Handles first-message junk filtering.
        """

        # Must be a non-empty string
        if not msg or not isinstance(msg, str):
            return False

        cleaned = msg.strip().lower()

        # Always ignored values
        if cleaned in self.IGNORE_SET:
            return False
        
        if not self.seen_first_real:
            if any(ch.isalpha() for ch in cleaned):
                self.seen_first_real = True
                return True
            return False

       
        return cleaned != ""
