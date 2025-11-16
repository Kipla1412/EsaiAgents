import os
import yaml
from dotenv import load_dotenv

class ConfigLoader:
    """
    Reusable configuration loader.
    - Loads .env file first (environment variables)
    - Then loads and parses YAML configuration file
    - Replaces placeholders like ${VAR_NAME} with actual env values
    """

    @staticmethod
    def load(config_path: str, dotenv_path: str = None) -> dict:
        """
        Load and merge configuration safely.
        Args:
            config_path: path to YAML config file
            dotenv_path: optional path to .env file (defaults to same directory)
        Returns:
            dict: merged config with environment variables substituted
        """
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            env_path = os.path.join(os.path.dirname(config_path), ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        def resolve_env_vars(obj):
            if isinstance(obj, dict):
                return {k: resolve_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_env_vars(i) for i in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_key = obj[2:-1]
                return os.getenv(env_key, f"<MISSING_ENV:{env_key}>")
            return obj

        config = resolve_env_vars(config)

        print(f"[ConfigLoader] Loaded config from: {config_path}")
        return config
