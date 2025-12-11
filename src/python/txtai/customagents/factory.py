from ..agent.base import Agent
from .chatagent.chat import ConversationalAgent
from ..eval.tracker import Tracker
from ..txtailogging.logger import get_logger
from .speechtotext.stt import SpeechToText
from .texttospeech.tts import TextToSpeechAgent

from .phiaagent.phia import PHIAAgent
from .phiaagent.duckdbtool import DuckDBSQLTool
from txtai.agent.tool.embeddings import EmbeddingsTool
from .phiaagent.final_tool import FinalAnswerTool
from .phiaagent.build_prompt import build_system_prompt

from .patientintake.intakeagent import MedicalConversationAgent
from .patientintake.tools.extractjson import ExtractJSONTool
#from .patientintake.tools.askquestion import AskQuestionTool
from .patientintake.tools.buildersummary import SummaryTool
import importlib.resources as resources
import os

class AgentFactory:

    @staticmethod
    def create_agent(agent_type: str, config):

        # Extract config blocks
        llm_config = config.get("llm", {})
        agent_config = config.get("agent", {})

        # Create tracker + logger for every agent
        tracker = Tracker()
        tracker.log_static(llm_config, agent_config)
        logger = get_logger(agent_type)

        if agent_type == "conversational":
            base_agent = Agent(**{
                **agent_config,
                "model": llm_config,
                "tools": []
            })
            return ConversationalAgent(base_agent, config, tracker, logger)

        elif agent_type == "speech_to_text":
            return SpeechToText(config, tracker, logger)

        elif agent_type == "text_to_speech":
            return TextToSpeechAgent(config, tracker, logger)

        elif agent_type == "phia":

            phia_config = config.get("phia", {})

            # Validate config
            required = ["summary_path", "activities_path", "memory"]
            for item in required:
                if not phia_config.get(item):
                    raise ValueError(f"Missing phia.{item}")

            sqltool = DuckDBSQLTool(
                summary_path=phia_config["summary_path"],
                activities_path=phia_config["activities_path"]
            )
            sqltool.name = "sql"

            embedtool = EmbeddingsTool(phia_config["memory"])
            finaltool = FinalAnswerTool()

            tools = [
                {"name": "sql", "tool": sqltool},
                {"name": "final_answer", "tool": finaltool},
            ]

            if hasattr(embedtool, "name") and embedtool.name:
                tools.append({"name": embedtool.name, "tool": embedtool})

            schema_text = sqltool.get_schema()

            fewshot_example = None
            fpath = phia_config.get("fewshots_path")
            if fpath and os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        fewshot_example = f.readline().strip()
                except:
                    logger.warning("Could not read fewshot example")

            system_prompt = build_system_prompt(schema_text, fewshot_example)

            llm_config["system"] = system_prompt

            base_agent = Agent(**{
                **agent_config,
                "model": llm_config,
                "tools": [t["tool"] for t in tools]
            })

            return PHIAAgent(
                base_agent=base_agent,
                tools=tools,
                system_prompt=system_prompt,
                config=config,
                tracker=tracker,
                logger=logger
            )
        
        elif agent_type == "medical_intake":

            #asktool = AskQuestionTool()
            #jsontool = ExtractJSONTool()
            summarytool = SummaryTool()
            finaltool = FinalAnswerTool()

            llm_tools = [
                {"name": "final_answer", "tool": finaltool}
            ]

            # Tools used only by Python (not exposed to LLM)
            system_tools = [
                {"name": "soap_note_generator", "tool": summarytool}
            ]

            tools = llm_tools + system_tools
            
            patient_info = config.get("patient_info", {})

            # tools = [
            #     {"name": "final_answer", "tool": finaltool},
            #     {"name": "soap_note_generator", "tool": summarytool},
            # ]
            #tools = [summarytool, finaltool]
                 
            base_agent = Agent(**{
                **agent_config,
                "model": llm_config,
                "tools": [t["tool"] for t in llm_tools]
            })

            with resources.open_text("txtai.customagents.patientintake", "systemprompt.txt") as f:
                system_prompt = f.read()

            return MedicalConversationAgent(
                base_agent=base_agent,
                tools=tools,
                system_prompt=system_prompt,
                config=config,
                tracker=tracker,
                logger=logger,
                patient_info=patient_info
                 
            )

        raise ValueError(f"Unknown agent type: {agent_type}")
