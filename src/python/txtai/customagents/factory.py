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

from .patientintake.tools.buildersummary import SummaryTool

from .patientsummaryagent import SimpleSummaryAgent
from .summaragent.summary import SOAPNoteAgent
from .orechestrate.orco import MedicalAgentOrchestrator
from .orechestrate.convo import ConversationalAgent
from .planassesmentagent import AssessmentPlanAgent

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
        elif agent_type == "simple_summary":

            base_agent = Agent(**{
                **agent_config,
                "model": llm_config,
                "tools": []
            })

            return SimpleSummaryAgent(
                base_agent=base_agent,
                config=config,
                tracker=tracker,
                logger=logger
            )
        
        elif agent_type == "soap_note":

            base_agent = Agent(**{
                **agent_config,
                "model": llm_config,
                "tools": []   # SOAP agent has no tools
            })

            return SOAPNoteAgent(
                base_agent=base_agent,
                config=config,
                tracker=tracker,
                logger=logger
            )

        
        elif agent_type == "conversation":
            base_agent = Agent(**{
                **agent_config,
                "model": llm_config,
                "tools": []
            })
            return ConversationalAgent(base_agent, config, tracker, logger)
        
        elif agent_type == "assessment_plan":

            base_agent = Agent(**{
                **agent_config,
                "model": llm_config,
                "tools": []
            }) 

            return AssessmentPlanAgent(
                base_agent=base_agent,
                config=config,
                tracker=tracker,
                logger=logger
            )

        elif agent_type == "medical_orchestrator":

            chat_agent = AgentFactory.create_agent("conversation", config)
            summary_agent = AgentFactory.create_agent("simple_summary", config)
            soap_agent = AgentFactory.create_agent("soap_note", config)
            assessment_agent =AgentFactory.create_agent("assessment_plan", config)

            return MedicalAgentOrchestrator(
                chat_agent=chat_agent,
                summary_agent=summary_agent,
                soap_agent=soap_agent,
                assessment_agent= assessment_agent
            )


        raise ValueError(f"Unknown agent type: {agent_type}")
