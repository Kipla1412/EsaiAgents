
# from fastapi import WebSocket, WebSocketDisconnect
# from ...customagents.configloader import ConfigLoader
# from ...customagents.factory import AgentFactory
# from ...customagents import chatagent
# from ...txtailogging.logger import get_logger
# import importlib.resources as resources
# from dotenv import load_dotenv
# from .validator import MessageValidator
# import mlflow
# import asyncio
# import time

# from fastapi.middleware.cors import CORSMiddleware


# SILENCE_TIMEOUT = 15   # seconds


# def register_medical_ws(app):

#     load_dotenv(dotenv_path="txtai/src/python/txtai/.env")
#     logger = get_logger("MedicalWS")

#     with resources.open_text(chatagent, "medical.yml") as f:
#         config = ConfigLoader.load(f.name)

#     mlflow.set_experiment("medical_orchestrator")

#     orchestrator = AgentFactory.create_agent("medical_orchestrator", config)
#     validator = MessageValidator()

#     @app.websocket("/ws/medical")
#     async def websocket_medical(websocket: WebSocket):

#         await websocket.accept()
#         logger.info("Client connected → /ws/medical")

#         orchestrator.reset()

#         greeting = orchestrator.chat_agent.get_initial_message()
#         await websocket.send_text(greeting)

#         last_message_time = time.time()

#         try:
#             while True:

#                 # --- SILENCE TIMEOUT CHECK ---
#                 if time.time() - last_message_time > SILENCE_TIMEOUT:
#                     await _auto_generate_reports(websocket, orchestrator, logger, reason="silence")
#                     break

#                 try:
#                     msg = await asyncio.wait_for(websocket.receive_text(), timeout=1)
#                 except asyncio.TimeoutError:
#                     continue

#                 text = msg.strip()
#                 last_message_time = time.time()

#                 if not validator.is_valid(text):
#                     continue

#                 # ----- END COMMAND -----
#                 if text == "__end__":
#                     await _auto_generate_reports(websocket, orchestrator, logger, reason="user-end")
#                     break

#                 # ----- NORMAL MESSAGE -----
#                 reply = await orchestrator.handle_user_message(text)
#                 await websocket.send_text(reply)

#         except WebSocketDisconnect:
#             logger.info("Client disconnected early.")
#             # Cannot send summary/soap here — WebSocket is dead.
#             orchestrator.reset()

#         except Exception as e:
#             logger.error(f"WebSocket error: {e}", exc_info=True)
#             try:
#                 await websocket.send_text("Internal error occurred.")
#             except:
#                 pass

#         finally:
#             try:
#                 await websocket.close()
#             except:
#                 pass
#             logger.info("/ws/medical WebSocket closed.")



# async def _auto_generate_reports(websocket, orchestrator, logger, reason: str):

#     """
#     SAFE FUNCTION:
#     Called BEFORE websocket closes.
#     Generates and sends summary + SOAP.
#     """

#     try:
#         await websocket.send_text(f"Conversation ended ({reason}). Generating report...")

#         summary = await orchestrator.generate_summary()
#         await websocket.send_text("[SUMMARY]\n" + summary)

#         soap = await orchestrator.generate_soap()
#         await websocket.send_text("[SOAP NOTE]\n" + soap)

#         logger.info("Auto summary + SOAP generated.")

#     except Exception as e:
#         logger.error(f"Report generation failed: {e}", exc_info=True)

#     orchestrator.reset()
#     await websocket.send_text("System ready for next patient.")

from fastapi import WebSocket, WebSocketDisconnect
from ...customagents.configloader import ConfigLoader
from ...customagents.factory import AgentFactory
from ...customagents import chatagent
from ...customagents import orechestrate
from datetime import datetime
import os
from ...customagents.util import SOAPReportGenerator
from ...customagents.util import PatientSummaryReportGenerator
from ...customagents.util import parse_soap_note 
from ...customagents.util import AssessmentPlanReportGenerator

from ...txtailogging.logger import get_logger
import importlib.resources as resources
from dotenv import load_dotenv
from .validator import MessageValidator
import mlflow
import asyncio


def register_medical_ws(app):

    load_dotenv(dotenv_path="txtai/src/python/txtai/.env")
    logger = get_logger("MedicalWS")

    # Correct config loading
    with resources.open_text(orechestrate, "med.yml") as f:
        config = ConfigLoader.load(f.name)

    mlflow.set_experiment("medical_orchestrator")

    orchestrator = AgentFactory.create_agent("medical_orchestrator", config)
    validator = MessageValidator()

    @app.websocket("/ws/medical")
    async def websocket_medical(websocket: WebSocket):

        await websocket.accept()
        logger.info("Client connected → /ws/medical")

        orchestrator.reset()

        greeting = orchestrator.chat_agent.get_initial_message()
        await websocket.send_text(greeting)

        try:
            while True:
                text = (await websocket.receive_text()).strip()

                if not validator.is_valid(text):
                    continue

                # User manually ends
                if text == "__end__":
                    await websocket.send_text("Conversation ended by user.")
                    await _generate_and_store_reports(orchestrator, logger)
                    break

                # Normal conversation
                reply = await orchestrator.handle_user_message(text)
                await websocket.send_text(reply)

        except WebSocketDisconnect:
            logger.info("Client disconnected. Auto-generating report…")

            # CLIENT DISCONNECTED → generate report on backend
            await _generate_and_store_reports(orchestrator, logger)

        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)

        finally:
            orchestrator.reset()
            logger.info("/ws/medical WebSocket closed.")
            try:
                await websocket.close()
            except:
                pass

async def _generate_and_store_reports(orchestrator, logger):

    try:
        # 1. Generate summary + SOAP text
        summary_text = await orchestrator.generate_summary()
        soap_text = await orchestrator.generate_soap()
        assessment_plan = await orchestrator.generate_assessment_plan()

        logger.info("Auto Summary Generated")
        logger.info("Auto SOAP Note Generated")

        # 2. Convert SOAP text into JSON sections
        soap_sections = parse_soap_note(soap_text)

        # 3. Add timestamp
        #soap_sections["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        soap_sections["date"] = now


        patient_info = orchestrator.chat_agent.config.get("patient_info", {})

        patient_id = patient_info.get('patient_id', 'unknown')

        # 4. Create PDF output directory
        os.makedirs("reports", exist_ok=True)

        store_patient_summary(
            patient_id=patient_id,
            summary=summary_text,
            soap=soap_sections,
            patient_info=patient_info,
            date=now
        )
        # 5. Generate PDF
        # ----------------------------------
        # PATIENT SUMMARY PDF
        # ----------------------------------
        patient_pdf = f"reports/{patient_id}_patient_summary.pdf"
        patient_pdf_gen = PatientSummaryReportGenerator(patient_info)
        patient_pdf_gen.generate(summary_text, patient_pdf)

        # DOCTOR SOAP PDF

        pdf_name = f"reports/{patient_id}_soap_report.pdf"
        generator = SOAPReportGenerator(patient_info)
        generator.generate(summary_data=soap_sections, output_file=pdf_name)

        assessment_pdf = f"reports/{patient_id}_assessment_plan.pdf"
        assessmentpdf = AssessmentPlanReportGenerator(patient_info)
        assessmentpdf.generate(
            assessment_plan,
            assessment_pdf
        )   
        logger.info(f"PDF SOAP Report saved: {pdf_name}")

    except Exception as e:
        logger.error(f"Report/PDF generation failed: {e}", exc_info=True)

    orchestrator.reset()

def store_patient_summary(patient_id: str, summary: str, soap: dict, patient_info: dict, date: str):
    """
    This function stores patient data WITHOUT hardcoding WHERE it is stored.

    You can later:
    - Save to DB
    - Save to Redis
    - Push to API
    - Write to file
    - Log only
    """
    print("\n========== PATIENT SUMMARY STORED ==========")
    print("Patient ID:", patient_id)
    print("Date:", date)
    print("Summary:", summary)
    print("SOAP:", soap)
    print("Patient Info:", patient_info)
    print("============================================\n")

    # If later you want DB:
    # db.save({ patient_id, summary, soap, patient_info, date })

    return True
