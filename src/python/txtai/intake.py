from txtai.customagents.patientintake.intakeagent import MedicalConversationAgent
from txtai.customagents.factory import AgentFactory
from txtai.customagents.configloader import ConfigLoader
import asyncio
config = ConfigLoader.load(r"D:/backend/txtai/src/python/txtai/customagents/patientintake/intake.yml")

agent = AgentFactory.create_agent("medical_intake", config)


async def run():
    print(agent.get_initial_message())

    while True:
        try:
            msg = input("Patient: ")

            # User exit (manual)
            if msg.lower().strip() in ["exit", "quit"]:
                print("\nConversation ended by user.\n")
                summary = await agent.auto_generate_summary()
                print("\n=== FINAL SOAP SUMMARY ===\n")
                print(summary)
                break

            # Normal generation
            response = await agent.generate_response(msg)
            print(response)

        except KeyboardInterrupt:
            print("\n\nSession interrupted.\nGenerating final summary...\n")
            summary = await agent.auto_generate_summary()
            print(summary)
            break

        except Exception:
            print("\nConnection or system error.\nGenerating final summary...\n")
            summary = await agent.auto_generate_summary()
            print(summary)
            break

asyncio.run(run())

# async def run():
#     while True:
#         msg = input("Patient: ")

#         if msg.lower().strip() in ["exit", "quit", "stop"]:
#             print("Conversation ended.\n")
#             break

#         response = await agent.generate_response(msg)
#         print(response)
#     print("\n=== SUMMARY ===")
#     # print(await agent.generate_summary())

# asyncio.run(run())