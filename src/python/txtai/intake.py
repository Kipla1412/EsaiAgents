from txtai.customagents.patientintake import MedicalIntakeAgent
from txtai.customagents.factory import AgentFactory
from txtai.customagents.configloader import ConfigLoader
import asyncio
config = ConfigLoader.load(r"D:/backend/txtai/src/python/txtai/customagents/patientintake/intake.yml")

agent = AgentFactory.create_agent("medical_intake", config)

print(agent.get_initial_message())

async def run():
    while True:
        user_input = input("Patient: ")

        # Stop convo
        if user_input.lower() in ["exit", "quit", "stop"]:
            print("\nConversation ended.\n")
            break

        # Generate agent response
        response = await agent.generate_response(user_input)
        print(f"Agent: {response}")

    # After exit, generate JSON + summary
    print("\n===JSON OUTPUT ==")
    print(await agent.extract_structured_json())

    print("\n==PRE-VISIT SUMMARY ==")
    print(await agent.build_previsit_summary())

asyncio.run(run())