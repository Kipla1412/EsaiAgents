#from txtai.app import Application
from txtai.agent import Agent
from txtai.customagents.phiaagent.duckdbtool import DuckDBSQLTool
from txtai.customagents.phiaagent.build_prompt import build_system_prompt
from txtai.agent.tool import EmbeddingsTool
from txtai.customagents.phiaagent.final_tool import FinalAnswerTool
import os
from dotenv import load_dotenv

load_dotenv()

question ="What was my average resting heart rate in the past two weeks?"

summary_path=  r"D:\backend\txtai\src\python\txtai\datas\activities.csv"
activities_path = r"D:\backend\txtai\src\python\txtai\datas\summary.csv"

sqltool = DuckDBSQLTool(summary_path, activities_path)
embedtool = EmbeddingsTool({
    "name": "memory",
    "description": "Search PHIA reasoning examples",
    "path":"D:\\backend\\txtai\\src\\python\\fewshots_index",
    "content": True
})
finaltool =FinalAnswerTool()
schema_text = sqltool.get_schema()
print(schema_text)

fewshot = embedtool.forward(question)
# best_example = fewshot[0]["text"] if fewshot else None
if fewshot:
    best_example = fewshot[0][1]   # <-- index 1 = text
else:
    best_example = None

system_prompt = build_system_prompt(schema_text, fewshot_example=best_example)
print(system_prompt)

agent = Agent(
    model={
        "method": "litellm",
        "path": "openrouter/x-ai/grok-4.1-fast:free",
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "api_base": "https://openrouter.ai/api/v1",
        # "system": system_prompt
    },
    description="Health SQL Reasoning Agent",
    tools=[sqltool,embedtool,finaltool],
    max_steps=10,
    #system=system_prompt
    method="tool" 
)
agent.process.model.llm.system = system_prompt
#agent.process.model.system(system_prompt)
#question = "how to improve me sleeping time?" #"What was my average resting heart rate in the past two weeks?"#"How has my sleep duration changed over the last 10 days?" #"Does my deep sleep correlate with my resting heart rate?" #"What is my average resting_heart_rate  in the last 5 days?"
#

print("\nUser:", question)
response = agent(question)
print("\nAgent:", response)

