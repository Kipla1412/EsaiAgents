import duckdb
import pandas as pd
from txtai.customagents.phiaagent.schemabuilder import SchemaManager

con = duckdb.connect()

summary_df =pd.read_csv(r"D:\backend\txtai\src\python\txtai\datas\summary.csv")
activities_df =pd.read_csv(r"D:\backend\txtai\src\python\txtai\datas\activities.csv")
con.register("summary", summary_df)       # Contains datetime, HRV, sleep, steps
con.register("activities", activities_df)
schema_tool = SchemaManager(con)

print("\n=== RAW SCHEMA DICT ===")
print(schema_tool.get_schema())

print("\n=== FORMATTED SCHEMA PROMPT ===")
print(schema_tool.format_schema_prompt())

print(con.execute("SELECT * FROM summary LIMIT 1").fetchall())
print(con.execute("SELECT * FROM activities LIMIT 1").fetchall())
