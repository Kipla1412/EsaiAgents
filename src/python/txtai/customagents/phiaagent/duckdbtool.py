
from smolagents import Tool
import duckdb
import pandas as pd
from .schemabuilder import SchemaManager
from typing import Dict, Any


class DuckDBSQLTool(Tool):
 
    name = "sql"
    description = "Execute DuckDB SQL queries on health datasets (summary, activities)."

    inputs = {
        "query": {
            "type": "string",
            "description": "A valid SQL query string using DuckDB syntax."
        }
    }

    output_type = "string"

    def __init__(self, summary_path: str, activities_path: str):
        super().__init__()

        self.db = duckdb.connect()

        summary = pd.read_csv(summary_path, parse_dates=True)
        activities = pd.read_csv(activities_path, parse_dates= True)

        self.db.register("summary", summary)
        self.db.register("activities", activities)

        self.schema = SchemaManager(self.db)

    def get_schema(self) -> str:
        return self.schema.format_schema_prompt()

    def forward(self, query: str) -> str:
        """
        Runs the given DuckDB SQL query and returns the result as a string.
        """
        try:
            #result = self.db.execute(query).fetchall()
            result = self.db.execute(query).df()
            return str(result)
        except Exception as e:
            return f"SQL ERROR: {e}"


# ===============================================================
#  Wrapper function (same pattern as weather_tool)
# ===============================================================

# def duckdb_sql_tool(query: str,
#                     summary_path: str,
#                     activities_path: str) -> str:
#     """
#     Execute a DuckDB SQL query against health data.

#     Args:
#         query (str): SQL query string.
#         summary_path (str): Path to summary CSV file.
#         activities_path (str): Path to activities CSV file.

#     Returns:
#         str: DuckDB SQL execution result.
#     """

#     tool = DuckDBSQLTool(summary_path, activities_path)
#     return tool.forward(query)
