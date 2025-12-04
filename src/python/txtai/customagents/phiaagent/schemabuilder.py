import duckdb

class SchemaManager:
    def __init__(self, db):

        self.db =db

    def get_schema(self):
        """
        Extracts full schema metadata from DuckDB and formats it cleanly
        so an LLM can understand it and build SQL queries correctly.
        """
        tables = self.db.execute("SHOW TABLES").fetchall()
        schema = {}

        for (table,) in tables:
            columns = self.db.execute(f"DESCRIBE {table}").fetchall()
            schema[table] = {col[0]: col[1] for col in columns}

        return schema
    
    def format_schema_prompt(self):

        schema = self.get_schema()

        formatted ="Available Tables and Columns:\n"
        for table, cols in schema.items():
            formatted += f"\nTABLE: {table}\n"
            for name, dtype in cols.items():
                formatted += f"  - {name} ({dtype})\n"

        return formatted
