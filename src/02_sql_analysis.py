"""Step 2 - Load the clean table into SQLite and run every query in sql/.

Each query's result is saved to outputs/tables/<query_name>.csv and printed.
SQLite ships with Python, so no database server is needed; the SQL is standard
(CTEs + window functions) and ports to SQL Server, PostgreSQL or DuckDB with minor edits.
"""
import sqlite3

import pandas as pd

import config as C

# Narrative text is not needed for SQL analysis; keeping it out keeps the database small.
DROP = ["narrative", "narrative_clean"]


def main() -> None:
    df = pd.read_csv(C.CLEAN).drop(columns=DROP)
    df["date_received"] = pd.to_datetime(df["date_received"]).dt.strftime("%Y-%m-%d")

    C.DB.unlink(missing_ok=True)
    with sqlite3.connect(C.DB) as con:
        df.to_sql("complaints", con, index=False)
        for col in ["product", "company_group", "month", "state"]:
            con.execute(f"CREATE INDEX idx_{col} ON complaints({col})")

        C.TABLES.mkdir(parents=True, exist_ok=True)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_columns", 20)
        for path in sorted(C.SQL_DIR.glob("*.sql")):
            result = pd.read_sql_query(path.read_text(), con)
            result.to_csv(C.TABLES / f"{path.stem}.csv", index=False)
            print(f"\n=== {path.name} ({len(result)} rows) ===")
            print(result.to_string(index=False))


if __name__ == "__main__":
    main()
