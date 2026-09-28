"""
dataplane/schema/apply_schema.py
================================

WHAT THIS FILE IS FOR
---------------------
Runs the four schema files, in order, against the mf_data database on
Amazon RDS:

    01_fund_house.sql    -> 9 tables
    02_market.sql        -> 6 tables
    03_investors.sql     -> 4 tables
    04_transactions.sql  -> 1 table, split into 60 monthly pieces

After it finishes, mf_data contains all 20 tables, empty and ready for the
data generator.

WHY A PYTHON SCRIPT, AND NOT psql
---------------------------------
psql would work, but it has to be installed on the machine, and the host,
user and password would have to be typed into the command. This script reads
one setting, DATA_DB_URL, from .env, which is already set up and already
tested. The same script then works unchanged on anyone else's machine.

WHY THE ORDER MATTERS
---------------------
Later files point to tables created by earlier ones. For example, file 02's
portfolio_holdings points to schemes, created in file 01. If file 02 ran
first, PostgreSQL would stop with "relation schemes does not exist". The
files are named 01_ to 04_, so sorting them by name gives the correct order.

IS IT SAFE TO RUN MORE THAN ONCE?
---------------------------------
Yes. Each SQL file starts by dropping its own tables, so every run rebuilds
the schema from empty. That also means every run DELETES any data already
loaded into mf_data.

    That is fine here, because mf_data holds synthetic data that this project
    generates in minutes. NEVER point this script at an RDS instance holding
    real data: with a client's own database, the whole dataplane/ folder is
    deleted and this script is never used.

BEFORE YOU RUN IT
-----------------
    python scripts/rds_setup.py      # creates agent_meta, switches on pgvector

HOW TO RUN
----------
    python dataplane/schema/apply_schema.py
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

# "os" reads settings (environment variables), here DATA_DB_URL.
import os

# "Path" handles file and folder paths in a way that works on any operating
# system. It is used to find the .sql files sitting next to this script.
from pathlib import Path

# "psycopg" is the PostgreSQL driver: it opens the connection to RDS and
# sends it SQL to run.
import psycopg

# Reads the .env file, so os.environ contains DATA_DB_URL.
from dotenv import load_dotenv

# __file__ is the path of THIS script, and .parent is the folder it sits in:
# dataplane/schema/. The four .sql files are in that same folder.
#
# Finding them relative to this script, rather than relative to whichever
# folder the terminal happens to be in, means the script works from anywhere.
SCHEMA_FOLDER = Path(__file__).resolve().parent

# How many SQL files this script expects to find. If the count is different,
# a file is missing or an extra one has appeared, and it stops rather than
# building half a schema.
EXPECTED_FILES = 4


def main() -> None:
    """Run every numbered .sql file in this folder, in name order."""

    # Load DATA_DB_URL (and everything else) from .env.
    load_dotenv()

    # The address of the mf_data database on RDS, for example:
    #   postgresql://fundgraph:secret@fundgraph-db.abc123.us-east-1.rds.amazonaws.com:5432/mf_data
    # Square brackets: if the setting is missing, Python stops here with a
    # clear error naming DATA_DB_URL.
    database_url = os.environ["DATA_DB_URL"]

    # Find the SQL files.
    #   glob("0*.sql") -> every file whose name starts with "0" and ends in
    #                     ".sql", which is 01_... to 04_...
    #   sorted(...)    -> put them in name order, which is the run order.
    sql_files = sorted(SCHEMA_FOLDER.glob("0*.sql"))
    if len(sql_files) != EXPECTED_FILES:
        raise SystemExit(
            f"Expected {EXPECTED_FILES} schema files in {SCHEMA_FOLDER}, found {len(sql_files)}."
        )

    # Open ONE connection and run all four files through it.
    #
    # "with ... as conn" means: if the block below finishes without an error,
    # psycopg COMMITS (saves) everything. If any file fails, it ROLLS BACK
    # (undoes) everything, and the database is left exactly as it was. There
    # is no way to end up with half a schema.
    #
    # connect_timeout=15: if RDS cannot be reached (still starting, wrong
    # security group, wrong host), fail in 15 seconds with a clear message
    # instead of waiting a long time.
    with psycopg.connect(database_url, connect_timeout=15) as conn:
        for sql_file in sql_files:
            # Show progress, so a slow or failing file is easy to spot.
            print(f"Running {sql_file.name} ...")

            # read_text() loads the whole file as one string of SQL.
            sql_text = sql_file.read_text()

            # Send the whole file to PostgreSQL in one go. psycopg allows
            # several statements in one execute() call as long as no
            # parameters are passed, which is the case here.
            conn.execute(sql_text)

    # Reaching this line means the "with" block committed successfully.
    print("Schema applied: all 20 tables created in mf_data.")
    print("Next: python -m dataplane.generator.run")


# This runs only when the file is run directly, not when it is imported.
if __name__ == "__main__":
    main()