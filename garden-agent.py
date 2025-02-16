import sqlite3
from typing import List, Dict, Any, Optional
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

def parse_org_file(org_file_path: str) -> List[Dict[str, str]]:
    """
    Parses the org file and returns a list of notes, each with:
    {
        "title": str,
        "created_date": str,
        "content": str
    }
    """
    pass

def note_already_imported(
    conn: sqlite3.Connection,
    title: str,
    created_date: str,
    content: str
) -> bool:
    """Checks if this note is already in the Imports table."""
    pass

def insert_into_imports(
    conn: sqlite3.Connection,
    title: str,
    created_date: str,
    content: str
) -> None:
    """Marks this note as imported."""
    pass

def get_all_plants(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Returns all rows from the Plants table as a list of dicts.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            id,
            name,
            sun_requirement,
            soil_ph_min,
            soil_ph_max,
            spacing_cm,
            germination_days,
            maturity_days
        FROM Plants
    """)

    rows = cursor.fetchall()
    plants = []
    for row in rows:
        plants.append({
            "id": row[0],
            "name": row[1],
            "sun_requirement": row[2],
            "soil_ph_min": row[3],
            "soil_ph_max": row[4],
            "spacing_cm": row[5],
            "germination_days": row[6],
            "maturity_days": row[7]
        })
    return plants

def get_seasons_data(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Returns all rows from the Seasons table as a list of dicts.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            id,
            year,
            expected_last_frost,
            actual_last_frost
        FROM Seasons
    """)

    rows = cursor.fetchall()
    seasons = []
    for row in rows:
        seasons.append({
            "id": row[0],
            "year": row[1],
            "expected_last_frost": row[2],
            "actual_last_frost": row[3]
        })
    return seasons

def get_plantings_for_plant(conn: sqlite3.Connection, plant_id: int) -> List[Dict[str, Any]]:
    """
    Returns all plantings rows for a given plant_id,
    possibly filtered by year, season, other?
    """
    pass

def get_seasons_data(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Returns all rows from Seasons table."""
    pass

def call_llm_for_decision(
    note_title: str,
    note_content: str,
    note_date: str,
    plants_data: List[Dict[str, Any]],
    plantings_data: List[Dict[str, Any]],
    seasons_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calls the LLM with the note + context data. 
    Returns a structured dict about what to do. Example:
    {
      "plant_id": 2,
      "action": "update",  # or "create"
      "planting_id": 7,    # if updating
      "year": 2025,
      "season": "Spring",
      "batch_number": 1,
      "fields_to_update": {
        "pests": ["aphids"],
        "notes_append": "2025-07-16 Saw aphids today"
      }
    }
    or
    {
      "error": "Ambiguous: multiple tomato varieties found..."
    }
    """
    pass

def apply_planting_update(
    conn: sqlite3.Connection,
    decision: Dict[str, Any]
) -> None:
    """
    Reads the LLM decision dict, then either updates an existing Plantings row
    or creates a new one. For example:
      - If decision["action"] == "update", run an UPDATE statement.
      - If it == "create", INSERT a new row with the specified columns.
    """
    pass

def main():
    pass

if __name__ == "__main__":
    main()
