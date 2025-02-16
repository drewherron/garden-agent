import sqlite3
from typing import List, Optional
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
    """Returns a list of all plants."""
    pass

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
