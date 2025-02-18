import re
import math
import difflib
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
    notes = []

    # Regex to detect a heading line: one or more '*' + space + text
    heading_regex = re.compile(r'^(\*+)\s+(.*)$')
    created_regex = re.compile(r'^:CREATED:\s*\[(.*)\]\s*$')

    current_title = None
    current_created = ""
    collecting_content = False
    content_lines = []

    with open(org_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')

            # 1. Check if this line is a heading
            heading_match = heading_regex.match(line)
            if heading_match:
                # If we already have a note in progress, store it before starting a new one
                if current_title is not None:
                    # Combine collected content into a single string
                    current_content = "\n".join(content_lines).strip()
                    notes.append({
                        "title": current_title.strip(),
                        "created_date": current_created.strip(),
                        "content": current_content
                    })

                # Start a new note
                current_title = heading_match.group(2)
                current_created = ""
                content_lines = []
                collecting_content = False
                continue

            # 2. If we're inside a note (we have a current_title)
            if current_title is not None:
                # Check if we hit the start of a PROPERTIES block
                if line.strip().lower() == ":properties:":
                    collecting_content = False  # We stop collecting content in favor of properties
                    continue

                # Check if we hit the end of the PROPERTIES block
                if line.strip().lower() == ":end:":
                    collecting_content = True  # Resume collecting content after :END:
                    continue

                # If we're inside the PROPERTIES block, look for :CREATED:
                if not collecting_content:
                    m = created_regex.match(line)
                    if m:
                        current_created = m.group(1).strip()
                    # We ignore other properties for now
                    continue

                # Otherwise, if collecting_content is True, we add lines to content
                content_lines.append(line)

        # End of file: if there's a note in progress, store it
        if current_title is not None:
            current_content = "\n".join(content_lines).strip()
            notes.append({
                "title": current_title.strip(),
                "created_date": current_created.strip(),
                "content": current_content
            })

    return notes

def fuzzy_match_plants(
    note_title: str,
    plants_data: List[Dict[str, Any]],
    cutoff: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Compare 'note_title' against the 'name' field of each plant in plants_data.
    Return a sorted list (descending similarity ratio) of all plants whose name
    is at least 'cutoff' similar to 'note_title'.
    """

    # Store tuples: (similarity_ratio, plant_dict)
    scored_plants = []

    for plant in plants_data:
        plant_name = plant["name"]
        ratio = difflib.SequenceMatcher(None, note_title.lower(), plant_name.lower()).ratio()
        if ratio >= cutoff:
            scored_plants.append((ratio, plant))

    # Sort descending by ratio
    scored_plants.sort(key=lambda x: x[0], reverse=True)

    # Extract just the plant dicts in sorted order
    matched_plants = [p[1] for p in scored_plants]

    return matched_plants

def note_already_imported(
    conn: sqlite3.Connection,
    title: str,
    created_date: str,
    content: str
) -> bool:
    """
    Checks if this exact note (title, created_date, content)
    is already in the Imports table.
    Returns True if it's already imported, otherwise False.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) FROM Imports
        WHERE title = ? AND created_date = ? AND note_content = ?
        """,
        (title, created_date, content)
    )
    (count,) = cursor.fetchone()
    return count > 0

def insert_into_imports(
    conn: sqlite3.Connection,
    title: str,
    created_date: str,
    content: str
) -> None:
    """
    Inserts a record into the Imports table to mark this note as processed.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO Imports (title, created_date, note_content)
        VALUES (?, ?, ?)
        """,
        (title, created_date, content)
    )
    conn.commit()

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

def get_plantings_for_plants(
    conn: sqlite3.Connection,
    plant_ids: List[int],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Returns up to 'limit' of the most recent (by descending id) Plantings rows
    for all plants in the given list of 'plant_ids'.

    Example:
        if plant_ids=[1,2,3], we do:
            SELECT ... FROM Plantings
            WHERE plant_id IN (1,2,3)
            ORDER BY id DESC
            LIMIT ?

        and return at most 'limit' rows.
    """
    if not plant_ids:
        return []  # No plants to look up, return empty list

    # Build the dynamic placeholders for the IN clause
    placeholders = ", ".join(["?"] * len(plant_ids))

    query = f"""
        SELECT
            id,
            plant_id,
            year,
            season,
            batch_number,
            seed_start_date,
            transplant_date,
            indoor,
            soil_ph,
            pests,
            disease,
            fertilizer,
            amendments,
            source,
            notes
        FROM Plantings
        WHERE plant_id IN ({placeholders})
        ORDER BY id DESC
        LIMIT ?
    """

    params = list(plant_ids) + [limit]

    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()

    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "plant_id": row[1],
            "year": row[2],
            "season": row[3],
            "batch_number": row[4],
            "seed_start_date": row[5],
            "transplant_date": row[6],
            "indoor": bool(row[7]),
            "soil_ph": row[8],
            "pests": row[9],
            "disease": row[10],
            "fertilizer": row[11],
            "amendments": row[12],
            "source": row[13],
            "notes": row[14]
        })
    return results

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

    conn = sqlite3.connect("garden.db")

    llm = None  # TODO

    # Load these tables before the loop
    plants_data = get_all_plants(conn)
    seasons_data = get_seasons_data(conn)

    # Parse the org file
    notes = parse_org_file("garden-log.org")

    # Process each note
    for note in notes:
        t = note["title"]
        d = note["created_date"]
        c = note["content"]

        # Check if already imported
        if note_already_imported(conn, t, d, c):
            continue

        # Fuzzy-search locally to narrow plant candidates
        candidate_plants = fuzzy_match_plants(t, plants_data)

        # If no candidates returned, we can skip or let the LLM handle it
        if not candidate_plants:
            print(f"No local fuzzy match for '{t}'; skipping or handle in LLM...")
            continue

        # Collect all candidate plant IDs:
        candidate_ids = [cp["id"] for cp in candidate_plants]

        # Now get the 10 most recent rows among all those plants:
        plantings_data = get_plantings_for_plants(conn, candidate_ids, limit=10)

        # LLM: finalize the decision
        decision = call_llm_for_decision(
            note_title=t,
            note_content=c,
            note_date=d,
            plants_data=candidate_plants,
            plantings_data=plantings_data,
            seasons_data=seasons_data
        )

        if "error" in decision:
            # Skip if ambiguous or unresolvable
            print(f"Skipping note due to error: {decision['error']}")
            continue

        # Apply the update
        apply_planting_update(conn, decision)

        # Mark imported
        insert_into_imports(conn, t, d, c)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
