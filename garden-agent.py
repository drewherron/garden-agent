import re
import json
import math
import openai
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
    Returns up to 'limit' of the most recent (by descending P.id) Plantings rows
    for all plants in the given list of 'plant_ids', along with 'expected_last_frost'
    and 'actual_last_frost' from Seasons (joined on matching year).
    """
    if not plant_ids:
        # No IDs => no rows
        return []

    # Build placeholders for the IN clause: e.g. "?, ?, ?"
    placeholders = ", ".join(["?"] * len(plant_ids))

    query = f"""
        SELECT
            P.id,
            P.plant_id,
            P.year,
            P.season,
            P.batch_number,
            P.seed_start_date,
            P.transplant_date,
            P.indoor,
            P.soil_ph,
            P.pests,
            P.disease,
            P.fertilizer,
            P.amendments,
            P.source,
            P.notes,
            S.expected_last_frost,
            S.actual_last_frost
        FROM Plantings AS P
        LEFT JOIN Seasons AS S
            ON P.year = S.year
        WHERE P.plant_id IN ({placeholders})
        ORDER BY P.id DESC
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
            "notes": row[14],
            "expected_last_frost": row[15],
            "actual_last_frost": row[16]
        })
    return results

def call_llm_for_decision(
    note_title: str,
    note_content: str,
    note_date: str,
    plants_data: List[Dict[str, Any]],
    plantings_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calls the LLM with the note + context data. Returns a structured dict about what to do.
    Format:
    {
      "action": "update" or "create",
      "plant_id": 123,
      "planting_id": 7,           # if updating
      "year": 2025,
      "season": "Spring",
      "batch_number": 2,
      "fields_to_update": {
        "notes_append": "2025-07-16 Saw aphids today",
        "pests_append": ["aphids"],
        "disease_append": [],
        "fertilizer_append": "",
        "seed_start_date": "2025-02-14",   # or None if not relevant
        ...
      }
    }
    or
    {
      "error": "Ambiguous. Multiple tomato matches..."
    }
    """

    # --- 1) Build prompt context ---

    # Convert the candidate plants and plantings data to JSON strings
    plants_json = json.dumps(plants_data, indent=2)
    plantings_json = json.dumps(plantings_data, indent=2)

    # Instruction for how we want the model to respond
    instruction = f"""
You are a gardening data assistant. The user has created a note with the following details:

Title: {note_title}
Created date: {note_date}
Content: {note_content}

We have the following candidate plants. Each item has fields like "id" and "name":
{plants_json}

We also have the newest Plantings rows for these plants. Each Plantings row might look like:
{plantings_json}

Your job:
1. Determine which plant this note refers to.
2. Decide if we should update an existing Plantings row or create a new one.
3. If we update, specify which row (planting_id).
4. If we create a new row, specify a new planting with (year, season, batch_number, etc.).
5. Indicate what fields to update/append. For instance:
   - Append to notes with a date-stamped line (like "2025-07-16 Saw aphids...")
   - If user mentions pests, append to pests list or set pests if none.
   - If user mentions fertilizer, append to fertilizer, etc.

If there's any ambiguity (e.g., multiple equally likely plants or no suitable plantings), return a JSON object with an "error" key describing the issue:
{{
  "error": "some description"
}}

Otherwise return valid JSON in this format (with keys from the original rows and NO extra keys!):
{{
  "action": "update" or "create",
  "plant_id": <number>,
  "planting_id": <number or null if create>,
  "year": <number>,
  "season": "<Spring|Summer|Fall|Winter|Unknown>",
  "batch_number": <number>,
  "fields_to_update": {{
      "notes_append": "<string to append>",
      "pests_append": ["<string>", ...],
      "disease_append": ["<string>", ...],
      "fertilizer_append": "<string>",
      "seed_start_date": "<YYYY-MM-DD or null>",
      "transplant_date": "<YYYY-MM-DD or null>",
      ...
  }}
}}

DO NOT include markdown formatting or extra text. Only return a single JSON object.
Make sure it's valid JSON that can be parsed by Python's json.loads.
"""

    # --- 2) Call OpenAI Chat Completion ---
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a helpful gardening assistant."},
                {"role": "user", "content": instruction}
            ],
        )

        llm_output = response["choices"][0]["message"]["content"].strip()

        # --- 3) Attempt to parse JSON ---
        decision = json.loads(llm_output)

        # We expect either {"error": "..."} or the detailed schema
        if "error" in decision:
            # Return error structure directly
            return {"error": decision["error"]}
        else:
            # Return the full decision if it has the required fields
            if not all(k in decision for k in ["action", "plant_id", "year", "season", "batch_number", "fields_to_update"]):
                # If it's missing keys, let's treat it as an error
                return {"error": "LLM response missing required keys."}
            # Assuming it's valid... maybe check?
            return decision

    except json.JSONDecodeError:
        return {"error": "LLM returned invalid JSON."}
    except Exception as e:
        return {"error": f"OpenAI API error: {str(e)}"}

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

    # Load Plants table before the loop
    plants_data = get_all_plants(conn)

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
            print(f"No local fuzzy match for '{t}' - skipping.")
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
        )

        if "error" in decision:
            # LLM gave us an error
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
