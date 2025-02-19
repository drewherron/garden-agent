# garden-agent

This is a personal project to simplify and (partially) automate keeping a gardening journal. The program reads an org-mode file, then uses an LLM agent to update a SQLite database based on the contents of that file.

Throughout development, I will try to keep this code generalizable and reusable for other projects. The program should be adaptable to any scenario where we want an LLM to decide how to update a SQL database based on some structured natural language input.

## How It Works

I use the Orgzly app on my phone to quickly add a top-level note to an org file. The program reads this file, extracts structured information, and then determines whether to update or create records in the database. Here’s an example of the formatting:

```
* Purple Cherokee Tomatoes
:PROPERTIES:
:CREATED:  [2024-05-01 Wed 18:30]
:END:
  
Transplanted to garden today
```

Each note in the org file is parsed, and a fuzzy matching algorithm is used to check if the note refers to an existing plant. If no match is found, the note is assumed to describe a new plant. The program then queries the LLM to decide how to process the note, ensuring correct updates based on context.

## Database

### Plants
- `id INTEGER` - A unique ID for each plant
- `name TEXT` - The name of the plant
- `sun_requirement TEXT` - Sun exposure requirement (Full sun, Partial sun, Shade)
- `soil_ph_min DECIMAL` - Minimum soil pH
- `soil_ph_max DECIMAL` - Maximum soil pH
- `spacing_cm DECIMAL` - Recommended spacing between plants (cm)
- `germination_days INT` - Days required for seeds to germinate
- `maturity_days INT` - Days until plant reaches maturity
- `start_weeks_before_lf DECIMAL` - Weeks before the last frost to start seeds indoors

### Plantings
- `id INTEGER` - A unique ID for each planting event
- `plant_id INTEGER` - The plant associated with this planting (references `Plants.id`)
- `year INT` - The year of the planting event
- `season TEXT` - Season of planting (Winter, Spring, Summer, Fall, Unknown)
- `batch_number INT` - Batch identifier for multiple plantings of the same plant
- `seed_start_date DATE` - Date when seeds were started (if applicable)
- `transplant_date DATE` - Date when plants were transplanted
- `indoor BOOLEAN` - Whether the planting was indoors (`0` for no, `1` for yes)
- `soil_ph DECIMAL` - Measured soil pH
- `pests TEXT` - List of pests encountered
- `disease TEXT` - List of diseases affecting the plant
- `fertilizer TEXT` - Fertilizers used
- `amendments TEXT` - Soil amendments applied
- `source TEXT` - Source of seeds or plants
- `notes TEXT` - Additional notes on the planting

### Seasons
- `id INTEGER` - A unique ID for each season entry
- `year INT` - The year for this season entry
- `expected_last_frost DATE` - Expected last frost date for the year
- `actual_last_frost DATE` - Actual last frost date for the year

### Imports
- `id INTEGER` - A unique ID for each imported note
- `title TEXT` - Title of the note from the org file
- `created_date TEXT` - Date the note was created
- `note_content TEXT` - Full content of the note
- `imported_at DATETIME` - Timestamp of when the note was processed

## Usage

To create the database:

```
sqlite3 garden.db < create_garden_db.sql
```

Then run the Python program manually as needed, or on a schedule:

```
python garden_agent.py
```

## Example Scenarios

### 1. Adding a New Plant

#### Input (org file note):

```
* Roma Tomato
:PROPERTIES:
:CREATED:  [2024-03-15 Fri 12:00]
:END:
  
Full sun. 6.0-7.0 pH. Spacing 30 cm. Germination 7 days. Maturity 70 days. Start indoors 6 weeks before last frost.
```

#### LLM Decision:

```
{
  "action": "create_plant",
  "plant_name": "Roma Tomato",
  "sun_requirement": "Full sun",
  "soil_ph_min": 6.0,
  "soil_ph_max": 7.0,
  "spacing_cm": 30,
  "germination_days": 7,
  "maturity_days": 70,
  "start_weeks_before_lf": 6.0
}
```

#### Database Update:
A new row is added to the `Plants` table. If the LLM decides that the note is adding a new plant (and it will help to be clear about this in the note), ChatGPT should have enough general knowledge to be able to fill in blanks. A note like this should work fine:

```
* Roma Tomato
:PROPERTIES:
:CREATED:  [2024-03-15 Fri 12:00]
:END:
  
Adding a new plant, needs full sun, not sure about the rest. Could you fill out the other fields?
```

---

### 2. Adding a New Planting

#### Input (org file note):

```
* Roma Tomato Batch #2
:PROPERTIES:
:CREATED:  [2024-04-01 Mon 10:30]
:END:
```
  
Started indoors today. Using organic compost.

#### LLM Decision:

```
{
  "action": "create",
  "plant_id": 1,
  "year": 2024,
  "season": "Spring",
  "batch_number": 2,
  "fields_to_update": {
    "seed_start_date": "2024-04-01",
    "fertilizer_append": "Organic",
    "notes_append": "Using organic compost."
  }
}
```

#### Database Update:
A new row is added to the `Plantings` table, linking this planting to the existing "Roma Tomato" plant.

---

### 3. Updating an Existing Planting

#### Input (org file note):

```
* Tomatoes
:PROPERTIES:
:CREATED:  [2024-04-20 Sat 15:45]
:END:
  
Transplanted to raised bed. Added bone meal.
```

#### Existing Plantings in Database:

| ID | Plant ID | Year | Season | Batch | Seed Start | Transplant Date | Fertilizer | Notes |
|----|---------|------|--------|-------|------------|-----------------|-----------|------------------|
| 5  | 1       | 2024 | Spring | 2     | 2024-04-01 | NULL            | Organic | Using organic compost. |

#### LLM Decision:

There may be many types of tomatoes in the `Plants` table, but the LLM can handle the ambiguity. The LLM checks existing plantings and sees that currently only one type of tomato is being grown, so it chooses the correct `plant_id`. This is clearly an update to an existing planting, if no currently growing tomatoes were found in the `Plantings` table, the LLM should return an error. There should also be an error if multiple batches are growing but not yet transplanted. In that case you'd want to specify batch number in the note. If batch 1 has already been transplanted, the LLM should be able to correctly select batch 2.

```
{
  "action": "update",
  "plant_id": 1,
  "planting_id": 5,
  "fields_to_update": {
    "transplant_date": "2024-04-20",
    "amendments_append": "Bone meal",
    "notes_append": "Transplanted to raised bed. Added bone meal."
  }
}
```

#### Database Update:
The three relevant fields are updated in the existing `Plantings` row.

---

This project is a work in progress, and contributions or ideas for improvement are **very** welcome.
