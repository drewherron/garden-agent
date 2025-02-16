# garden-agent

This is a personal project to simplify and (partially) automate keeping a gardening journal. The program reads an org-mode file, then uses an LLM agent to update a SQLite database based on the contents of that file.

Throughout development, I will try to keep this code generalizable/reusable for other projects. The program should be adaptable to any project where we want an LLM to decide on its own how to update a SQL database.

## How it works

I use the Orgzly app on my phone to quickly add a top-level note to an org file. Here's an example of the formatting:

```
* Purple Cherokee Tomatoes 
:PROPERTIES:
:CREATED:  [2024-05-01 Wed 18:30]
:END:

Transplanted to garden today
```

The database consists of three tables: `Plants`, `Logs`, and `Imports`. Our LLM agent will read the database for context, then look at the note and decide how to update the database. Every note added to the database will be recorded in the `Imports` table to prevent duplicate entries. Each plant will only have one `Logs` row per season. In the note example above, the agent will update the existing `Logs` row for this plant and season, adding the date of the note to the `transplanted` field of the database.

## Usage

To create the database:

```
sqlite3 garden.db < create_garden_db.sql
```

Then run the Python program manually as needed, or run it on a schedule.
