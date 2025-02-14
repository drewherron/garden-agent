-- Create the Plants table
CREATE TABLE Plants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    sun_requirement TEXT CHECK (sun_requirement IN ('Full sun', 'Partial sun', 'Shade')),
    soil_ph_min DECIMAL,
    soil_ph_max DECIMAL,
    spacing_cm DECIMAL,
    germination_days INT,
    maturity_days INT
);

-- Create the Logs table
CREATE TABLE Logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id INTEGER REFERENCES Plants(id) ON DELETE CASCADE,
    year INT NOT NULL,
    season TEXT CHECK (season IN ('Spring', 'Summer', 'Fall', 'Winter')),
    indoor BOOLEAN NOT NULL DEFAULT 0,
    planting_date DATE,
    transplant_date DATE,
    soil_ph DECIMAL,
    pests TEXT,
    fertilizer TEXT,
    amendments TEXT,
    source TEXT,
    notes TEXT
);

-- Create the Imports table
CREATE TABLE Imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_date TEXT NOT NULL,
    note_content TEXT NOT NULL,
    imported_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
