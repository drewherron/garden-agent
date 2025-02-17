DROP TABLE IF EXISTS Imports;
DROP TABLE IF EXISTS Plantings;
DROP TABLE IF EXISTS Seasons;
DROP TABLE IF EXISTS Plants;

-- 1. Plants
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

-- 2. Plantings
CREATE TABLE Plantings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plant_id INTEGER NOT NULL REFERENCES Plants(id) ON DELETE CASCADE,
    year INT NOT NULL,
    season TEXT NOT NULL CHECK (
        season IN ('Winter', 'Spring', 'Summer', 'Fall', 'Unknown')
    ) DEFAULT 'Unknown',
    batch_number INT NOT NULL DEFAULT 1,
    seed_start_date DATE,    -- If this is null, we assume not from seed.
    transplant_date DATE,
    indoor BOOLEAN NOT NULL DEFAULT 0,
    soil_ph DECIMAL,
    pests TEXT,
    disease TEXT,
    fertilizer TEXT,
    amendments TEXT,
    source TEXT,
    notes TEXT
);

-- 3. Seasons
CREATE TABLE Seasons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INT NOT NULL,
    expected_last_frost DATE,
    actual_last_frost DATE
);

-- 4. Imports
CREATE TABLE Imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_date TEXT NOT NULL,
    note_content TEXT NOT NULL,
    imported_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

