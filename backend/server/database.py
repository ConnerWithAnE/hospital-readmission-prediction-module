import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "inventory.db"


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS drugs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ndc_code TEXT UNIQUE NOT NULL,
            generic_name TEXT NOT NULL,
            brand_name TEXT,
            dosage_form TEXT,
            strength TEXT,
            model_field TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_drugs_model_field ON drugs(model_field);
        CREATE INDEX IF NOT EXISTS idx_drugs_generic_name ON drugs(generic_name);

        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_id INTEGER NOT NULL REFERENCES drugs(id) ON DELETE CASCADE,
            quantity_on_hand INTEGER NOT NULL DEFAULT 0,
            reorder_level INTEGER NOT NULL DEFAULT 10,
            unit TEXT NOT NULL DEFAULT 'units',
            last_updated TEXT DEFAULT (datetime('now')),
            notes TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_inventory_drug ON inventory(drug_id);

        CREATE TABLE IF NOT EXISTS inventory_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_id INTEGER NOT NULL REFERENCES drugs(id) ON DELETE CASCADE,
            change_amount INTEGER NOT NULL,
            reason TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.close()