# Medication Inventory Management Dashboard — Implementation Plan

## Overview

Add a medication inventory tracking system that integrates with the existing readmission prediction model. Uses the FDA's NDC (National Drug Code) directory as the drug catalog and SQLite for local storage. The key value-add is cross-referencing patient medication needs against current hospital stock to surface supply gaps for high-risk patients.

---

## Architecture

### Database (SQLite — no external dependencies)

**File:** `backend/data/inventory.db` (auto-created at startup)

Three tables:

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `drugs` | Drug catalog (seeded from FDA NDC API) | `ndc_code`, `generic_name`, `brand_name`, `dosage_form`, `strength`, `model_field` |
| `inventory` | Current stock per drug | `drug_id`, `quantity_on_hand`, `reorder_level`, `unit` |
| `inventory_log` | Audit trail of stock changes | `drug_id`, `change_amount`, `reason`, `timestamp` |

The `model_field` column is the bridge between inventory and the prediction model — it maps an NDC drug entry to one of the 23 medication field names used by the model (e.g., `"metformin"`, `"insulin"`, `"glyburide-metformin"`). This enables the supply gap cross-reference.

### NDC Data Seeding

- The FDA NDC API (`https://api.fda.gov/drug/ndc.json`) is free, requires no API key, and supports search by generic name
- A "Seed from NDC" button triggers queries for all 23 diabetes medications in `data_maps.py:med_columns`
- Results are stored locally in SQLite — works offline after initial seed
- Users can also manually add drugs not found via the API

---

## Backend Changes

### New Files

| File | Purpose |
|------|---------|
| `backend/server/database.py` | SQLite connection management, schema creation (`init_db()`), helper queries |
| `backend/server/inventory_models.py` | Pydantic models: `DrugOut`, `InventoryItem`, `InventoryUpdate`, `AddInventoryItem`, `SupplyGap` |
| `backend/server/inventory.py` | FastAPI `APIRouter` with all inventory endpoints |

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/inventory` | List all inventory items with drug info. Supports `?low_stock=true` filter |
| `GET` | `/api/inventory/{drug_id}` | Single item + transaction history |
| `POST` | `/api/inventory` | Add a new drug + initial stock |
| `PATCH` | `/api/inventory/{drug_id}` | Update stock (restock/dispense). Creates audit log entry |
| `DELETE` | `/api/inventory/{drug_id}` | Remove a drug from inventory tracking |
| `GET` | `/api/inventory/supply-gaps` | Drugs mapped to model fields AND below reorder level |
| `GET` | `/api/drugs/search?q=metformin` | Search drug catalog by name or NDC code |
| `POST` | `/api/drugs/seed` | Fetch diabetes meds from FDA NDC API and populate catalog |

### Modified Files

**`backend/server/main.py`** — 3 lines added:
```python
from .inventory import router as inventory_router
from .database import init_db
init_db()
app.include_router(inventory_router)
```

### Dependencies

No new Python packages needed:
- `sqlite3` — standard library
- `urllib.request` — standard library (for FDA API calls)

---

## Frontend Changes

### New Page: Medication Inventory (`/inventory`)

**Layout (top to bottom):**

1. **Supply Gap Alert Banner** — Only visible when model-mapped drugs are low/out of stock
   - Red cards for zero stock, amber for below reorder level
   - Shows which model medication field each drug maps to
   - Answers the question: "Are we low on any medications that affect readmission risk?"

2. **Summary Cards Row** — Three cards:
   - Total drugs tracked
   - Items below reorder level
   - Items at zero stock

3. **Inventory Table** — Plain HTML table (same pattern as scenario comparison page)
   - Columns: NDC Code, Generic Name, Brand Name, Dosage/Strength, Model Field (badge), Quantity (color-coded green/amber/red), Reorder Level, Unit, Actions
   - Client-side text search/filter at top
   - Per-row action buttons: Restock, Dispense

4. **Action Bar** — "Add Drug" and "Seed from NDC" buttons

### New Files

| File | Purpose |
|------|---------|
| `frontend/src/pages/medication-inventory.tsx` | Main inventory dashboard page |
| `frontend/src/components/inventory/stock-update-dialog.tsx` | Dialog for restock/dispense (amount + reason) |
| `frontend/src/components/inventory/add-drug-dialog.tsx` | Dialog to add a new drug with model_field dropdown |

### Modified Files

| File | Change |
|------|--------|
| `frontend/src/App.tsx` | Add route: `/inventory` → `MedicationInventoryPage` |
| `frontend/src/components/app-sidebar.tsx` | Add "Medication Inventory" link under "Doctor Resources" |
| `.gitignore` | Add `backend/data/` (SQLite DB shouldn't be committed) |

---

## Optional Integration: Supply Gap Warning on Assessment Results

After a prediction runs in the patient assessment page, the score window could:
1. Call `GET /api/inventory/supply-gaps`
2. Filter to only medications the patient has active (non-"No" status)
3. Display a small warning: "Low inventory: insulin, metformin" below the risk badge

This is purely informational — it does NOT modify predictions or the model. Can be implemented independently of the main inventory page.

---

## Implementation Phases

### Phase 1: Backend Database + API
- Create SQLite schema and connection management
- Create Pydantic models for request/response validation
- Create API router with CRUD endpoints
- Add FDA NDC API seeding endpoint
- Add supply-gaps endpoint
- Wire into main.py
- **Test:** Use FastAPI's `/docs` Swagger UI to verify all endpoints

### Phase 2: Frontend Inventory Page
- Create the page component with table, search, and summary cards
- Add route and sidebar navigation
- Build add-drug and stock-update dialogs
- Add supply gap alert banner
- **Test:** Navigate to `/inventory`, seed data, add/restock/dispense drugs

### Phase 3: Prediction Integration (optional)
- Add supply gap warning to score-window.tsx
- **Test:** Run a patient assessment with active meds, verify warning appears when those meds are low stock

---

## What This Does NOT Do

- Does not modify the prediction model or retrain anything
- Does not require any external database services
- Does not add authentication or multi-user support
- Does not track real patient assignments to medications (just hospital-level stock)
- Does not auto-order medications — it's a visibility/alerting tool