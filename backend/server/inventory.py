import json
import urllib.request
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query

from .database import get_db
from .inventory_models import (
    AddInventoryItem,
    DrugOut,
    InventoryDetail,
    InventoryItem,
    InventoryUpdate,
    LogEntry,
    SupplyGap,
)

router = APIRouter(prefix="/api", tags=["inventory"])

# The 23 medication field names used by the prediction model
MODEL_MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def _row_to_drug(row) -> DrugOut:
    return DrugOut(
        id=row["id"],
        ndc_code=row["ndc_code"],
        generic_name=row["generic_name"],
        brand_name=row["brand_name"],
        dosage_form=row["dosage_form"],
        strength=row["strength"],
        model_field=row["model_field"],
    )


def _row_to_item(row) -> InventoryItem:
    drug = DrugOut(
        id=row["drug_id"] if "drug_id" in row.keys() else row["id"],
        ndc_code=row["ndc_code"],
        generic_name=row["generic_name"],
        brand_name=row["brand_name"],
        dosage_form=row["dosage_form"],
        strength=row["strength"],
        model_field=row["model_field"],
    )
    return InventoryItem(
        id=row["inv_id"] if "inv_id" in row.keys() else row["id"],
        drug=drug,
        quantity_on_hand=row["quantity_on_hand"],
        reorder_level=row["reorder_level"],
        unit=row["unit"],
        last_updated=row["last_updated"],
        notes=row["notes"],
        is_low_stock=row["quantity_on_hand"] <= row["reorder_level"],
    )


# ── List inventory ───────────────────────────────────────────────────────────

@router.get("/inventory", response_model=list[InventoryItem])
def list_inventory(low_stock: bool = Query(False)):
    conn = get_db()
    query = """
        SELECT d.id AS drug_id, d.ndc_code, d.generic_name, d.brand_name,
               d.dosage_form, d.strength, d.model_field,
               i.id AS inv_id, i.quantity_on_hand, i.reorder_level,
               i.unit, i.last_updated, i.notes
        FROM inventory i
        JOIN drugs d ON d.id = i.drug_id
    """
    if low_stock:
        query += " WHERE i.quantity_on_hand <= i.reorder_level"
    query += " ORDER BY d.generic_name"

    rows = conn.execute(query).fetchall()
    conn.close()
    return [_row_to_item(r) for r in rows]


# ── Get single inventory item with history ───────────────────────────────────

@router.get("/inventory/supply-gaps", response_model=list[SupplyGap])
def get_supply_gaps():
    conn = get_db()
    rows = conn.execute("""
        SELECT d.id AS drug_id, d.ndc_code, d.generic_name, d.brand_name,
               d.dosage_form, d.strength, d.model_field,
               i.quantity_on_hand, i.reorder_level
        FROM inventory i
        JOIN drugs d ON d.id = i.drug_id
        WHERE d.model_field IS NOT NULL
          AND i.quantity_on_hand <= i.reorder_level
        ORDER BY (i.reorder_level - i.quantity_on_hand) DESC
    """).fetchall()
    conn.close()
    return [
        SupplyGap(
            drug=DrugOut(
                id=r["drug_id"], ndc_code=r["ndc_code"],
                generic_name=r["generic_name"], brand_name=r["brand_name"],
                dosage_form=r["dosage_form"], strength=r["strength"],
                model_field=r["model_field"],
            ),
            quantity_on_hand=r["quantity_on_hand"],
            reorder_level=r["reorder_level"],
            model_field=r["model_field"],
            deficit=r["reorder_level"] - r["quantity_on_hand"],
        )
        for r in rows
    ]


@router.get("/inventory/model-medications", response_model=list[str])
def get_model_medications():
    return MODEL_MEDICATIONS


@router.get("/inventory/{drug_id}", response_model=InventoryDetail)
def get_inventory_item(drug_id: int):
    conn = get_db()
    row = conn.execute("""
        SELECT d.id AS drug_id, d.ndc_code, d.generic_name, d.brand_name,
               d.dosage_form, d.strength, d.model_field,
               i.id AS inv_id, i.quantity_on_hand, i.reorder_level,
               i.unit, i.last_updated, i.notes
        FROM inventory i
        JOIN drugs d ON d.id = i.drug_id
        WHERE d.id = ?
    """, (drug_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Inventory item not found")

    history = conn.execute("""
        SELECT id, change_amount, reason, created_at
        FROM inventory_log
        WHERE drug_id = ?
        ORDER BY created_at DESC
        LIMIT 50
    """, (drug_id,)).fetchall()
    conn.close()

    return InventoryDetail(
        item=_row_to_item(row),
        history=[LogEntry(id=h["id"], change_amount=h["change_amount"],
                          reason=h["reason"], created_at=h["created_at"]) for h in history],
    )


# ── Add drug + initial stock ─────────────────────────────────────────────────

@router.post("/inventory", response_model=InventoryItem)
def add_inventory_item(data: AddInventoryItem):
    conn = get_db()
    try:
        cur = conn.execute("""
            INSERT INTO drugs (ndc_code, generic_name, brand_name, dosage_form, strength, model_field)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (data.ndc_code, data.generic_name, data.brand_name,
              data.dosage_form, data.strength, data.model_field))
        drug_id = cur.lastrowid

        conn.execute("""
            INSERT INTO inventory (drug_id, quantity_on_hand, reorder_level, unit)
            VALUES (?, ?, ?, ?)
        """, (drug_id, data.initial_quantity, data.reorder_level, data.unit))

        if data.initial_quantity > 0:
            conn.execute("""
                INSERT INTO inventory_log (drug_id, change_amount, reason)
                VALUES (?, ?, 'initial stock')
            """, (drug_id, data.initial_quantity))

        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.close()
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=409, detail=f"Drug with NDC {data.ndc_code} already exists")
        raise HTTPException(status_code=500, detail=str(e))

    row = conn.execute("""
        SELECT d.id AS drug_id, d.ndc_code, d.generic_name, d.brand_name,
               d.dosage_form, d.strength, d.model_field,
               i.id AS inv_id, i.quantity_on_hand, i.reorder_level,
               i.unit, i.last_updated, i.notes
        FROM inventory i
        JOIN drugs d ON d.id = i.drug_id
        WHERE d.id = ?
    """, (drug_id,)).fetchone()
    conn.close()
    return _row_to_item(row)


# ── Update stock (restock / dispense) ────────────────────────────────────────

@router.patch("/inventory/{drug_id}", response_model=InventoryItem)
def update_stock(drug_id: int, data: InventoryUpdate):
    conn = get_db()
    row = conn.execute("SELECT id, quantity_on_hand FROM inventory WHERE drug_id = ?", (drug_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Inventory item not found")

    new_qty = row["quantity_on_hand"] + data.change_amount
    if new_qty < 0:
        conn.close()
        raise HTTPException(status_code=400, detail="Stock cannot go below zero")

    conn.execute("""
        UPDATE inventory SET quantity_on_hand = ?, last_updated = datetime('now')
        WHERE drug_id = ?
    """, (new_qty, drug_id))
    conn.execute("""
        INSERT INTO inventory_log (drug_id, change_amount, reason)
        VALUES (?, ?, ?)
    """, (drug_id, data.change_amount, data.reason))
    conn.commit()

    updated = conn.execute("""
        SELECT d.id AS drug_id, d.ndc_code, d.generic_name, d.brand_name,
               d.dosage_form, d.strength, d.model_field,
               i.id AS inv_id, i.quantity_on_hand, i.reorder_level,
               i.unit, i.last_updated, i.notes
        FROM inventory i
        JOIN drugs d ON d.id = i.drug_id
        WHERE d.id = ?
    """, (drug_id,)).fetchone()
    conn.close()
    return _row_to_item(updated)


# ── Delete ───────────────────────────────────────────────────────────────────

@router.delete("/inventory/{drug_id}")
def delete_inventory_item(drug_id: int):
    conn = get_db()
    row = conn.execute("SELECT id FROM drugs WHERE id = ?", (drug_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Drug not found")

    conn.execute("DELETE FROM drugs WHERE id = ?", (drug_id,))
    conn.commit()
    conn.close()
    return {"detail": "Deleted"}


# ── Search drugs ─────────────────────────────────────────────────────────────

@router.get("/drugs/search", response_model=list[DrugOut])
def search_drugs(q: str = Query(..., min_length=1)):
    conn = get_db()
    pattern = f"%{q}%"
    rows = conn.execute("""
        SELECT id, ndc_code, generic_name, brand_name, dosage_form, strength, model_field
        FROM drugs
        WHERE generic_name LIKE ? OR brand_name LIKE ? OR ndc_code LIKE ?
        ORDER BY generic_name
        LIMIT 50
    """, (pattern, pattern, pattern)).fetchall()
    conn.close()
    return [_row_to_drug(r) for r in rows]


# ── Seed from FDA NDC API ────────────────────────────────────────────────────

@router.post("/drugs/seed")
def seed_from_ndc():
    conn = get_db()
    added = 0
    skipped = 0
    errors = []

    for med_name in MODEL_MEDICATIONS:
        # Map model field names to FDA search terms
        search_name = med_name.replace("-", " ")
        url = f"https://api.fda.gov/drug/ndc.json?search=generic_name:\"{quote(search_name)}\"&limit=5"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HospitalReadmissionApp/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            for result in data.get("results", []):
                ndc = result.get("product_ndc", "")
                if not ndc:
                    continue

                generic = result.get("generic_name", search_name).upper()
                brand = result.get("brand_name")
                dosage_form = result.get("dosage_form")

                # Get strength from active_ingredients
                strength = None
                ingredients = result.get("active_ingredients", [])
                if ingredients:
                    strength = ingredients[0].get("strength")

                # Check if already exists
                existing = conn.execute("SELECT id FROM drugs WHERE ndc_code = ?", (ndc,)).fetchone()
                if existing:
                    skipped += 1
                    continue

                cur = conn.execute("""
                    INSERT INTO drugs (ndc_code, generic_name, brand_name, dosage_form, strength, model_field)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (ndc, generic, brand, dosage_form, strength, med_name))
                drug_id = cur.lastrowid

                max_stock = 100
                conn.execute("""
                    INSERT INTO inventory (drug_id, quantity_on_hand, reorder_level, unit)
                    VALUES (?, ?, 10, 'units')
                """, (drug_id, max_stock))

                conn.execute("""
                    INSERT INTO inventory_log (drug_id, change_amount, reason)
                    VALUES (?, ?, 'initial stock')
                """, (drug_id, max_stock))

                added += 1

        except Exception as e:
            errors.append(f"{med_name}: {str(e)}")

    conn.commit()
    conn.close()

    return {
        "added": added,
        "skipped": skipped,
        "errors": errors,
        "message": f"Seeded {added} drugs ({skipped} already existed, initial stock: 100)"
    }


# ── Randomize stock levels (for testing) ─────────────────────────────────────

@router.post("/inventory/randomize")
def randomize_stock(seed: int = Query(0)):
    import random
    rng = random.Random(seed if seed != 0 else None)

    conn = get_db()
    rows = conn.execute("SELECT drug_id FROM inventory").fetchall()
    if not rows:
        conn.close()
        return {"detail": "No inventory items to randomize", "count": 0}

    for row in rows:
        qty = rng.randint(0, 100)
        conn.execute("""
            UPDATE inventory SET quantity_on_hand = ?, last_updated = datetime('now')
            WHERE drug_id = ?
        """, (qty, row["drug_id"]))
        conn.execute("""
            INSERT INTO inventory_log (drug_id, change_amount, reason)
            VALUES (?, ?, ?)
        """, (row["drug_id"], qty, f"randomize (seed={seed})"))

    conn.commit()
    conn.close()
    return {"detail": f"Randomized {len(rows)} items (seed={seed})", "count": len(rows)}
