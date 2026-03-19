from pydantic import BaseModel


class DrugOut(BaseModel):
    id: int
    ndc_code: str
    generic_name: str
    brand_name: str | None = None
    dosage_form: str | None = None
    strength: str | None = None
    model_field: str | None = None


class InventoryItem(BaseModel):
    id: int
    drug: DrugOut
    quantity_on_hand: int
    reorder_level: int
    unit: str
    last_updated: str | None = None
    notes: str | None = None
    is_low_stock: bool


class LogEntry(BaseModel):
    id: int
    change_amount: int
    reason: str | None = None
    created_at: str | None = None


class InventoryDetail(BaseModel):
    item: InventoryItem
    history: list[LogEntry]


class InventoryUpdate(BaseModel):
    change_amount: int
    reason: str = "adjustment"


class AddInventoryItem(BaseModel):
    ndc_code: str
    generic_name: str
    brand_name: str | None = None
    dosage_form: str | None = None
    strength: str | None = None
    model_field: str | None = None
    initial_quantity: int = 0
    reorder_level: int = 10
    unit: str = "units"


class SupplyGap(BaseModel):
    drug: DrugOut
    quantity_on_hand: int
    reorder_level: int
    model_field: str
    deficit: int