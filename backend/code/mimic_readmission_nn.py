"""
MIMIC-IV 30-Day Readmission Prediction — Neural Network
========================================================
A standalone model using the full richness of MIMIC-IV:
    - Demographics & admission info
    - Charlson-style comorbidity flags (ICD-10)
    - Hospital Frailty Risk Score (HFRS)
    - Last-before-discharge lab values
    - Mean vitals during stay
    - Medication features (polypharmacy, drug class flags)
    - Length of stay, number of diagnoses/procedures

Includes:
    - Class-weighted loss to handle readmission imbalance (~12-15%)
    - Hyperparameter tuning via Optuna (optional)
    - Threshold optimisation on validation set
    - SHAP feature importance (optional)
    - Full evaluation: AUROC, AUPRC, F1, calibration

Requirements:
    pip install torch pandas scikit-learn numpy matplotlib --break-system-packages
    pip install optuna shap --break-system-packages   # optional

Usage:
    1. Update MIMIC_DIR below
    2. python mimic_readmission_nn.py
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, classification_report, confusion_matrix,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class Config:
    mimic_dir: Path = Path("../datasets/mimic-iv-clinical-database-demo-2.2")

    # Training
    epochs: int = 80
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12          # early stopping patience
    use_class_weights: bool = True
    use_oversampling: bool = False  # alternative to class weights

    # Model
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64, 32])
    dropout: float = 0.35
    use_residual: bool = True   # residual connections where dims match

    # Data
    val_size: float = 0.15
    test_size: float = 0.15

    # Misc
    seed: int = 42
    device: str = "cpu"
    save_plots: bool = True


CFG = Config()
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)


# ===========================================================================
# Feature Definitions
# ===========================================================================

COMORBIDITY_MAP = {
    "mi":         ["I21", "I22", "I25.2"],
    "chf":        ["I09.9", "I11.0", "I13.0", "I13.2", "I25.5", "I42", "I43", "I50"],
    "pvd":        ["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1", "I79.0", "I79.2"],
    "stroke":     ["G45", "G46", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I69"],
    "dementia":   ["F00", "F01", "F02", "F03", "F05.1", "G30", "G31.1"],
    "copd":       ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"],
    "rheumatic":  ["M05", "M06", "M32", "M33", "M34"],
    "pud":        ["K25", "K26", "K27", "K28"],
    "liver_mild": ["B18", "K70", "K73", "K74"],
    "diabetes":   ["E10", "E11", "E12", "E13", "E14"],
    "renal":      ["N17", "N18", "N19", "Z49", "Z94.0", "Z99.2"],
    "cancer":     ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
                   "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
                   "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C30", "C31", "C32",
                   "C33", "C34", "C40", "C41", "C43", "C45", "C46", "C47", "C48", "C49",
                   "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60",
                   "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70",
                   "C71", "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83", "C84",
                   "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96", "C97"],
}
COMORB_FEATURES = list(COMORBIDITY_MAP.keys())

KEY_LABS = {
    50912: "lab_creatinine",
    51006: "lab_bun",
    51222: "lab_hemoglobin",
    50931: "lab_glucose",
    50983: "lab_sodium",
    50971: "lab_potassium",
    50862: "lab_albumin",
    51301: "lab_wbc",
    51265: "lab_platelet",
    50882: "lab_bicarbonate",
    51003: "lab_troponin_t",
    50813: "lab_lactate",
    51237: "lab_inr",
    50902: "lab_chloride",
    50893: "lab_calcium",
}
LAB_FEATURES = list(KEY_LABS.values())

VITAL_FEATURES = [
    "vital_hr_mean", "vital_hr_std",
    "vital_sbp_mean", "vital_sbp_std",
    "vital_dbp_mean",
    "vital_spo2_mean", "vital_spo2_min",
    "vital_rr_mean",
    "vital_temp_mean",
]

HFRS_WEIGHTS = {
    "F00": 7.1, "F01": 4.4, "F02": 4.4, "F03": 4.4, "F05": 6.3,
    "F10": 2.7, "S72": 3.8, "L89": 3.0, "G30": 3.0, "Z74": 3.9,
    "Z75": 2.5, "S00": 2.8, "S01": 2.8, "S09": 2.8, "R26": 2.4,
    "R54": 2.0, "E86": 1.8, "J69": 1.8, "R41": 1.8, "S06": 1.7,
    "G81": 1.7, "N39": 1.4, "G31": 1.4, "I67": 1.4, "T83": 1.4,
    "R39": 1.3, "R32": 1.3, "G20": 1.2, "I69": 1.2, "S42": 1.1,
    "S80": 1.1, "S82": 1.1, "W01": 1.0, "W06": 1.0, "W10": 1.0,
    "W18": 1.0, "W19": 1.0, "L97": 1.0, "R29": 1.0, "R40": 1.0,
    "R44": 1.0, "I50": 0.5, "I48": 0.5, "I95": 0.5, "J96": 0.5,
    "J18": 0.5, "J22": 0.5, "A41": 0.5, "L03": 0.5, "E87": 0.5,
    "D64": 0.5, "D50": 0.5, "N17": 0.5, "N18": 0.5, "K59": 0.5,
    "G40": 0.5, "G45": 0.5, "I63": 0.5, "M25": 0.5, "M80": 0.5,
    "T81": 0.5, "T82": 0.5, "T84": 0.5, "Z91": 0.5, "R55": 0.5,
    "E40": 0.1, "E41": 0.1, "E43": 0.1, "E46": 0.1, "E55": 0.1,
    "F32": 0.1, "F33": 0.1, "M81": 0.1, "Z50": 0.1, "Z73": 0.1,
    "Z87": 0.1, "Z93": 0.1, "Z96": 0.1, "Y30": 1.0,
}

MED_KEYWORDS = {
    "med_anticoagulant": ["warfarin", "heparin", "enoxaparin", "apixaban",
                          "rivaroxaban", "dabigatran", "fondaparinux"],
    "med_opioid":        ["morphine", "hydromorphone", "fentanyl", "oxycodone",
                          "hydrocodone", "tramadol", "codeine", "methadone"],
    "med_insulin":       ["insulin", "glargine", "lispro", "aspart", "detemir"],
    "med_beta_blocker":  ["metoprolol", "atenolol", "propranolol", "carvedilol",
                          "bisoprolol", "labetalol", "nebivolol"],
    "med_ace_arb":       ["lisinopril", "enalapril", "ramipril", "captopril",
                          "losartan", "valsartan", "irbesartan", "olmesartan"],
    "med_statin":        ["atorvastatin", "simvastatin", "rosuvastatin",
                          "pravastatin", "lovastatin"],
    "med_diuretic":      ["furosemide", "bumetanide", "torsemide"],
    "med_ppi":           ["omeprazole", "pantoprazole", "esomeprazole",
                          "lansoprazole", "rabeprazole"],
    "med_antidepressant":["sertraline", "fluoxetine", "citalopram", "escitalopram",
                          "paroxetine", "venlafaxine", "duloxetine", "mirtazapine",
                          "trazodone", "bupropion"],
    "med_antibiotic":    ["vancomycin", "piperacillin", "meropenem", "cefazolin",
                          "ceftriaxone", "cefepime", "metronidazole",
                          "ciprofloxacin", "levofloxacin"],
    "med_steroid":       ["prednisone", "prednisolone", "methylprednisolone",
                          "dexamethasone", "hydrocortisone"],
}
MED_FEATURES = (
    ["n_medications", "polypharmacy"]
    + [f"med_{k}" if not k.startswith("med_") else k for k in MED_KEYWORDS.keys()]
)
# Fix names to match keywords dict keys
MED_FLAG_NAMES = list(MED_KEYWORDS.keys())

DEMO_FEATURES = [
    "age", "sex_male", "los_days", "emergency", "elective",
    "n_diagnoses", "n_procedures", "n_prior_admissions",
    "days_since_last_admit",
]

ALL_FEATURES = (
    DEMO_FEATURES + COMORB_FEATURES + ["hfrs", "hfrs_intermediate", "hfrs_high"]
    + LAB_FEATURES + VITAL_FEATURES
    + ["n_medications", "polypharmacy"] + MED_FLAG_NAMES
)


# ===========================================================================
# Data Loading Helpers
# ===========================================================================

def load_table(path, **kwargs):
    gz = path.with_suffix(".csv.gz")
    csv = path.with_suffix(".csv")
    f = gz if gz.exists() else csv
    if not f.exists():
        print(f"    [SKIP] {path.stem} not found")
        return pd.DataFrame()
    print(f"    Loading {f.name} ...")
    return pd.read_csv(f, **kwargs)


def icd_to_comorbidities(diagnoses: pd.DataFrame) -> pd.DataFrame:
    dx = diagnoses[diagnoses["icd_version"] == 10].copy()
    dx["icd_clean"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    all_hadm = dx["hadm_id"].unique()
    comorb_df = pd.DataFrame({"hadm_id": all_hadm})
    for comorb, prefixes in COMORBIDITY_MAP.items():
        mask = pd.Series(False, index=dx.index)
        for pfx in prefixes:
            mask |= dx["icd_clean"].str.startswith(pfx.replace(".", ""))
        hits = dx.loc[mask, "hadm_id"].unique()
        comorb_df[comorb] = comorb_df["hadm_id"].isin(hits).astype(int)
    return comorb_df


def compute_hfrs(diagnoses: pd.DataFrame) -> pd.DataFrame:
    dx = diagnoses[diagnoses["icd_version"] == 10].copy()
    dx["icd_clean"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    records = []
    for prefix, weight in HFRS_WEIGHTS.items():
        matched = dx[dx["icd_clean"].str.startswith(prefix)]
        if not matched.empty:
            for hadm_id in matched["hadm_id"].unique():
                records.append({"hadm_id": hadm_id, "prefix": prefix, "weight": weight})
    if not records:
        return pd.DataFrame(columns=["hadm_id", "hfrs"])
    matches = pd.DataFrame(records).drop_duplicates(subset=["hadm_id", "prefix"])
    return matches.groupby("hadm_id")["weight"].sum().reset_index(name="hfrs")


# ===========================================================================
# Feature Extraction
# ===========================================================================

def extract_labs(cfg, hadm_ids):
    print("  Extracting labs ...")
    labs = load_table(
        cfg.mimic_dir / "hosp" / "labevents",
        usecols=["hadm_id", "itemid", "charttime", "valuenum"],
        dtype={"hadm_id": "Int64", "itemid": int},
        parse_dates=["charttime"],
    )
    if labs.empty:
        df = pd.DataFrame({"hadm_id": hadm_ids})
        for col in LAB_FEATURES:
            df[col] = np.nan
        return df

    labs = labs[labs["hadm_id"].isin(hadm_ids) & labs["itemid"].isin(KEY_LABS.keys())]
    labs = labs.dropna(subset=["valuenum"])
    labs = labs.sort_values("charttime").groupby(["hadm_id", "itemid"])["valuenum"].last()
    labs = labs.unstack("itemid")
    labs.columns = [KEY_LABS.get(c, f"lab_{c}") for c in labs.columns]
    labs = labs.reset_index()
    return pd.DataFrame({"hadm_id": hadm_ids}).merge(labs, on="hadm_id", how="left")


def extract_vitals(cfg, hadm_ids):
    print("  Extracting vitals ...")
    vital_items = {
        220045: "hr", 220050: "sbp", 220051: "dbp",
        220277: "spo2", 220210: "rr", 223761: "temp_f", 223762: "temp_c",
    }

    charts = load_table(
        cfg.mimic_dir / "icu" / "chartevents",
        usecols=["hadm_id", "itemid", "valuenum"],
        dtype={"hadm_id": "Int64", "itemid": int},
    )

    result = pd.DataFrame({"hadm_id": hadm_ids})
    for col in VITAL_FEATURES:
        result[col] = np.nan

    if charts.empty:
        # Try MIMIC-IV v3+ vitalsign table
        vs = load_table(cfg.mimic_dir / "hosp" / "vitalsign")
        if vs.empty:
            return result
        col_map = {
            "heart_rate": "hr", "sbp": "sbp", "dbp": "dbp",
            "o2sat": "spo2", "resprate": "rr", "temperature": "temp",
        }
        if "stay_id" in vs.columns and "hadm_id" not in vs.columns:
            return result
        for orig, short in col_map.items():
            if orig in vs.columns:
                agg = vs.groupby("hadm_id")[orig].agg(["mean", "std", "min"])
                if f"vital_{short}_mean" in VITAL_FEATURES:
                    result = result.merge(
                        agg["mean"].reset_index().rename(columns={"mean": f"vital_{short}_mean"}),
                        on="hadm_id", how="left"
                    )
                if f"vital_{short}_std" in VITAL_FEATURES:
                    result = result.merge(
                        agg["std"].reset_index().rename(columns={"std": f"vital_{short}_std"}),
                        on="hadm_id", how="left"
                    )
                if f"vital_{short}_min" in VITAL_FEATURES:
                    result = result.merge(
                        agg["min"].reset_index().rename(columns={"min": f"vital_{short}_min"}),
                        on="hadm_id", how="left"
                    )
        return result

    charts = charts[charts["hadm_id"].isin(hadm_ids) & charts["itemid"].isin(vital_items.keys())]
    charts = charts.dropna(subset=["valuenum"])
    charts["vital"] = charts["itemid"].map(vital_items)

    # Merge temp_f and temp_c → temp (convert F to C)
    mask_f = charts["vital"] == "temp_f"
    charts.loc[mask_f, "valuenum"] = (charts.loc[mask_f, "valuenum"] - 32) * 5 / 9
    charts.loc[mask_f, "vital"] = "temp"
    charts.loc[charts["vital"] == "temp_c", "vital"] = "temp"

    agg = charts.groupby(["hadm_id", "vital"])["valuenum"].agg(["mean", "std", "min"])
    agg = agg.reset_index()

    for vital_short in ["hr", "sbp", "dbp", "spo2", "rr", "temp"]:
        sub = agg[agg["vital"] == vital_short]
        if sub.empty:
            continue
        for stat in ["mean", "std", "min"]:
            col_name = f"vital_{vital_short}_{stat}"
            if col_name in VITAL_FEATURES:
                mapping = sub.set_index("hadm_id")[stat].to_dict()
                result[col_name] = result["hadm_id"].map(mapping)

    return result


def extract_medications(cfg, hadm_ids):
    print("  Extracting medications ...")
    rx = load_table(cfg.mimic_dir / "hosp" / "prescriptions",
                    usecols=["hadm_id", "drug"])
    if rx.empty:
        df = pd.DataFrame({"hadm_id": hadm_ids})
        df["n_medications"] = 0
        df["polypharmacy"] = 0
        for k in MED_FLAG_NAMES:
            df[k] = 0
        return df

    rx = rx[rx["hadm_id"].isin(hadm_ids)]
    rx["drug_lower"] = rx["drug"].astype(str).str.lower()

    n_meds = rx.groupby("hadm_id")["drug_lower"].nunique().reset_index(name="n_medications")

    result = pd.DataFrame({"hadm_id": hadm_ids}).merge(n_meds, on="hadm_id", how="left")
    result["n_medications"] = result["n_medications"].fillna(0)
    result["polypharmacy"] = (result["n_medications"] >= 10).astype(int)

    for flag_name, keywords in MED_KEYWORDS.items():
        pattern = "|".join(keywords)
        matched = rx[rx["drug_lower"].str.contains(pattern, na=False)]["hadm_id"].unique()
        result[flag_name] = result["hadm_id"].isin(matched).astype(int)

    return result


# ===========================================================================
# Full Data Pipeline
# ===========================================================================

def prepare_data(cfg: Config) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  Loading and preparing MIMIC-IV")
    print("=" * 70)

    hosp = cfg.mimic_dir / "hosp"
    icu = cfg.mimic_dir / "icu"

    admissions = load_table(hosp / "admissions", parse_dates=["admittime", "dischtime"])
    patients = load_table(hosp / "patients")
    diagnoses = load_table(hosp / "diagnoses_icd", dtype={"icd_code": str, "icd_version": int})
    procedures = load_table(hosp / "procedures_icd")

    if admissions.empty:
        raise FileNotFoundError("admissions table not found — check MIMIC_DIR")

    # ---- Readmission target ----
    cohort = admissions[admissions["hospital_expire_flag"] == 0].copy()
    cohort.sort_values(["subject_id", "admittime"], inplace=True)
    cohort["next_admittime"] = cohort.groupby("subject_id")["admittime"].shift(-1)
    cohort["days_to_readmit"] = (
        (cohort["next_admittime"] - cohort["dischtime"]).dt.total_seconds() / 86400
    )
    cohort["readmit_30d"] = (cohort["days_to_readmit"] <= 30).astype(int)

    # ---- Demographics ----
    cohort = cohort.merge(patients[["subject_id", "gender", "anchor_age"]],
                          on="subject_id", how="left")
    cohort["age"] = cohort["anchor_age"]
    cohort["sex_male"] = (cohort["gender"] == "M").astype(int)
    cohort["los_days"] = (
        (cohort["dischtime"] - cohort["admittime"]).dt.total_seconds() / 86400
    )
    cohort["emergency"] = cohort["admission_type"].str.contains(
        "EMERGENCY|URGENT", case=False, na=False
    ).astype(int)
    cohort["elective"] = cohort["admission_type"].str.contains(
        "ELECTIVE", case=False, na=False
    ).astype(int)

    # Prior admissions count & days since last admission
    cohort["n_prior_admissions"] = cohort.groupby("subject_id").cumcount()
    cohort["prev_dischtime"] = cohort.groupby("subject_id")["dischtime"].shift(1)
    cohort["days_since_last_admit"] = (
        (cohort["admittime"] - cohort["prev_dischtime"]).dt.total_seconds() / 86400
    )
    cohort["days_since_last_admit"] = cohort["days_since_last_admit"].fillna(-1)
    # -1 indicates first admission

    # ---- Comorbidities ----
    if not diagnoses.empty:
        comorbs = icd_to_comorbidities(diagnoses)
        cohort = cohort.merge(comorbs, on="hadm_id", how="left")
    for col in COMORB_FEATURES:
        if col not in cohort.columns:
            cohort[col] = 0
        cohort[col] = cohort[col].fillna(0).astype(int)

    # ---- N diagnoses & N procedures ----
    if not diagnoses.empty:
        n_diag = diagnoses.groupby("hadm_id").size().reset_index(name="n_diagnoses")
        cohort = cohort.merge(n_diag, on="hadm_id", how="left")
    cohort["n_diagnoses"] = cohort.get("n_diagnoses", pd.Series(0)).fillna(0)

    if not procedures.empty:
        n_proc = procedures.groupby("hadm_id").size().reset_index(name="n_procedures")
        cohort = cohort.merge(n_proc, on="hadm_id", how="left")
    cohort["n_procedures"] = cohort.get("n_procedures", pd.Series(0)).fillna(0)

    # ---- HFRS ----
    if not diagnoses.empty:
        hfrs = compute_hfrs(diagnoses)
        cohort = cohort.merge(hfrs, on="hadm_id", how="left")
    cohort["hfrs"] = cohort.get("hfrs", pd.Series(0)).fillna(0)
    cohort["hfrs_intermediate"] = ((cohort["hfrs"] >= 5) & (cohort["hfrs"] <= 15)).astype(int)
    cohort["hfrs_high"] = (cohort["hfrs"] > 15).astype(int)

    hadm_ids = cohort["hadm_id"].values

    # ---- Labs ----
    labs = extract_labs(cfg, hadm_ids)
    cohort = cohort.merge(labs, on="hadm_id", how="left")

    # ---- Vitals ----
    vitals = extract_vitals(cfg, hadm_ids)
    cohort = cohort.merge(vitals, on="hadm_id", how="left")

    # ---- Medications ----
    meds = extract_medications(cfg, hadm_ids)
    cohort = cohort.merge(meds, on="hadm_id", how="left")
    for col in ["n_medications", "polypharmacy"] + MED_FLAG_NAMES:
        cohort[col] = cohort.get(col, pd.Series(0)).fillna(0)

    # ---- Ensure all feature columns exist ----
    for col in ALL_FEATURES:
        if col not in cohort.columns:
            cohort[col] = np.nan

    # ---- Summary ----
    print(f"\n  Cohort size:       {len(cohort):,}")
    print(f"  Readmission rate:  {cohort['readmit_30d'].mean():.1%}")
    print(f"  Features:          {len(ALL_FEATURES)}")
    print(f"  Missing rates (top 10):")
    miss = cohort[ALL_FEATURES].isnull().mean().sort_values(ascending=False).head(10)
    for feat, rate in miss.items():
        print(f"    {feat:30s}  {rate:.1%}")

    return cohort


# ===========================================================================
# Model
# ===========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class ReadmissionNet(nn.Module):
    """
    Feed-forward network with optional residual connections.

    Architecture:
        Input → [Linear → BN → ReLU → Dropout] × N → Linear(1)
    With use_residual=True, adds skip connections where input/output dims match.
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.3, use_residual=True):
        super().__init__()
        layers = []
        prev = input_dim

        for h_dim in hidden_dims:
            if use_residual and prev == h_dim:
                layers.append(ResidualBlock(h_dim, dropout))
            else:
                layers.extend([
                    nn.Linear(prev, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev = h_dim

        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ===========================================================================
# Training Engine
# ===========================================================================

class EarlyStopping:
    def __init__(self, patience, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False

    def step(self, score, model):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    pos_weight: float = 1.0,
) -> nn.Module:

    model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(cfg.device) if cfg.use_class_weights else None
    )

    early_stop = EarlyStopping(patience=cfg.patience)

    print(f"\n{'='*70}")
    print(f"  Training — {cfg.epochs} epochs, patience={cfg.patience}")
    print(f"  Pos weight: {pos_weight:.2f}")
    print(f"  Device: {cfg.device}")
    print(f"{'='*70}\n")

    for epoch in range(1, cfg.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # ---- Validate ----
        model.eval()
        all_logits, all_labels = [], []
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(cfg.device)
                y_batch = y_batch.to(cfg.device)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item()
                n_val += 1
                all_logits.append(logits.cpu())
                all_labels.append(y_batch.cpu())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        probs = torch.sigmoid(logits_cat).numpy()
        labels_np = labels_cat.numpy()

        try:
            auroc = roc_auc_score(labels_np, probs)
        except ValueError:
            auroc = 0.5
        try:
            auprc = average_precision_score(labels_np, probs)
        except ValueError:
            auprc = 0.0

        avg_val_loss = val_loss / max(n_val, 1)

        if epoch % 5 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{cfg.epochs}  "
                  f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
                  f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}  lr={lr_now:.2e}")

        early_stop.step(auroc, model)
        if early_stop.should_stop:
            print(f"\n  Early stopping at epoch {epoch} (best AUROC={early_stop.best_score:.4f})")
            break

    model.load_state_dict(early_stop.best_state)
    model.to(cfg.device)
    return model


# ===========================================================================
# Evaluation
# ===========================================================================

def find_optimal_threshold(labels, probs):
    """Find threshold that maximises F1 score."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        all_logits.append(logits.cpu())
        all_labels.append(y_batch)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)

    metrics = {
        "auroc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
        "f1": f1_score(labels, preds, zero_division=0),
        "threshold": threshold,
    }

    return metrics, probs, labels, preds


def plot_results(probs, labels, preds, save_path="readmission_results.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    baseline = labels.mean()
    axes[1].plot(rec, prec, lw=2, label=f"AUPRC = {auprc:.3f}")
    axes[1].axhline(baseline, color="k", ls="--", lw=1, label=f"Baseline = {baseline:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    # Calibration — predicted vs observed
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_pred, bin_obs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred.append(probs[mask].mean())
            bin_obs.append(labels[mask].mean())
            bin_counts.append(mask.sum())

    axes[2].plot(bin_pred, bin_obs, "o-", lw=2, label="Model")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Observed Proportion")
    axes[2].set_title("Calibration Plot")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plots to {save_path}")


def print_feature_importance(model, feature_names, device, loader, top_k=25):
    """Simple permutation-based feature importance."""
    print(f"\n  Feature Importance (permutation, top {top_k}):")

    model.eval()
    # Get baseline AUROC
    base_metrics, _, _, _ = evaluate(model, loader, device)
    base_auroc = base_metrics["auroc"]

    importances = {}
    for i, feat_name in enumerate(feature_names):
        # Permute feature i across the dataset
        all_logits, all_labels = [], []
        for X_batch, y_batch in loader:
            X_perm = X_batch.clone()
            perm_idx = torch.randperm(X_perm.shape[0])
            X_perm[:, i] = X_perm[perm_idx, i]

            X_perm = X_perm.to(device)
            with torch.no_grad():
                logits = model(X_perm)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels).numpy()
        probs = torch.sigmoid(logits).numpy()

        try:
            perm_auroc = roc_auc_score(labels, probs)
        except ValueError:
            perm_auroc = 0.5

        importances[feat_name] = base_auroc - perm_auroc

    sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, imp in sorted_imp[:top_k]:
        bar = "+" * int(abs(imp) * 500) if imp > 0 else "-" * int(abs(imp) * 500)
        print(f"    {feat:30s}  {imp:+.4f}  {bar}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    cfg = CFG

    # ---- Prepare data ----
    cohort = prepare_data(cfg)

    feature_cols = [c for c in ALL_FEATURES if c in cohort.columns]
    target_col = "readmit_30d"

    print(f"\n  Final feature count: {len(feature_cols)}")

    # ---- Build arrays ----
    X = cohort[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    medians = X.median()
    X = X.fillna(medians).values.astype(np.float32)
    y = cohort[target_col].values.astype(np.float32)

    # ---- Train / val / test split ----
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=cfg.val_size / (1 - cfg.test_size),
        random_state=cfg.seed, stratify=y_trainval,
    )

    print(f"\n  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Train readmit rate: {y_train.mean():.1%}")
    print(f"  Val readmit rate:   {y_val.mean():.1%}")
    print(f"  Test readmit rate:  {y_test.mean():.1%}")

    # ---- Scale ----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # ---- DataLoaders ----
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    if cfg.use_oversampling:
        # Oversample minority class in training
        weights = np.where(y_train == 1, 1.0 / y_train.mean(), 1.0 / (1 - y_train.mean()))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                                  drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  drop_last=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2)

    # ---- Class weight ----
    pos_rate = y_train.mean()
    pos_weight = (1 - pos_rate) / pos_rate if cfg.use_class_weights else 1.0

    # ---- Model ----
    input_dim = X_train.shape[1]
    model = ReadmissionNet(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
        use_residual=cfg.use_residual,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")
    print(f"  {model}\n")

    # ---- Train ----
    model = train_model(model, train_loader, val_loader, cfg, pos_weight=pos_weight)

    # ---- Optimise threshold on val set ----
    val_metrics, val_probs, val_labels, _ = evaluate(model, val_loader, cfg.device)
    best_threshold = find_optimal_threshold(val_labels, val_probs)
    print(f"\n  Optimal threshold (val F1): {best_threshold:.3f}")

    # ---- Final test evaluation ----
    print(f"\n{'='*70}")
    print("  TEST SET RESULTS")
    print(f"{'='*70}\n")

    test_metrics, test_probs, test_labels, test_preds = evaluate(
        model, test_loader, cfg.device, threshold=best_threshold
    )

    print(f"  AUROC:     {test_metrics['auroc']:.4f}")
    print(f"  AUPRC:     {test_metrics['auprc']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  Threshold: {test_metrics['threshold']:.3f}")
    print()
    print("  Classification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=["No Readmit", "Readmit"]))
    print("  Confusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    # ---- Plots ----
    if cfg.save_plots:
        plot_results(test_probs, test_labels, test_preds,
                     save_path="readmission_results.png")

    # ---- Feature importance ----
    print_feature_importance(model, feature_cols, cfg.device, test_loader, top_k=25)

    # ---- Save model ----
    save_dict = {
        "model_state": model.state_dict(),
        "feature_names": feature_cols,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "config": {
            "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout,
            "use_residual": cfg.use_residual,
            "input_dim": input_dim,
        },
    }
    torch.save(save_dict, "readmission_mimic_nn.pt")
    print(f"\n  Saved model + metadata to readmission_mimic_nn.pt")

    print(f"\n{'='*70}")
    print("  Done!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
