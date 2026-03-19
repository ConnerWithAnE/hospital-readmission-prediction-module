import base64
import io
import sys
from pathlib import Path

# Allow running this file directly (python data_processing.py) by fixing the
# package path before relative imports are resolved.
if __name__ == "__main__" and __package__ is None:
    _backend = str(Path(__file__).resolve().parent.parent)
    if _backend not in sys.path:
        sys.path.insert(0, _backend)
    __package__ = "code"

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sb
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, classification_report, \
    confusion_matrix

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from .cci import add_comorbidities_to_dataframe
from .data_maps import keep, admission_type_map, discharge_disposition_map, age_map, admission_source_map, med_columns, icd9_to_category


class PredictionModel:

    def __init__(self):
        self.diabetes_130_us_hospitals = fetch_ucirepo(id=296)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def refine_dataset(self):
        df = self.diabetes_130_us_hospitals.data.features.copy()

        df["readmitted"] = self.diabetes_130_us_hospitals.data.targets["readmitted"]

        df = add_comorbidities_to_dataframe(df)

        # Aggregate medication features (computed before keep filter drops individual med columns)
        med_df = df[med_columns]
        df["n_active_meds"] = (med_df != "No").sum(axis=1)
        df["n_med_changes"] = med_df.isin(["Up", "Down"]).sum(axis=1)
        df["any_dose_up"] = med_df.isin(["Up"]).any(axis=1).astype(int)

        df = df[keep + ["n_active_meds", "n_med_changes", "any_dose_up"]].copy()

        df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).fillna(-1)

        # Map admission type by importance
        df['admission_type'] = df['admission_type_id'].map(admission_type_map).fillna("unknown")
        df['admission_source'] = df['admission_source_id'].map(admission_source_map).fillna("unknown")
        df['discharge_group'] = df['discharge_disposition_id'].map(discharge_disposition_map).fillna("unknown")

        # Drop birth as they are different compared to an emergency or other issue (frequent visits)
        df = df[df["admission_type"] != "birth"]
        df = df[df['discharge_group'] != "hospice_death"]


        df['age'] = df['age'].map(age_map)
        df['race'] = df['race'].fillna("unknown")

        # Diabetes-specific clinical columns
        df["A1Cresult"] = df["A1Cresult"].map({"Norm": 0, ">7": 1, ">8": 2}).fillna(-1)
        df["max_glu_serum"] = df["max_glu_serum"].map({"Norm": 0, ">200": 1, ">300": 2}).fillna(-1)
        df["diabetesMed"] = df["diabetesMed"].map({"No": 0, "Yes": 1})
        df["change"] = df["change"].map({"No": 0, "Ch": 1})

        # Individual medication encoding (ordinal: No < Steady < Down < Up)
        med_ordinal = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
        df["insulin"] = df["insulin"].map(med_ordinal)
        df["metformin"] = df["metformin"].map(med_ordinal)

        # Interaction: prior utilization intensity
        df["total_prior_visits"] = df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
        df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)

        # Medication burden relative to stay
        df["meds_per_day"] = df["num_medications"] / (df["time_in_hospital"] + 1)

        # Diagnosis complexity relative to sta
        stay = np.where(df["time_in_hospital"] == 0, 1, df["time_in_hospital"])
        df["diag_per_day"] = np.where(
            df["number_diagnoses"] == 0,
            0,
            df["number_diagnoses"] / stay
        )

        # Lab-to-procedure ratio (diagnostic uncertainty signal)
        df["lab_proc_ratio"] = df["num_lab_procedures"] / (df["num_procedures"] + 1)

        # Emergency frequency ratio (emergency-heavy prior utilization)
        df["emergency_ratio"] = df["number_emergency"] / (df["total_prior_visits"] + 1)

        # Insulin x A1C interaction (poorly controlled despite treatment)
        df["insulin_x_a1c"] = df["insulin"] * df["A1Cresult"]

        # Primary diagnosis category from ICD-9 code
        df["diag_category"] = df["diag_1"].apply(icd9_to_category)

        cat_cols = ['race', 'discharge_group', 'admission_type', 'admission_source', 'diag_category']
        encoded_values = self.encoder.fit_transform(df[cat_cols])
        new_cols = self.encoder.get_feature_names_out(cat_cols)

        df_encoded = pd.DataFrame(encoded_values, columns=new_cols, index=df.index)
        data_final = pd.concat([df.drop(columns=['race', 'discharge_disposition_id', 'discharge_group',
                                                 'admission_type', 'admission_source', 'admission_type_id',
                                                 'admission_source_id', 'diag_1', 'diag_category']), df_encoded],
                               axis=1)
        return data_final

    def get_split(self, ds):
        X = ds.drop(columns=["readmitted"])
        y = ds["readmitted"].map({"<30": 1, ">30": 0, "NO": 0})
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def train(self):
        refined_data = self.refine_dataset()
        self.get_split(refined_data)

        neg, pos = int((self.y_train == 0).sum()), int((self.y_train == 1).sum())
        spw = neg / pos
        print(f"Class balance: {neg} neg, {pos} pos (ratio {spw:.1f}:1)")

        # SMOTE to balance training set — gives wider probability spread
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        print(f"SMOTE: {len(self.X_train)} -> {len(X_resampled)} training samples")

        # Feature pruning: train scout LGBM to identify low-importance features
        lgbm_scout = LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
        )
        lgbm_scout.fit(X_resampled, y_resampled)
        importances = lgbm_scout.feature_importances_
        feature_names = self.X_train.columns.tolist()
        indexed = sorted(zip(feature_names, importances), key=lambda x: x[1])
        n_drop = max(1, len(indexed) // 7)  # ~15%
        self.drop_features = [name for name, imp in indexed[:n_drop]]
        print(f"Pruning {n_drop} low-importance features: {self.drop_features}")

        X_resampled = X_resampled.drop(columns=self.drop_features)
        self.X_train = self.X_train.drop(columns=self.drop_features)
        self.X_test = self.X_test.drop(columns=self.drop_features)

        base_models = [
            ('lr', LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs')),
            ('lgbm', LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=12,
            )),
            ('catboost', CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                l2_leaf_reg=3, random_seed=42, verbose=0,
            )),
        ]

        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5,
            stack_method='predict_proba',
            passthrough=False
        )

        self.model.fit(X_resampled, y_resampled)

        # Save reference to the fitted LGBM for feature contributions
        self.lgbm_estimator = self.model.named_estimators_["lgbm"]

        # Isotonic calibration — widens probability spread for high-risk patients
        calibrated = CalibratedClassifierCV(
            self.model, method="isotonic", cv=3
        )
        calibrated.fit(self.X_train, self.y_train)
        self.model = calibrated

        # Compute optimal F1 threshold
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        self.best_threshold = float(thresholds[f1_scores.argmax()])
        print(f"Optimal F1 threshold: {self.best_threshold:.3f}")


    def save_model(self):
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self.model, MODELS_DIR / "stacking_readmit_model.pkl")
        joblib.dump(self.encoder, MODELS_DIR / "encoder.pkl")
        joblib.dump(self.X_train, MODELS_DIR / "X_train.pkl")
        joblib.dump(self.X_test, MODELS_DIR / "X_test.pkl")
        joblib.dump(self.y_train, MODELS_DIR / "y_train.pkl")
        joblib.dump(self.y_test, MODELS_DIR / "y_test.pkl")
        joblib.dump(self.best_threshold, MODELS_DIR / "best_threshold.pkl")
        joblib.dump(self.lgbm_estimator, MODELS_DIR / "lgbm_estimator.pkl")
        joblib.dump(self.drop_features, MODELS_DIR / "drop_features.pkl")

    @classmethod
    def load_or_train(cls):
        """Load a saved model if available, otherwise train a new one."""
        try:
            print("Trying to load model...")
            instance = object.__new__(cls)
            instance.model = joblib.load(MODELS_DIR / "stacking_readmit_model.pkl")
            instance.encoder = joblib.load(MODELS_DIR / "encoder.pkl")
            instance.X_train = joblib.load(MODELS_DIR / "X_train.pkl")
            instance.X_test = joblib.load(MODELS_DIR / "X_test.pkl")
            instance.y_train = joblib.load(MODELS_DIR / "y_train.pkl")
            instance.y_test = joblib.load(MODELS_DIR / "y_test.pkl")
            try:
                instance.best_threshold = joblib.load(MODELS_DIR / "best_threshold.pkl")
            except FileNotFoundError:
                instance.best_threshold = 0.5
            try:
                instance.lgbm_estimator = joblib.load(MODELS_DIR / "lgbm_estimator.pkl")
            except FileNotFoundError:
                instance.lgbm_estimator = None
            try:
                instance.drop_features = joblib.load(MODELS_DIR / "drop_features.pkl")
            except FileNotFoundError:
                instance.drop_features = []
            return instance
        except FileNotFoundError:
            print("No model found, training...")
            instance = cls()
            instance.train()
            instance.save_model()
            return instance

    @staticmethod
    def _fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def get_model_stats(self):

        if self.X_test is None or self.y_test is None:
            return {"error": "No test set loaded"}

        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        metrics = {}

        # Core metrics
        auroc = roc_auc_score(self.y_test, y_proba)
        auprc = average_precision_score(self.y_test, y_proba)
        metrics["AUROC"] = f"{auroc:.3f}"
        metrics["AUPRC"] = f"{auprc:.3f}"

        # Find a good threshold using precision-recall trade-off
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_proba)

        # Computes F1 score (Harmonic mean), adds small 1e-8 to prevent division by 0 error
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[f1_scores.argmax()]
        metrics["best_f1_threshold"] = f"{best_threshold:.3f}"

        # Confusion matrix at that threshold
        y_pred = (y_proba >= best_threshold).astype(int)
        metrics["classification_report"] = classification_report(self.y_test, y_pred, target_names=['No Readmit', 'Readmit'])
        cm = confusion_matrix(self.y_test, y_pred)

        # Confusion matrix PNG
        fig_cm, ax_cm = plt.subplots()
        sb.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=["No Readmit", "Readmit"],
                   yticklabels=["No Readmit", "Readmit"], ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        metrics["confusion_matrix_png"] = self._fig_to_base64(fig_cm)

        # Calibration curve PNG
        prob_true, prob_pred = calibration_curve(self.y_test, y_proba, n_bins=10)
        fig_cal, ax_cal = plt.subplots()
        ax_cal.plot(prob_pred, prob_true, marker='o', label="Model")
        ax_cal.plot([0, 1], [0, 1], linestyle='--', label="Perfectly calibrated")
        ax_cal.set_xlabel("Predicted probability")
        ax_cal.set_ylabel("Observed frequency")
        ax_cal.set_title("Calibration Curve")
        ax_cal.legend()
        metrics["calibration_curve_png"] = self._fig_to_base64(fig_cal)
        print(metrics)
        return metrics

    def _prepare_input(self, patient_input):
        """Convert PatientInput into the encoded DataFrame the model expects."""
        raw_df = patient_input.to_raw_df()

        categorical_cols = ['race', 'discharge_group', 'admission_type', 'admission_source', 'diag_category']
        encoded_values = self.encoder.transform(raw_df[categorical_cols])
        encoded_cols = self.encoder.get_feature_names_out(categorical_cols)
        df_encoded = pd.DataFrame(encoded_values, columns=encoded_cols, index=raw_df.index)

        result = pd.concat([raw_df.drop(columns=categorical_cols), df_encoded], axis=1)
        if self.drop_features:
            result = result.drop(columns=[c for c in self.drop_features if c in result.columns])
        return result

    def predict(self, patient_input):
        """Accept a PatientInput, return risk score and contributing factors."""
        model_input = self._prepare_input(patient_input)

        proba = float(self.model.predict_proba(model_input)[:, 1][0])

        # 5-tier risk categorization using threshold-relative boundaries
        t = self.best_threshold
        if proba >= t * 2.5:
            risk_category = "very_high"
        elif proba >= t * 1.25:
            risk_category = "high"
        elif proba >= t * 0.75:
            risk_category = "moderate"
        elif proba >= t * 0.4:
            risk_category = "low"
        else:
            risk_category = "very_low"

        # Get feature contributions from the saved LightGBM base model
        lgbm = self.lgbm_estimator
        shap_values = lgbm.predict_proba(model_input, pred_contrib=True)[0]
        # pred_contrib returns one value per feature + a bias term at the end
        feature_names = model_input.columns.tolist()
        contributions = shap_values[:-1]  # drop bias

        # Build sorted list of contributing factors
        factor_indices = np.argsort(np.abs(contributions))[::-1]
        contributing_factors = []
        for idx in factor_indices:
            impact = float(contributions[idx])
            if abs(impact) < 0.001:
                break
            contributing_factors.append({
                "feature": feature_names[idx],
                "value": float(model_input.iloc[0, idx]),
                "impact": impact,
            })

        return {
            "risk_score": proba,
            "risk_category": risk_category,
            "contributing_factors": contributing_factors,
        }


    def get_metrics(self):
        if self.model is None:
            print("No model")
            return None
        y_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Core metrics
        auroc = roc_auc_score(self.y_test, y_proba)
        auprc = average_precision_score(self.y_test, y_proba)
        print(f"AUROC: {auroc:.3f}")
        print(f"AUPRC: {auprc:.3f}")

        # Find a good threshold using precision-recall trade-off
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[f1_scores.argmax()]
        print(f"Best threshold (by F1): {best_threshold:.3f}")

        # Confusion matrix at that threshold
        y_pred = (y_proba >= best_threshold).astype(int)
        print(classification_report(self.y_test, y_pred, target_names=['No Readmit', 'Readmit']))
        cm = confusion_matrix(self.y_test, y_pred)

        # Calibration curve
        prob_true, prob_pred = calibration_curve(self.y_test, y_proba, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed frequency')
        plt.title('Calibration Curve')

        return {
            "figure": plt.figimage(),
            "AUROC": auroc,
            "AUPRC": auprc,
        }

    @classmethod
    def train_and_test(cls):
        """Train a fresh model and print evaluation metrics."""
        instance = cls()
        print("Refining dataset...")
        refined_data = instance.refine_dataset()
        print(f"Dataset: {refined_data.shape[0]} rows, {refined_data.shape[1]} features")

        instance.get_split(refined_data)
        print(f"Train: {len(instance.X_train)}, Test: {len(instance.X_test)}")

        print("Training stacking classifier...")
        instance.train()
        print("Training complete.")

        # --- Evaluate on test set ---
        y_proba = instance.model.predict_proba(instance.X_test)[:, 1]

        auroc = roc_auc_score(instance.y_test, y_proba)
        auprc = average_precision_score(instance.y_test, y_proba)
        print(f"\nAUROC: {auroc:.3f}")
        print(f"AUPRC: {auprc:.3f}")

        precision, recall, thresholds = precision_recall_curve(instance.y_test, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold = thresholds[f1_scores.argmax()]
        print(f"Best threshold (by F1): {best_threshold:.3f}")

        y_pred = (y_proba >= best_threshold).astype(int)
        print(classification_report(instance.y_test, y_pred, target_names=['No Readmit', 'Readmit']))

        # Save model
        instance.save_model()
        print(f"Model saved to {MODELS_DIR}")

        return instance


    @classmethod
    def compare_approaches(cls):
        """Compare model approaches: XGBoost vs improved stacking vs current baseline."""
        instance = cls()
        refined_data = instance.refine_dataset()
        instance.get_split(refined_data)

        X_train, X_test = instance.X_train, instance.X_test
        y_train, y_test = instance.y_train, instance.y_test

        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        spw = neg / pos
        print(f"Class balance: {neg} neg, {pos} pos (ratio {spw:.1f}:1)\n")

        results = {}

        # ── Approach A: Single XGBoost ────────────────────────────────
        print("=" * 60)
        print("A: XGBoost + scale_pos_weight + sigmoid calibration")
        print("=" * 60)

        xgb_model = XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            eval_metric="aucpr",
        )
        xgb_model.fit(X_train, y_train)

        # Uncalibrated
        y_proba_a_raw = xgb_model.predict_proba(X_test)[:, 1]
        auroc_a_raw = roc_auc_score(y_test, y_proba_a_raw)
        auprc_a_raw = average_precision_score(y_test, y_proba_a_raw)
        print(f"  (uncalibrated)  AUROC={auroc_a_raw:.3f}  AUPRC={auprc_a_raw:.3f}")

        # Sigmoid calibration
        cal_a = CalibratedClassifierCV(xgb_model, method="sigmoid", cv=3)
        cal_a.fit(X_train, y_train)
        y_proba_a = cal_a.predict_proba(X_test)[:, 1]
        auroc_a = roc_auc_score(y_test, y_proba_a)
        auprc_a = average_precision_score(y_test, y_proba_a)
        prec_a, rec_a, thr_a = precision_recall_curve(y_test, y_proba_a)
        f1_a = 2 * (prec_a * rec_a) / (prec_a + rec_a + 1e-8)
        thresh_a = float(thr_a[f1_a.argmax()])
        y_pred_a = (y_proba_a >= thresh_a).astype(int)
        print(f"  (calibrated)    AUROC={auroc_a:.3f}  AUPRC={auprc_a:.3f}  Thresh={thresh_a:.3f}")
        print(classification_report(y_test, y_pred_a, target_names=["No Readmit", "Readmit"]))
        results["A"] = {"auroc": auroc_a, "auprc": auprc_a, "threshold": thresh_a}

        # ── Approach B: Stacking with class weights ───────────────────
        print("=" * 60)
        print("B: Stacking + class weights (no SMOTE) + sigmoid calibration")
        print("=" * 60)

        stacker_b = StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs", class_weight="balanced")),
                ("lgbm", LGBMClassifier(
                    n_estimators=300, num_leaves=31, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                )),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced")),
            ],
            final_estimator=LogisticRegression(),
            cv=5, stack_method="predict_proba", passthrough=False,
        )
        stacker_b.fit(X_train, y_train)

        cal_b = CalibratedClassifierCV(stacker_b, method="sigmoid", cv=3)
        cal_b.fit(X_train, y_train)
        y_proba_b = cal_b.predict_proba(X_test)[:, 1]
        auroc_b = roc_auc_score(y_test, y_proba_b)
        auprc_b = average_precision_score(y_test, y_proba_b)
        prec_b, rec_b, thr_b = precision_recall_curve(y_test, y_proba_b)
        f1_b = 2 * (prec_b * rec_b) / (prec_b + rec_b + 1e-8)
        thresh_b = float(thr_b[f1_b.argmax()])
        y_pred_b = (y_proba_b >= thresh_b).astype(int)
        print(f"  AUROC={auroc_b:.3f}  AUPRC={auprc_b:.3f}  Thresh={thresh_b:.3f}")
        print(classification_report(y_test, y_pred_b, target_names=["No Readmit", "Readmit"]))
        results["B"] = {"auroc": auroc_b, "auprc": auprc_b, "threshold": thresh_b}

        # ── Baseline: Current pipeline (SMOTE + stacking + isotonic) ──
        print("=" * 60)
        print("BASELINE: Current (SMOTE + stacking + isotonic calibration)")
        print("=" * 60)

        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        stacker_c = StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")),
                ("lgbm", LGBMClassifier(
                    n_estimators=300, num_leaves=31, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                )),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=12)),
            ],
            final_estimator=LogisticRegression(),
            cv=5, stack_method="predict_proba", passthrough=False,
        )
        stacker_c.fit(X_resampled, y_resampled)

        cal_c = CalibratedClassifierCV(stacker_c, method="isotonic", cv=3)
        cal_c.fit(X_train, y_train)
        y_proba_c = cal_c.predict_proba(X_test)[:, 1]
        auroc_c = roc_auc_score(y_test, y_proba_c)
        auprc_c = average_precision_score(y_test, y_proba_c)
        prec_c, rec_c, thr_c = precision_recall_curve(y_test, y_proba_c)
        f1_c = 2 * (prec_c * rec_c) / (prec_c + rec_c + 1e-8)
        thresh_c = float(thr_c[f1_c.argmax()])
        y_pred_c = (y_proba_c >= thresh_c).astype(int)
        print(f"  AUROC={auroc_c:.3f}  AUPRC={auprc_c:.3f}  Thresh={thresh_c:.3f}")
        print(classification_report(y_test, y_pred_c, target_names=["No Readmit", "Readmit"]))
        results["baseline"] = {"auroc": auroc_c, "auprc": auprc_c, "threshold": thresh_c}

        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Approach':<48} {'AUROC':>7} {'AUPRC':>7} {'Thresh':>7}")
        print("-" * 70)
        print(f"{'A: XGBoost + sigmoid':<48} {results['A']['auroc']:>7.3f} {results['A']['auprc']:>7.3f} {results['A']['threshold']:>7.3f}")
        print(f"{'B: Stacking + class weights + sigmoid':<48} {results['B']['auroc']:>7.3f} {results['B']['auprc']:>7.3f} {results['B']['threshold']:>7.3f}")
        print(f"{'Baseline: SMOTE + stacking + isotonic':<48} {results['baseline']['auroc']:>7.3f} {results['baseline']['auprc']:>7.3f} {results['baseline']['threshold']:>7.3f}")

        return results


    @classmethod
    def compare_xgb_vs_pruned(cls):
        """Compare 4-model stacking vs 3-model + feature pruning."""
        from imblearn.over_sampling import SMOTE

        instance = cls()
        refined_data = instance.refine_dataset()
        instance.get_split(refined_data)
        X_train, X_test = instance.X_train, instance.X_test
        y_train, y_test = instance.y_train, instance.y_test

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"SMOTE: {len(X_train)} -> {len(X_res)} samples\n")

        def evaluate(model, X_te, y_te, label):
            cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
            cal.fit(X_train, y_train)
            y_p = cal.predict_proba(X_te)[:, 1]
            auroc = roc_auc_score(y_te, y_p)
            auprc = average_precision_score(y_te, y_p)
            pr, rc, th = precision_recall_curve(y_te, y_p)
            f1 = 2 * (pr * rc) / (pr + rc + 1e-8)
            thresh = float(th[f1.argmax()])
            y_pred = (y_p >= thresh).astype(int)
            print(f"{label}")
            print(f"  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Thresh={thresh:.3f}")
            print(classification_report(y_te, y_pred, target_names=["No Readmit", "Readmit"]))
            return {"auroc": auroc, "auprc": auprc, "threshold": thresh}

        # ── A: 4-model stacking (current) ─────────────────────────────
        print("=" * 60)
        stacker_a = StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")),
                ("lgbm", LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8)),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=12)),
                ("xgb", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)),
            ],
            final_estimator=LogisticRegression(), cv=5,
            stack_method="predict_proba", passthrough=False,
        )
        stacker_a.fit(X_res, y_res)
        res_a = evaluate(stacker_a, X_test, y_test, "A: 4-model (LR+LGBM+RF+XGB)")

        # ── B: 3-model + feature pruning ──────────────────────────────
        # First train a LGBM to get feature importances
        print("=" * 60)
        print("Identifying low-importance features...")
        lgbm_scout = LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
        )
        lgbm_scout.fit(X_res, y_res)
        importances = lgbm_scout.feature_importances_
        feature_names = X_train.columns.tolist()

        # Sort by importance, find bottom 15%
        indexed = sorted(zip(feature_names, importances), key=lambda x: x[1])
        n_drop = max(1, len(indexed) // 7)  # ~15%
        drop_features = [name for name, imp in indexed[:n_drop]]
        print(f"Dropping {n_drop} features: {drop_features}")

        # Prune and retrain
        X_res_pruned = X_res.drop(columns=drop_features)
        X_train_pruned = X_train.drop(columns=drop_features)
        X_test_pruned = X_test.drop(columns=drop_features)

        stacker_b = StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")),
                ("lgbm", LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8)),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=12)),
            ],
            final_estimator=LogisticRegression(), cv=5,
            stack_method="predict_proba", passthrough=False,
        )
        stacker_b.fit(X_res_pruned, y_res)

        # Need to recalibrate with pruned train data
        cal_b = CalibratedClassifierCV(stacker_b, method="isotonic", cv=3)
        cal_b.fit(X_train_pruned, y_train)
        y_p_b = cal_b.predict_proba(X_test_pruned)[:, 1]
        auroc_b = roc_auc_score(y_test, y_p_b)
        auprc_b = average_precision_score(y_test, y_p_b)
        pr_b, rc_b, th_b = precision_recall_curve(y_test, y_p_b)
        f1_b = 2 * (pr_b * rc_b) / (pr_b + rc_b + 1e-8)
        thresh_b = float(th_b[f1_b.argmax()])
        y_pred_b = (y_p_b >= thresh_b).astype(int)
        print(f"\nB: 3-model pruned ({len(X_test_pruned.columns)} features)")
        print(f"  AUROC={auroc_b:.3f}  AUPRC={auprc_b:.3f}  Thresh={thresh_b:.3f}")
        print(classification_report(y_test, y_pred_b, target_names=["No Readmit", "Readmit"]))
        res_b = {"auroc": auroc_b, "auprc": auprc_b, "threshold": thresh_b}

        # ── C: 4-model + feature pruning ──────────────────────────────
        print("=" * 60)
        stacker_c = StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")),
                ("lgbm", LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8)),
                ("rf", RandomForestClassifier(n_estimators=200, max_depth=12)),
                ("xgb", XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, random_state=42)),
            ],
            final_estimator=LogisticRegression(), cv=5,
            stack_method="predict_proba", passthrough=False,
        )
        stacker_c.fit(X_res_pruned, y_res)

        cal_c = CalibratedClassifierCV(stacker_c, method="isotonic", cv=3)
        cal_c.fit(X_train_pruned, y_train)
        y_p_c = cal_c.predict_proba(X_test_pruned)[:, 1]
        auroc_c = roc_auc_score(y_test, y_p_c)
        auprc_c = average_precision_score(y_test, y_p_c)
        pr_c, rc_c, th_c = precision_recall_curve(y_test, y_p_c)
        f1_c = 2 * (pr_c * rc_c) / (pr_c + rc_c + 1e-8)
        thresh_c = float(th_c[f1_c.argmax()])
        y_pred_c = (y_p_c >= thresh_c).astype(int)
        print(f"C: 4-model pruned ({len(X_test_pruned.columns)} features)")
        print(f"  AUROC={auroc_c:.3f}  AUPRC={auprc_c:.3f}  Thresh={thresh_c:.3f}")
        print(classification_report(y_test, y_pred_c, target_names=["No Readmit", "Readmit"]))
        res_c = {"auroc": auroc_c, "auprc": auprc_c, "threshold": thresh_c}

        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Approach':<48} {'AUROC':>7} {'AUPRC':>7} {'Thresh':>7}")
        print("-" * 70)
        print(f"{'A: 4-model (LR+LGBM+RF+XGB)':<48} {res_a['auroc']:>7.3f} {res_a['auprc']:>7.3f} {res_a['threshold']:>7.3f}")
        print(f"{'B: 3-model pruned':<48} {res_b['auroc']:>7.3f} {res_b['auprc']:>7.3f} {res_b['threshold']:>7.3f}")
        print(f"{'C: 4-model pruned':<48} {res_c['auroc']:>7.3f} {res_c['auprc']:>7.3f} {res_c['threshold']:>7.3f}")
        print(f"\nDropped features: {drop_features}")


    @classmethod
    def optuna_tune(cls, n_trials=75):
        """Bayesian hyperparameter search over LGBM + RF params in the stacking ensemble."""
        import optuna
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from imblearn.over_sampling import SMOTE

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        instance = cls()
        refined_data = instance.refine_dataset()
        instance.get_split(refined_data)
        X_train, X_test = instance.X_train, instance.X_test
        y_train, y_test = instance.y_train, instance.y_test

        # Feature pruning (same as production train)
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        lgbm_scout = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8)
        lgbm_scout.fit(X_res, y_res)
        importances = lgbm_scout.feature_importances_
        indexed = sorted(zip(X_train.columns.tolist(), importances), key=lambda x: x[1])
        n_drop = max(1, len(indexed) // 7)
        drop_features = [name for name, _ in indexed[:n_drop]]
        print(f"Pruning {n_drop} features: {drop_features}")

        X_res = X_res.drop(columns=drop_features)
        X_train_p = X_train.drop(columns=drop_features)
        X_test_p = X_test.drop(columns=drop_features)

        print(f"Tuning on {len(X_res)} SMOTE samples, {len(X_res.columns)} features")
        print(f"Running {n_trials} Optuna trials...\n")

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        def objective(trial):
            lgbm_params = {
                'n_estimators': trial.suggest_int('lgbm_n_estimators', 200, 800),
                'num_leaves': trial.suggest_int('lgbm_num_leaves', 15, 63),
                'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('lgbm_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('lgbm_colsample', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('lgbm_min_child', 20, 100),
                'reg_alpha': trial.suggest_float('lgbm_reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('lgbm_reg_lambda', 1e-3, 10, log=True),
                'verbose': -1,
            }
            rf_params = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 400),
                'max_depth': trial.suggest_int('rf_max_depth', 8, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            }

            stacker = StackingClassifier(
                estimators=[
                    ('lr', LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs')),
                    ('lgbm', LGBMClassifier(**lgbm_params)),
                    ('rf', RandomForestClassifier(**rf_params)),
                ],
                final_estimator=LogisticRegression(),
                cv=3, stack_method='predict_proba', passthrough=False,
            )
            scores = cross_val_score(stacker, X_res, y_res, cv=cv, scoring='roc_auc', n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction='maximize', study_name='stacking_tune')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f"\nBest AUROC (CV on SMOTE data): {study.best_value:.4f}")
        print(f"Best params: {study.best_params}\n")

        # Retrain with best params on full SMOTE data
        bp = study.best_params
        best_stacker = StackingClassifier(
            estimators=[
                ('lr', LogisticRegression(C=1.0, max_iter=3000, solver='lbfgs')),
                ('lgbm', LGBMClassifier(
                    n_estimators=bp['lgbm_n_estimators'], num_leaves=bp['lgbm_num_leaves'],
                    learning_rate=bp['lgbm_learning_rate'], subsample=bp['lgbm_subsample'],
                    colsample_bytree=bp['lgbm_colsample'], min_child_samples=bp['lgbm_min_child'],
                    reg_alpha=bp['lgbm_reg_alpha'], reg_lambda=bp['lgbm_reg_lambda'], verbose=-1,
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=bp['rf_n_estimators'], max_depth=bp['rf_max_depth'],
                    min_samples_leaf=bp['rf_min_samples_leaf'],
                )),
            ],
            final_estimator=LogisticRegression(),
            cv=5, stack_method='predict_proba', passthrough=False,
        )
        best_stacker.fit(X_res, y_res)

        # Isotonic calibration
        cal = CalibratedClassifierCV(best_stacker, method='isotonic', cv=3)
        cal.fit(X_train_p, y_train)

        y_proba = cal.predict_proba(X_test_p)[:, 1]
        auroc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        pr, rc, th = precision_recall_curve(y_test, y_proba)
        f1 = 2 * (pr * rc) / (pr + rc + 1e-8)
        thresh = float(th[f1.argmax()])
        y_pred = (y_proba >= thresh).astype(int)

        print("=" * 60)
        print("OPTUNA-TUNED MODEL (test set)")
        print(f"  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Thresh={thresh:.3f}")
        print(classification_report(y_test, y_pred, target_names=["No Readmit", "Readmit"]))
        print(f"\nBaseline was: AUROC=0.670  AUPRC=0.223  Thresh=0.149")


if __name__ == "__main__":
    PredictionModel.train_and_test()