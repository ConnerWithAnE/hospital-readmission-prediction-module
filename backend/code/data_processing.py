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
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sb
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, classification_report, \
    confusion_matrix

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from .cci import add_comorbidities_to_dataframe
from .data_maps import keep, admission_type_map, discharge_disposition_map, age_map, admission_source_map


class PredictionModel:

    def __init__(self):
        self.diabetes_130_us_hospitals = fetch_ucirepo(id=296)
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def refine_dataset(self):
        df = self.diabetes_130_us_hospitals.data.features.copy()

        df["readmitted"] = self.diabetes_130_us_hospitals.data.targets["readmitted"]

        df = add_comorbidities_to_dataframe(df)


        df = df[keep].copy()

        df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).fillna(-1)

        # Map admission type by importance
        df['admission_type'] = df['admission_type_id'].map(admission_type_map).fillna("unknown")
        df['admission_source'] = df['admission_source_id'].map(admission_source_map).fillna("unknown")
        df['discharge_group'] = df['discharge_disposition_id'].map(discharge_disposition_map).fillna("unknown")

        # Drop birth as they are different compared to an emergency or other issue (frequent visits)
        df = df[df["admission_type"] != "birth"]
        #df = df[df['discharge_group'] != "hospice_death"]


        df['age'] = df['age'].map(age_map)
        df['race'] = df['race'].fillna("unknown")

        # Interaction: prior utilization intensity
        df["total_prior_visits"] = df["number_inpatient"] + df["number_outpatient"] + df["number_emergency"]
        df["has_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)

        # Medication burden relative to stay
        df["meds_per_day"] = df["num_medications"] / (df["time_in_hospital"] + 1)

        # Diagnosis complexity relative to stay
        stay = np.where(df["time_in_hospital"] == 0, 1, df["time_in_hospital"])
        df["diag_per_day"] = np.where(
            df["number_diagnoses"] == 0,
            0,
            df["number_diagnoses"] / stay
        )

        # Age-adjusted CCI (original Charlson adds 1 point per decade over 40)
        df["age_adjusted_cci"] = df["cci_score"] + np.maximum(0, df["age"] - 4)

        # Age x utilization interactions (frailty proxies)
        df["age_x_inpatient"] = df["age"] * df["number_inpatient"]
        df["age_x_medications"] = df["age"] * df["num_medications"]
        df["age_x_cci"] = df["age"] * df["cci_score"]

        # Non-linear age thresholds
        df["age_over_70"] = (df["age"] >= 7).astype(int)
        df["age_over_80"] = (df["age"] >= 8).astype(int)


        encoded_values = self.encoder.fit_transform(df[['race', 'discharge_group', 'admission_type', 'admission_source']])
        new_cols = self.encoder.get_feature_names_out(['race', 'discharge_group', 'admission_type', 'admission_source'])

        df_encoded = pd.DataFrame(encoded_values, columns=new_cols, index=df.index)
        data_final = pd.concat([df.drop(columns=['race', 'discharge_disposition_id', 'discharge_group',
                                                 'admission_type', 'admission_source', 'admission_type_id', 'admission_source_id']), df_encoded],
                               axis=1)
        return data_final

    def get_split(self, ds):
        X = ds.drop(columns=["readmitted"])
        y = ds["readmitted"].map({"<30": 1, ">30": 1, "NO": 0})
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    def train(self):
        refined_data = self.refine_dataset()
        self.get_split(refined_data)

        base_models = [
            ('lr', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')),
            ('lgbm', LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, is_unbalance=True
            )),
            ('rf', RandomForestClassifier(
                n_estimators=200, max_depth=12, class_weight='balanced'
            ))
        ]

        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=5,  # 5-fold for out-of-fold predictions
            stack_method='predict_proba',  # use probabilities, not hard labels
            passthrough=False  # only base model outputs go to meta-learner
        )

        # Using non scaled data as LR is the only model that benefits from it.
        self.model.fit(self.X_train, self.y_train)


    def save_model(self):
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self.model, MODELS_DIR / "stacking_readmit_model.pkl")
        joblib.dump(self.encoder, MODELS_DIR / "encoder.pkl")
        joblib.dump(self.X_train, MODELS_DIR / "X_train.pkl")
        joblib.dump(self.X_test, MODELS_DIR / "X_test.pkl")
        joblib.dump(self.y_train, MODELS_DIR / "y_train.pkl")
        joblib.dump(self.y_test, MODELS_DIR / "y_test.pkl")

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

        categorical_cols = ['race', 'discharge_group', 'admission_type', 'admission_source']
        encoded_values = self.encoder.transform(raw_df[categorical_cols])
        encoded_cols = self.encoder.get_feature_names_out(categorical_cols)
        df_encoded = pd.DataFrame(encoded_values, columns=encoded_cols, index=raw_df.index)

        return pd.concat([raw_df.drop(columns=categorical_cols), df_encoded], axis=1)

    def predict(self, patient_input):
        """Accept a PatientInput, return risk score and contributing factors."""
        model_input = self._prepare_input(patient_input)

        proba = float(self.model.predict_proba(model_input)[:, 1][0])

        if proba >= 0.6:
            risk_category = "high"
        elif proba >= 0.4:
            risk_category = "moderate"
        else:
            risk_category = "low"

        # Get feature contributions from the LightGBM base model
        lgbm = self.model.named_estimators_["lgbm"]
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


if __name__ == "__main__":
    PredictionModel.train_and_test()