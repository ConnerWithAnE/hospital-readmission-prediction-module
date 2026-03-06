import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv("../datasets/CleanedDataSet_jh.csv")

#dataset.hist(bins=50, figsize=(10,10))
#plt.show()

def create_datasets(ds):
    # X = everything except the label column

    # Create date features

    # Hospital ward
    ds = pd.get_dummies(ds, columns=['hospital_ward'], prefix="hospital_ward")
    # Referral Department
    ds = pd.get_dummies(ds, columns=['department_referral'], prefix="referred_from")
    # Referral Department
    ds = pd.get_dummies(ds, columns=['doctor_specialty'], prefix="doctor_specialty")
    # Referral Department
    ds = pd.get_dummies(ds, columns=['patient_disease'], prefix="disease")
    # Discharge Status
    ds = pd.get_dummies(ds, columns=['discharge_status'], prefix="discharge_status")

    ds["time_slot"] = pd.to_datetime(ds["time_slot"], format="%I:%M:%S %p")

    ds["admission_hour"] = ds["time_slot"].dt.hour  # 0-23
    ds["admission_minute"] = ds["time_slot"].dt.minute

    ds = ds.drop(columns=["time_slot"])

    ds = ds.drop(columns=['patient_last_name'])
    ds = ds.drop(columns=['doctor_name'])
    ds = ds.drop(columns=['patient_checkin_date'])
    ds = ds.drop(columns=['patient_checkout_date'])

    # Gender
    ds["patient_gender"] = ds["patient_gender"].map({"Male": 1, "Female": 0}).fillna(-1)

    # Patient Race
    ds = pd.get_dummies(ds, columns=['patient_race'], prefix="patient_race")

    print(ds.head())

    ds = ds.drop(columns=["Admission_date"])

    ds.to_csv("../datasets/processed_data.csv", index=False)

    X = ds.drop(columns=["readmission"]).values

    # y = just the label column
    y = ds["readmission"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #model = LogisticRegression(max_iter=1000)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring="f1",  # Optimize for F1 score
        n_jobs=-1  # Use all CPU cores
    )

    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return

if __name__ == "__main__":
    create_datasets(dataset)

