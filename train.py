from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/telco_churn.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
SEED = 42

COST_FP = 50    # wasted retention offer
COST_FN = 184   # missed churner: 0.30 * 780 - 50

# 1. Load and clean
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Blank TotalCharges:", df["TotalCharges"].isna().sum(),
      "| their tenure values:", df.loc[df["TotalCharges"].isna(), "tenure"].unique())
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])

addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
df[addon_cols + ["MultipleLines"]] = df[addon_cols + ["MultipleLines"]].replace(
    {"No internet service": "No", "No phone service": "No"})

# 2. Feature engineering
df["num_addons"] = (df[addon_cols] == "Yes").sum(axis=1)
df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                            labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                "SeniorCitizen", "num_addons"]
categorical_cols = [c for c in df.columns
                    if c not in numeric_cols + ["Churn"]]

# 3. Split
X = df.drop(columns=["Churn"])
y = df["Churn"].to_numpy()
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Train: {len(y_train)} rows, churn rate {y_train.mean():.3f}")
print(f"Test:  {len(y_test)} rows, churn rate {y_test.mean():.3f}")

# 4. Preprocess (fit on train only)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                          sparse_output=False), categorical_cols),
])
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
feature_names = list(preprocessor.get_feature_names_out())
print("Features after preprocessing:", X_train.shape[1])

# 5. Baseline: predict nobody churns
baseline_acc = (y_test == 0).mean()
baseline_cost = y_test.sum() * COST_FN
print(f"Baseline accuracy: {baseline_acc:.3f}  "
      f"| baseline cost on test set: EUR {baseline_cost:,}")

# 6. Save for later stages
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
X_test_raw.assign(Churn=y_test).to_csv(MODELS_DIR / "test_set.csv", index=False)
with open(MODELS_DIR / "config.json", "w") as f:
    json.dump({"cost_fp": COST_FP, "cost_fn": COST_FN,
               "feature_names": feature_names,
               "baseline_accuracy": float(baseline_acc),
               "baseline_cost": int(baseline_cost)}, f, indent=2)
print("Part 1 done.")
