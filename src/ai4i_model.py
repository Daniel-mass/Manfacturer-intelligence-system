# src/ai4i_model.py
import polars as pl
import numpy as np
import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)

FEATURE_COLS = [
    "rpm", "torque", "tool_wear", "air_temp_c", "proc_temp_c", "power_w",
    "temp_delta", "stress_index", "overstrain_margin",
    "air_temp_c_smooth", "proc_temp_c_smooth", "rpm_smooth",
    "torque_smooth", "tool_wear_smooth", "power_w_smooth"
]

TARGET_COL = "failure"
EXCLUDE_COLS = ["failure", "TWF", "HDF", "PWF", "OSF", "RNF", "prod_type"]

MODEL_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "ai4i_classifier.joblib")
ISO_FOREST_PATH = os.path.join(MODEL_DIR, "ai4i_iso_forest.joblib")
EVAL_REPORT_PATH = os.path.join(MODEL_DIR, "ai4i_eval_report.json")


class AI4IModelTrainer:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def load_data(self):
        df = pl.read_parquet(self.input_path).to_pandas()
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns from Layer 2: {missing}")
        return df

    def train_classifier(self, df):
        print("\n🌲 Training AI4I Random Forest Classifier...")

        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        print(f"  Class distribution — Normal: {(y==0).sum()} | Failure: {(y==1).sum()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        print(f"  Train set — Normal: {neg} | Failure: {pos}")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        print(f"\n  📊 TEST SET RESULTS (held-out, never seen during training):")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")

        cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
        print(f"  5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        importances = dict(zip(
            FEATURE_COLS,
            [round(float(v), 4) for v in model.feature_importances_]
        ))
        importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        print(f"\n  Top 5 features: {list(importances_sorted.items())[:5]}")

        eval_report = {
            "model": "RandomForestClassifier",
            "test_size": 0.2,
            "f1_score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "roc_auc": round(auc, 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": cm,
            "feature_importances": importances_sorted,
            "class_distribution": {"normal": int(neg), "failure": int(pos)}
        }

        return model, eval_report

    def train_isolation_forest(self, df):
        print("\n🌲 Training Isolation Forest on NORMAL data only...")

        normal_df = df[df[TARGET_COL] == 0]
        print(f"  Training on {len(normal_df)} normal samples (failure=0 only)")

        X_normal = normal_df[FEATURE_COLS]

        iso = IsolationForest(
            contamination=0.034,
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        iso.fit(X_normal)

        X_all = df[FEATURE_COLS]
        y_all = df[TARGET_COL]
        preds = iso.predict(X_all)
        iso_flags = (preds == -1).astype(int)

        iso_f1 = f1_score(y_all, iso_flags)
        iso_recall = recall_score(y_all, iso_flags)
        print(f"  Isolation Forest vs ground truth — F1: {iso_f1:.4f} | Recall: {iso_recall:.4f}")
        print(f"  (Lower than RF is expected — this is unsupervised)")

        return iso

    def run(self):
        print("🚀 AI4I Model Training...")
        os.makedirs(MODEL_DIR, exist_ok=True)

        df = self.load_data()

        classifier, eval_report = self.train_classifier(df)
        iso_forest = self.train_isolation_forest(df)

        joblib.dump(classifier, CLASSIFIER_PATH)
        joblib.dump(iso_forest, ISO_FOREST_PATH)
        print(f"\n✅ Classifier saved: {CLASSIFIER_PATH}")
        print(f"✅ Isolation Forest saved: {ISO_FOREST_PATH}")

        with open(EVAL_REPORT_PATH, "w") as f:
            json.dump(eval_report, f, indent=4)
        print(f"✅ Eval report saved: {EVAL_REPORT_PATH}")

        # Save enriched dataset for Layer 4 (causal analysis)
        # This block is INSIDE run(), BEFORE return
        print("\n💾 Saving enriched dataset for causal analysis...")
        full_df = self.load_data()
        X_all = full_df[FEATURE_COLS]

        full_df["failure_prob"] = classifier.predict_proba(X_all)[:, 1]
        full_df["failure_pred"] = classifier.predict(X_all)
        full_df["anomaly_score"] = iso_forest.decision_function(X_all)
        full_df["is_anomaly"] = (iso_forest.predict(X_all) == -1).astype(int)

        results_path = "data/processed/ai4i_results.parquet"
        pl.from_pandas(full_df).write_parquet(results_path)
        print(f"✅ Enriched dataset saved: {results_path}")

        return eval_report


if __name__ == "__main__":
    trainer = AI4IModelTrainer("data/processed/ai4i_features.parquet")
    trainer.run()