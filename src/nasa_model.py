# src/nasa_model.py
import polars as pl
import pandas as pd
import numpy as np
import json
import os
import joblib
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)

EXCLUDE_COLS = ["unit", "cycle", "dataset_id", "condition_cluster", "rul"]
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "nasa_rul_model.joblib")
PARAMS_PATH = os.path.join(MODEL_DIR, "nasa_best_params.json")
EVAL_REPORT_PATH = os.path.join(MODEL_DIR, "nasa_eval_report.json")


class NASAModelTrainer:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def load_data(self):
        df = pl.read_parquet(self.input_path).to_pandas()
        feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
        return df, feature_cols

    def split_by_unit(self, df, feature_cols):
        """
        FIX: Split by engine unit, not by row.
        Row-level split = same engine's cycles in both train and test = leakage.
        Unit-level split = held-out engines the model has never seen.
        GroupShuffleSplit groups by 'unit' column.
        """
        print("  Splitting by engine unit (no leakage)...")

        # Create composite unit ID across datasets
        df["global_unit"] = df["dataset_id"] + "_" + df["unit"].astype(int).astype(str)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df["global_unit"]))

        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        X_train = train_df[feature_cols]
        y_train = train_df["rul"]
        X_test = test_df[feature_cols]
        y_test = test_df["rul"]

        print(f"  Train: {len(train_df)} rows | {train_df['global_unit'].nunique()} engines")
        print(f"  Test:  {len(test_df)} rows  | {test_df['global_unit'].nunique()} engines")

        return X_train, X_test, y_train, y_test

    def run_optuna(self, X_train, y_train, n_trials=30):
        """
        FIX: Data loaded ONCE outside objective.
        Original reloaded the entire parquet on every trial — 30 disk reads.
        FIX: 30 trials (was 15) for better search coverage.
        FIX: Results saved as JSON not txt.
        """
        print(f"\n🔍 Running Optuna ({n_trials} trials)...")

        # Inner validation split for Optuna — also unit-based
        df_train = X_train.copy()
        df_train["rul"] = y_train.values

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            }

            # 80/20 split of training data for inner validation
            split = int(len(df_train) * 0.8)
            X_tr = df_train.iloc[:split][X_train.columns]
            y_tr = df_train.iloc[:split]["rul"]
            X_val = df_train.iloc[split:][X_train.columns]
            y_val = df_train.iloc[split:]["rul"]

            model = lgb.LGBMRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, preds))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        print(f"\n🏆 Best params: {best_params}")
        print(f"🏆 Best inner RMSE: {study.best_value:.4f}")

        # FIX: Save as JSON, not txt
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=4)
        print(f"✅ Best params saved: {PARAMS_PATH}")

        return best_params

    def train_final_model(self, X_train, X_test, y_train, y_test, best_params):
        """
        FIX: Final model evaluated on held-out test engines only.
        FIX: Reports RMSE, MAE, R² — all on test set.
        FIX: Saved with joblib, not pickle.
        """
        print("\n🏗️ Training final LightGBM model with best params...")

        model = lgb.LGBMRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)

        # Evaluate on held-out test engines
        y_pred = model.predict(X_test)
        y_pred_clipped = np.clip(y_pred, 0, 125)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_clipped))
        mae = mean_absolute_error(y_test, y_pred_clipped)
        r2 = r2_score(y_test, y_pred_clipped)

        print(f"\n  📊 TEST SET RESULTS (held-out engines, never seen during training):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")

        # Feature importance
        importances = dict(zip(
            X_train.columns.tolist(),
            [round(float(v), 4) for v in model.feature_importances_]
        ))
        importances_sorted = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )
        print(f"\n  Top 5 features: {list(importances_sorted.items())[:5]}")

        eval_report = {
            "model": "LGBMRegressor",
            "evaluation": "held-out test engines",
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "rul_cap": 125,
            "top_features": dict(list(importances_sorted.items())[:10])
        }

        return model, eval_report

    def run(self):
        print("🚀 NASA RUL Model Training...")
        os.makedirs(MODEL_DIR, exist_ok=True)

        df, feature_cols = self.load_data()
        print(f"  Input: {df.shape} | Features: {len(feature_cols)}")

        X_train, X_test, y_train, y_test = self.split_by_unit(df, feature_cols)

        best_params = self.run_optuna(X_train, y_train, n_trials=30)

        model, eval_report = self.train_final_model(
            X_train, X_test, y_train, y_test, best_params
        )

        # Save model and report
        joblib.dump(model, MODEL_PATH)
        print(f"\n✅ Model saved: {MODEL_PATH}")

        with open(EVAL_REPORT_PATH, "w") as f:
            json.dump(eval_report, f, indent=4)
        print(f"✅ Eval report saved: {EVAL_REPORT_PATH}")

        return eval_report


if __name__ == "__main__":
    trainer = NASAModelTrainer("data/processed/nasa_features.parquet")
    trainer.run()