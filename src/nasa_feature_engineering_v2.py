# src/nasa_feature_engineering.py
import polars as pl
import numpy as np
import os

# Must match Layer 1 exactly
SENSORS_TO_DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
INFORMATIVE_SENSORS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in SENSORS_TO_DROP]

# Columns never touched by scaling
EXCLUDE_FROM_SCALING = ["unit", "cycle", "dataset_id", "condition_cluster", "rul"]

# Sensors with strongest degradation signal in CMAPSS literature
# Used for interaction and velocity features
PRIMARY_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]


class NASAEngineer:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def compute_physics_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Physics-grounded interaction features based on CMAPSS sensor definitions:

        thermal_stress:       s11 (temperature) * s15 (pressure ratio)
                              High temp + high pressure = accelerated degradation

        pressure_heat_index:  s4 (total pressure) * s11 (temp)
                              Compressor thermal load proxy

        efficiency_proxy:     s9 (physical fan speed) / (s8 (Mach) + 1e-6)
                              Thrust efficiency — drops as engine degrades

        compressor_delta:     s3 (HPC outlet temp) - s2 (LPC outlet temp)
                              Temperature rise across compressor stages —
                              rises as HPC degrades (primary fault mode)
        """
        return df.with_columns([
            (pl.col("s11") * pl.col("s15")).alias("thermal_stress"),
            (pl.col("s4") * pl.col("s11")).alias("pressure_heat_index"),
            (pl.col("s9") / (pl.col("s8") + 1e-6)).alias("efficiency_proxy"),
            (pl.col("s3") - pl.col("s2")).alias("compressor_delta"),
        ])

    def compute_rolling_features(self, df: pl.DataFrame, window: int = 10) -> pl.DataFrame:
        """
        FIX: Original code only computed velocity/trend for 3 sensors (s4, s11, s15).
        Now applied to all PRIMARY_SENSORS.

        velocity: cycle-over-cycle change — detects acceleration of drift
        trend:    rolling mean over window — smooths noise, shows direction

        Both computed per unit (over("unit")) so engine boundaries are respected.
        """
        new_cols = []
        for col in PRIMARY_SENSORS:
            if col not in df.columns:
                continue
            new_cols.append(
                (pl.col(col) - pl.col(col).shift(1))
                .over("unit")
                .alias(f"{col}_velocity")
            )
            new_cols.append(
                pl.col(col)
                .rolling_mean(window)
                .over("unit")
                .alias(f"{col}_trend")
            )

        return df.with_columns(new_cols)

    def clip_rul(self, df: pl.DataFrame, cap: int = 125) -> pl.DataFrame:
        """
        FIX: RUL clipping was silently applied in the model layer with no
        documentation. Moving it here so it's explicit and consistent
        across optimization and training.

        Cap at 125: early cycles have RUL=300+ but the model only needs to
        predict near-failure behavior. Capping reduces the regression target
        range and improves model focus on the critical degradation zone.
        This is standard practice in CMAPSS benchmarks.
        """
        return df.with_columns(
            pl.col("rul").clip(upper_bound=cap).alias("rul")
        )

    def apply_z_score_scaling(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        FIX: Original used global z-score. Now per-condition z-score was already
        applied in Layer 1 for raw sensors. Here we standardize the newly
        engineered features (velocities, trends, interactions) globally —
        these are already condition-relative by construction since they're
        derived from condition-normalized sensors.

        Null handling: velocity and trend produce nulls at series start
        (shift(1) and rolling_mean). We fill with 0 — meaning
        'no change detected yet', which is physically correct for cycle 1.
        """
        # Fill nulls from velocity/trend before scaling
        df = df.with_columns([
            pl.col(c).fill_null(0.0)
            for c in df.columns
            if c not in EXCLUDE_FROM_SCALING and df[c].dtype == pl.Float64
        ])

        cols_to_scale = [
            c for c in df.columns
            if c not in EXCLUDE_FROM_SCALING
            and df[c].dtype == pl.Float64
            # Only scale engineered features — raw sensors already normalized in Layer 1
            and any(tag in c for tag in ["velocity", "trend", "thermal", "pressure", "efficiency", "compressor"])
        ]

        return df.with_columns([
            ((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-6)).alias(c)
            for c in cols_to_scale
        ])

    def validate(self, df: pl.DataFrame):
        assert "thermal_stress" in df.columns, "thermal_stress missing"
        assert "compressor_delta" in df.columns, "compressor_delta missing"
        assert "s2_velocity" in df.columns, "velocity features missing"
        assert "s2_trend" in df.columns, "trend features missing"
        assert df["rul"].max() <= 125, "RUL not clipped"
        assert df.filter(pl.col("rul") < 0).height == 0, "Negative RUL found"
        null_counts = df.null_count().sum_horizontal().sum()
        assert null_counts == 0, f"Nulls found: {null_counts}"
        print(f"✅ NASA Feature validation passed. Shape: {df.shape}")

    def run(self):
        print("🚀 NASA Feature Engineering...")

        df = pl.read_parquet(self.input_path)
        print(f"  Input shape: {df.shape}")

        df = self.compute_physics_features(df)
        print(f"  After physics features: {df.shape}")

        df = self.compute_rolling_features(df)
        print(f"  After rolling features: {df.shape}")

        df = self.clip_rul(df)
        print(f"  RUL clipped at 125. Max RUL: {df['rul'].max()}")

        df = self.apply_z_score_scaling(df)
        print(f"  After scaling engineered features: {df.shape}")

        self.validate(df)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.write_parquet(self.output_path)
        print(f"✅ Saved to: {self.output_path}")
        print(f"📊 Final columns ({len(df.columns)}): {df.columns}")
        return df


if __name__ == "__main__":
    engineer = NASAEngineer(
        input_path="data/processed/nasa_master.parquet",
        output_path="data/processed/nasa_features.parquet"
    )
    engineer.run()