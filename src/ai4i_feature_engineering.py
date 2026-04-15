# src/ai4i_feature_engineering.py
import polars as pl
import numpy as np
from scipy.signal import savgol_filter
import os

# Columns that must never be scaled or engineered over
TARGET_COLS = ["failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLS = ["prod_type"]  # ordinal, not continuous — exclude from scaling

EXCLUDE_FROM_SCALING = TARGET_COLS + ID_COLS

# Sensor columns that smoothing applies to
SENSOR_COLS = ["air_temp_c", "proc_temp_c", "rpm", "torque", "tool_wear", "power_w"]


class AI4IEngineer:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self) -> pl.DataFrame:
        return pl.read_parquet(self.input_path)

    def compute_physics_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute domain-grounded interaction features.

        temp_delta: Process temp minus air temp. HDF failure fires when
                    this drops below 8.6K — so this is a direct causal signal.

        stress_index: Normalized torque * normalized RPM. Kept as a ratio,
                      NOT as a duplicate of power_w. power_w is absolute watts,
                      stress_index is a dimensionless load ratio [0,1].
                      They carry different information.

        overstrain_margin: How close is the machine to OSF threshold?
                           OSF fires when tool_wear * torque exceeds
                           11000 (L), 12000 (M), 13000 (H).
                           Margin < 0 means failure imminent.
        """
        osf_threshold = (
            pl.when(pl.col("prod_type") == 0).then(11000)
            .when(pl.col("prod_type") == 1).then(12000)
            .otherwise(13000)
        )

        return df.with_columns([
            (pl.col("proc_temp_c") - pl.col("air_temp_c")).alias("temp_delta"),

            (
                (pl.col("torque") / pl.col("torque").max()) *
                (pl.col("rpm") / pl.col("rpm").max())
            ).alias("stress_index"),

            # Positive = safe, negative = OSF imminent
            (osf_threshold - (pl.col("tool_wear") * pl.col("torque"))).alias("overstrain_margin"),
        ])

    def apply_smoothing(self, df: pl.DataFrame, window: int = 11) -> pl.DataFrame:
        """
        FIX: Savgol smoothing was defined in original BaseEngineer but never
        called for AI4I. Now explicitly applied to all sensor columns.
        Adds _smooth variants — keeps originals for causal analysis layer.
        window=11, poly=3 is standard for industrial sensor smoothing.
        """
        def _savgol(x: pl.Series) -> pl.Series:
            arr = x.to_numpy()
            # If series shorter than window, skip smoothing
            if len(arr) < window:
                return x
            return pl.Series(savgol_filter(arr, window, 3))

        smooth_cols = []
        for col in SENSOR_COLS:
            if col in df.columns:
                smooth_cols.append(
                    pl.col(col).map_batches(
                        lambda x: _savgol(x), return_dtype=pl.Float64
                    ).alias(f"{col}_smooth")
                )

        return df.with_columns(smooth_cols)

    def apply_min_max_scaling(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        FIX: Scale only continuous feature columns.
        Explicitly excludes targets, prod_type (ordinal), and
        smooth variants are scaled separately since they're new columns.
        Division guard: if max==min (constant column), result is 0.
        """
        cols_to_scale = [
            col for col in df.columns
            if col not in EXCLUDE_FROM_SCALING
            and df[col].dtype in [pl.Float64, pl.Int64]
        ]

        return df.with_columns([
            pl.when(pl.col(c).max() == pl.col(c).min())
            .then(pl.lit(0.0))
            .otherwise(
                (pl.col(c) - pl.col(c).min()) /
                (pl.col(c).max() - pl.col(c).min())
            )
            .alias(c)
            for c in cols_to_scale
        ])

    def validate(self, df: pl.DataFrame):
        assert "temp_delta" in df.columns, "temp_delta missing"
        assert "overstrain_margin" in df.columns, "overstrain_margin missing"
        assert "air_temp_c_smooth" in df.columns, "smoothing not applied"
        assert all(c in df.columns for c in TARGET_COLS), "targets missing"
        # Targets must NOT be scaled — verify they're still integers
        for c in TARGET_COLS:
            assert df[c].dtype == pl.Int64, f"{c} was scaled — should stay Int64"
        print(f"✅ AI4I Feature validation passed. Shape: {df.shape}")

    def run(self):
        print("🚀 AI4I Feature Engineering...")

        df = self.load_data()
        print(f"  Input shape: {df.shape}")

        df = self.compute_physics_features(df)
        print(f"  After physics features: {df.shape}")

        df = self.apply_smoothing(df)
        print(f"  After smoothing: {df.shape}")

        df = self.apply_min_max_scaling(df)
        print(f"  After scaling: {df.shape}")

        self.validate(df)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.write_parquet(self.output_path)
        print(f"✅ Saved to: {self.output_path}")
        print(f"📊 Final columns: {df.columns}")
        return df


if __name__ == "__main__":
    engineer = AI4IEngineer(
        input_path="data/processed/ai4i_cleaned.parquet",
        output_path="data/processed/ai4i_features.parquet"
    )
    engineer.run()