#src/data_processing_ai4i.py
import polars as pl
import numpy as np
import os

class AI4IProcessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

        self.column_mapping = {
            "Type": "prod_type",
            "Air temperature [K]": "air_temp_k",
            "Process temperature [K]": "proc_temp_k",
            "Rotational speed [rpm]": "rpm",
            "Torque [Nm]": "torque",
            "Tool wear [min]": "tool_wear",
            "Machine failure": "failure"
        }

        self.type_mapping = {"L": 0, "M": 1, "H": 2}

        # Both binary and multi-label targets kept.
        # Downstream code decides which to use.
        self.target_cols = ["failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

    def load_data(self) -> pl.DataFrame:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Raw data not found at {self.input_path}")
        return pl.read_csv(self.input_path)

    def clean_and_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        df = (
            df.rename(self.column_mapping)
            .drop(["UDI", "Product ID"])
            .with_columns([
                pl.col("prod_type").replace(self.type_mapping).cast(pl.Int64),
                (pl.col("air_temp_k") - 273.15).alias("air_temp_c"),
                (pl.col("proc_temp_k") - 273.15).alias("proc_temp_c"),
                (pl.col("torque") * (pl.col("rpm") * 2 * np.pi / 60)).alias("power_w"),
            ])
            # Drop Kelvin columns — Celsius is sufficient, keeping both is redundant
            .drop(["air_temp_k", "proc_temp_k"])
        )
        return df

    def validate(self, df: pl.DataFrame):
        assert df.null_count().sum_horizontal().sum() == 0, "Nulls found in processed data"
        assert "air_temp_k" not in df.columns, "Kelvin column should have been dropped"
        assert "air_temp_c" in df.columns, "Celsius column missing"
        assert all(c in df.columns for c in self.target_cols), "Target columns missing"
        print(f"✅ Validation passed. Shape: {df.shape}")

    def run(self):
        print("🚀 Starting AI4I Data Processing...")
        raw_df = self.load_data()
        processed_df = self.clean_and_transform(raw_df)
        self.validate(processed_df)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        processed_df.write_parquet(self.output_path)

        print(f"✅ Processing Complete. Saved to: {self.output_path}")
        print(f"📊 Columns: {processed_df.columns}")
        print(f"📊 Shape: {processed_df.shape}")
        print(f"\nTarget distribution:")
        print(processed_df.select(["failure", "TWF", "HDF", "PWF", "OSF", "RNF"]).sum())


if __name__ == "__main__":
    processor = AI4IProcessor(
        input_path="data/raw/ai4i2020.csv",
        output_path="data/processed/ai4i_cleaned.parquet"
    )
    processor.run()