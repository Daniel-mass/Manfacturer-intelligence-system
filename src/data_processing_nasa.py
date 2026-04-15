#src/data_processing_nasa.py
import polars as pl
import numpy as np
import os

SENSORS_TO_DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
INFORMATIVE_SENSORS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in SENSORS_TO_DROP]


class NASAProcessor:
    def __init__(self, input_path: str, output_path: str, dataset_id: str):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_id = dataset_id.upper()

        self.columns = (
            ["unit", "cycle", "op_set1", "op_set2", "op_set3"]
            + [f"s{i}" for i in range(1, 22)]
        )

    def load_data(self) -> pl.DataFrame:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"NASA file not found at {self.input_path}")

        raw_lines = []
        with open(self.input_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 26:
                    raw_lines.append(parts[:26])

        df = pl.DataFrame(raw_lines, schema=self.columns, orient="row")
        return df.select([pl.col(c).cast(pl.Float64) for c in self.columns])

    def drop_constant_sensors(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.drop([s for s in SENSORS_TO_DROP if s in df.columns])

    def cluster_operating_conditions(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.dataset_id in ["FD002", "FD004"]:
            df = df.with_columns(
                pl.col("op_set1").round(2).alias("op_set1_rounded")
            )
            conditions = (
                df.select("op_set1_rounded")
                .unique()
                .sort("op_set1_rounded")
                .with_row_index("condition_cluster")
            )
            df = (
                df.join(conditions, on="op_set1_rounded", how="left")
                .drop("op_set1_rounded")
                .with_columns(pl.col("condition_cluster").cast(pl.Int32))
            )
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int32).alias("condition_cluster"))

        return df

    def normalize_per_condition(self, df: pl.DataFrame) -> pl.DataFrame:
        normalized_cols = []
        for col in INFORMATIVE_SENSORS:
            if col not in df.columns:
                continue
            mean_expr = pl.col(col).mean().over("condition_cluster")
            std_expr = pl.col(col).std().over("condition_cluster")
            normalized_cols.append(
                ((pl.col(col) - mean_expr) / (std_expr + 1e-6)).alias(col)
            )
        return df.with_columns(normalized_cols)

    def calculate_rul(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col("cycle").max().over("unit").alias("max_cycle")
            )
            .with_columns(
                (pl.col("max_cycle") - pl.col("cycle")).alias("rul")
            )
            .drop("max_cycle")
        )

    def run(self):
        print(f"🚀 Processing {self.dataset_id}: {self.input_path}")

        df = self.load_data()
        print(f"  Loaded: {df.shape}")

        df = self.drop_constant_sensors(df)
        print(f"  After dropping constant sensors: {df.shape}")

        df = self.cluster_operating_conditions(df)
        print(f"  Operating conditions found: {df['condition_cluster'].n_unique()}")

        df = self.normalize_per_condition(df)
        df = self.calculate_rul(df)
        df = df.with_columns(pl.lit(self.dataset_id).alias("dataset_id"))

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.write_parquet(self.output_path)

        print(f"✅ Done. Shape: {df.shape} | Saved: {self.output_path}")
        return df


if __name__ == "__main__":
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        processor = NASAProcessor(
            input_path=f"data/raw/train_{ds}.txt",
            output_path=f"data/processed/nasa_train_{ds.lower()}_cleaned.parquet",
            dataset_id=ds
        )
        processor.run()