# src/consolidate_nasa.py
import polars as pl
import os

def consolidate_nasa():
    processed_dir = "data/processed"
    datasets = ["FD001", "FD002", "FD003", "FD004"]

    dfs = []
    for ds in datasets:
        path = os.path.join(processed_dir, f"nasa_train_{ds.lower()}_cleaned.parquet")
        if not os.path.exists(path):
            print(f"⚠️  Missing: {path} — run data_processing_nasa.py first.")
            continue

        df = pl.read_parquet(path)
        dfs.append(df)
        print(f"📦 Loaded {ds}: {df.height} rows, {df.width} cols")

    if not dfs:
        raise RuntimeError("No processed NASA files found.")

    master_df = pl.concat(dfs, how="diagonal")

    master_path = os.path.join(processed_dir, "nasa_master.parquet")
    master_df.write_parquet(master_path)

    print("-" * 40)
    print(f"✅ NASA Master Created!")
    print(f"📊 Total Rows: {master_df.height}")
    print(f"📊 Rows per dataset:")
    print(master_df.group_by("dataset_id").agg(pl.len().alias("rows")).sort("dataset_id"))
    print(f"📂 Saved to: {master_path}")


if __name__ == "__main__":
    consolidate_nasa()