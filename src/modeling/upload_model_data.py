"""Upload model_staging parquet files to Modal Volume.

Run once before running modal_tabnet_pead.py:
    python src/modeling/upload_model_data.py
"""
import modal
from pathlib import Path

vol = modal.Volume.from_name("model-staging", create_if_missing=True)

FILES = [
    ("src/data/model_staging/tech_modeling_table.parquet",
     "tech_modeling_table.parquet"),
    ("src/data/model_staging/finbert_tx_agg_weighted.parquet",
     "finbert_tx_agg_weighted.parquet"),
    ("src/data/model_staging/nz_sentiment.parquet",
     "nz_sentiment.parquet"),
    ("src/data/model_staging/fundamentalIndicators/modeling_fundamentals.parquet",
     "fundamentalIndicators/modeling_fundamentals.parquet"),
]

total = 0
with vol.batch_upload() as upload:
    for local, remote in FILES:
        p = Path(local)
        if not p.exists():
            print(f"SKIP: {local} not found")
            continue
        size_mb = p.stat().st_size / 1e6
        total += size_mb
        print(f"Uploading {remote} ({size_mb:.1f} MB)...")
        upload.put_file(local, remote)

print(f"\nDone. {sum(1 for l,_ in FILES if Path(l).exists())} files "
      f"({total:.1f} MB) uploaded to Volume 'model-staging'.")
