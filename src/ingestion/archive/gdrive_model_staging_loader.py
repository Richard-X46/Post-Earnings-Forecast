"""Upload local model_staging parquet files to Google Drive using a service account.

One-time setup:
    1. Go to https://console.cloud.google.com/apis/credentials
    2. Create a Service Account, download JSON key as service_account.json
    3. Place it next to this script OR set SA_KEY_PATH env var
    4. Share the Drive folder with the service account's email (e.g. editor)

Usage:
    python src/ingestion/gdrive_model_staging_loader.py

Colab loads from: /content/drive/MyDrive/model_staging/
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from pydrive2.auth import GoogleAuth, ServiceAccountCredentials
    from pydrive2.drive import GoogleDrive
except ImportError:
    raise ImportError("pip install pydrive2")

LOCAL_ROOT = Path("src/data/model_staging")

FILES = [
    ("tech_modeling_table.parquet", ""),
    ("finbert_tx_agg_weighted.parquet", ""),
    ("nz_sentiment.parquet", ""),
    ("fundamentalIndicators/modeling_fundamentals.parquet", "fundamentalIndicators"),
    ("analyst_upgrades.parquet", ""),
    ("finbert_tx_agg_mean.parquet", ""),
]


def _find_or_create_folder(drive: GoogleDrive, folder_name: str) -> str:
    query = (
        f"title = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' "
        "and trashed = false"
    )
    results = drive.ListFile({"q": query}).GetList()
    if results:
        return results["id"]

    folder = drive.CreateFile({
        "title": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
    })
    folder.Upload()
    print(f"  Created folder: {folder_name}")
    return folder["id"]


def main() -> None:
    key_path = os.getenv("SA_KEY_PATH", "src/ingestion/service_account.json")
    if not os.path.exists(key_path):
        raise FileNotFoundError(
            f"{key_path} not found. Download the service account JSON key "
            "from Google Cloud Console and place it there, or set SA_KEY_PATH."
        )

    gauth = GoogleAuth()
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        key_path, scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive = GoogleDrive(gauth)

    folder_id = _find_or_create_folder(drive, "model_staging")
    sub_id = _find_or_create_folder(drive, "fundamentalIndicators")
    print()

    for filename, subfolder in FILES:
        local_path = LOCAL_ROOT / filename
        parent_id = sub_id if subfolder else folder_id

        gfile = drive.CreateFile({
            "title": Path(filename).name,
            "parents": [{"id": parent_id}],
        })
        gfile.SetContentFile(str(local_path))
        gfile.Upload()
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  {filename}  →  {gfile['title']}  ({size_mb:.1f} MB)")

    print(f"\nDone. Colab path: /content/drive/MyDrive/model_staging/")


if __name__ == "__main__":
    main()
