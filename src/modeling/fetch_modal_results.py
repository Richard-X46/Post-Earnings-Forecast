"""Download all PEAD pipeline outputs from Modal Volume.

Run after modal_tabnet_pead.py completes:
    python src/modeling/fetch_results.py

All files land in: src/modeling/modal_tabnet_outputs/
"""

import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "modal_tabnet_outputs"
OUT_DIR.mkdir(exist_ok=True)

FILES = [
    ("tabnet-pead-output", "results.parquet",           "results.parquet"),
    ("tabnet-pead-output", "per_fold_summary.parquet",  "per_fold_summary.parquet"),
    ("tabnet-pead-output", "histories_clf.json",        "histories_clf.json"),
    ("tabnet-pead-output", "histories_reg.json",        "histories_reg.json"),
]

ok = 0
for volume_name, remote, local_name in FILES:
    local_path = OUT_DIR / local_name
    print(f"\nDownloading {remote} ...")
    result = subprocess.run(
        [
            "modal", "volume", "get",
            volume_name,
            remote,
            str(local_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        size_mb = local_path.stat().st_size / 1_000_000
        print(f"  saved {local_name}  ({size_mb:.1f} MB)")
        ok += 1
    else:
        stderr = result.stderr.strip()
        print(f"  FAILED\n{stderr}")
        sys.exit(1)

print(f"\n{'='*50}")
print(f"Downloaded {ok}/{len(FILES)} files → {OUT_DIR}")
print(f"{'='*50}")
