"""Extract the Kaggle phishing-email archive into data/raw/phishing/.

Usage:  python scripts/06_extract_phishing.py [path/to/archive.zip]
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import DATA_RAW

DEFAULT_ZIP = Path.home() / "Downloads" / "archive.zip"
DEST = DATA_RAW / "phishing"


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ZIP
    if not src.exists():
        print(f"archive not found: {src}")
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        print(f"{len(members)} csv files in {src.name}\n")
        for name in members:
            out = DEST / Path(name).name
            out.write_bytes(zf.read(name))

    for path in sorted(DEST.glob("*.csv")):
        df = pd.read_csv(path, nrows=5)
        full = pd.read_csv(path, usecols=[df.columns[-1]])
        print(f"{path.name}")
        print(f"   rows    {len(full)}")
        print(f"   columns {list(df.columns)}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
