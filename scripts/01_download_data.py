"""Fetch the UCI SMS Spam Collection into data/raw/.

Usage:  python scripts/01_download_data.py
"""
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config import DATA_RAW

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
TARGET = DATA_RAW / "SMSSpamCollection"


def main() -> int:
    if TARGET.exists():
        print(f"already present: {TARGET}")
        return 0

    print(f"downloading {URL}")
    try:
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        print(f"\nDownload failed: {exc}\n")
        print("Manual fallback:")
        print(f"  1. open {URL}")
        print("  2. extract the file named 'SMSSpamCollection' (no extension)")
        print(f"  3. place it at {TARGET}")
        return 1

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith("SMSSpamCollection")]
        if not names:
            print(f"unexpected archive contents: {zf.namelist()}")
            return 1
        TARGET.write_bytes(zf.read(names[0]))

    n = sum(1 for _ in TARGET.open(encoding="utf-8", errors="replace"))
    print(f"wrote {TARGET} ({n} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
