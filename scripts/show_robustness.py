"""Print the robustness table as a compact pivot (recall by attack strength)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import RESULTS

df = pd.read_csv(RESULTS / "robustness.csv")
for model in df["model"].unique():
    sub = df[df["model"] == model]
    print(f"\n=== {model} (spam recall) ===")
    print(sub.pivot(index="attack", columns="rate", values="recall")
             .to_string())
