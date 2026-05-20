from pathlib import Path
import pandas as pd
from datasets import Dataset
from huggingface_hub import login

login("")

CSV_DIR = r"C:\Users\ramyu\OneDrive\Desktop\MachineT\lab\data\synthetic"
REPO_ID = "ramyibrahim/holol-ar-en-synth-v2"
PRIVATE = False

csv_files = list(Path(CSV_DIR).glob("*.csv"))
dfs = []
for file in csv_files:
    print(f"Loading: {file}")
    df = pd.read_csv(file)
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)
print("Total rows:", len(merged_df))
dataset = Dataset.from_pandas(merged_df, preserve_index=False)
dataset.push_to_hub(REPO_ID, private=PRIVATE)
print("Done.")
