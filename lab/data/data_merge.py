import pandas as pd
import pyarabic.araby as araby
from datasets import Dataset, load_dataset
from huggingface_hub import login
from cleantext import clean

login("")


def normalize_english(text):
    if not isinstance(text, str):
        return ""

    return clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
    )


def normalize_arabic(text):
    if not isinstance(text, str):
        return ""

    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        no_line_breaks=True,
        no_urls=True,
    )

    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = araby.normalize_alef(text)
    text = araby.normalize_hamza(text)
    text = araby.normalize_ligature(text)
    text = araby.normalize_teh(text)

    return text


old_dataset = load_dataset("ramyibrahim/cniomt-ar-en-v5", split="train")

old_df = old_dataset.to_pandas()
old_df = old_df[["en", "ar"]]

df = pd.read_csv(r"C:\Users\ramyu\code\MachineT\data_05_21_2026_v3.csv")
print(df.head())
df = df[["en", "ar"]]
df["en"] = df["en"].apply(normalize_english)
df["ar"] = df["ar"].apply(normalize_arabic)
df.replace("", pd.NA, inplace=True)
df.dropna(subset=["en", "ar"], inplace=True)
df["en"] = df["en"].str.strip()
df["ar"] = df["ar"].str.strip()

final_df = pd.concat([old_df, df], ignore_index=True)
final_df.drop_duplicates(subset=["en", "ar"], inplace=True)
final_df.replace("", pd.NA, inplace=True)
final_df.dropna(subset=["en", "ar"], inplace=True)
final_df.reset_index(drop=True, inplace=True)

hf_dataset = Dataset.from_pandas(final_df, preserve_index=False)
print(hf_dataset)
hf_dataset.push_to_hub("ramyibrahim/cniomt-ar-en-v6", private=False)

print("Dataset uploaded successfully!")
