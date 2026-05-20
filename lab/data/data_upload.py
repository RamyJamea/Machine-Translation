import pandas as pd
import pyarabic.araby as araby
from datasets import Dataset
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


df = pd.read_excel(
    r"C:\Users\ramyu\OneDrive\Desktop\MachineT\assets\data\phrases.xlsx",
    engine="openpyxl",
)

df["en_normalized"] = df["english_sentence"].apply(normalize_english)
df["ar_normalized"] = df["arabic_sentence"].apply(normalize_arabic)

df.replace("", pd.NA, inplace=True)
df.dropna(subset=["en_normalized", "ar_normalized"], inplace=True)

final_df = df[["en_normalized", "ar_normalized"]].rename(
    columns={"en_normalized": "en", "ar_normalized": "ar"}
)

final_df.reset_index(drop=True, inplace=True)
hf_dataset = Dataset.from_pandas(final_df)

hf_dataset.push_to_hub("ramyibrahim/english-arabic-translation", private=False)

print("Dataset uploaded successfully with preserved Arabic script!")
