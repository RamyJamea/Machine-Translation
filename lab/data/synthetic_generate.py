import os
import time
import random
import pandas as pd
from tqdm import tqdm
from google import genai
from datetime import datetime
from pydantic import BaseModel, Field
from datasets import load_dataset, Dataset

MODEL_NAME = "gemini-3.1-flash-lite"
DATASET_NAME = "ramyibrahim/holol-ar-en-mt-v3"
API_KEYS = []
key_index = 0


def get_client():
    global key_index
    return genai.Client(api_key=API_KEYS[key_index])


def rotate_key():
    global key_index
    key_index = (key_index + 1) % len(API_KEYS)


PROMPT = """<PROMPT>
    <ROLE>
        You are a Specialized Linguistic Synthesizer and Bilingual Sample Generator. Your expertise lies in "Domain-Bound Heuristic Generation"—creating novel, diverse content by extracting core terminologies from a dataset while strictly maintaining logical coherence and high linguistic quality, even if the source material is flawed.
    </ROLE>

    <ANALYSIS_PHASE>
        Before generating, analyze the provided <EXAMPLES> for the following attributes:
        1. TERMINOLOGY_MAP: Isolate domain-specific jargon, named entities, and key semantic phrases.
        2. TONE_PROFILE: Identify the intended level of formality and register (e.g., clinical, technical, legal), ignoring any accidental drops in quality in the source.
        3. LOGICAL_FILTERING: Critically evaluate the examples to identify and discard non-logical inspirations, structural flaws, or poor translation artifacts. Abstract the intended meaning without inheriting the errors.
    </ANALYSIS_PHASE>

    <OBJECTIVE>
        Generate high-quality, structurally diverse bilingual samples. The goal is to produce novel, logical combinations that faithfully utilize the source's terminology, while actively elevating the overall linguistic quality and repairing any non-logical patterns found in the base examples.
    </OBJECTIVE>

    <CONSTRAINTS>
        <RULE>
            TERMINOLOGY ANCHORING: Strictly preserve and utilize the domain terminologies, named entities, and core concepts from the <EXAMPLES>. While you must anchor the content in these specific terms, you are permitted (and encouraged) to use high-quality structural vocabulary and natural grammatical connectors to ensure the sentence flows logically.
        </RULE>

        <RULE>
            LOGICAL SANITY OVER STRICT RECOMBINATION: Do not mimic non-logical structures, disjointed thoughts, or low-quality phrasing from the base examples. If an example is poorly constructed, extract its core terminology and reframe it into a robust, logically sound syntax.
        </RULE>

        <RULE>
            STRUCTURAL & SEMANTIC DIVERSITY: Explicitly vary sentence lengths and syntactic structures. Mix simple, compound, and complex sentences (including conditional, causal, and deeply descriptive clauses). Actively avoid repetitive syntactic loops to ensure maximum structural variety.
        </RULE>

        <RULE>
            GRAMMATICAL INTEGRITY: Ensure the TARGET remains a natural, idiomatic, and highly accurate translation of the SOURCE, adhering strictly to the highest grammatical standards of both languages.
        </RULE>
    </CONSTRAINTS>

    <EXAMPLES>
        {examples}
    </EXAMPLES>

    <OUTPUT_FORMAT>
        Generate [Number] samples. Ensure a rich mix of short, medium, and complexly structured sentences.
        
        <SAMPLE>
            <SOURCE>[Text in Source Language]</SOURCE>
            <TARGET>[Text in Target Language]</TARGET>
        </SAMPLE>
    </OUTPUT_FORMAT>
</PROMPT>
"""


def get_examples(dataset: Dataset, n=10):
    indices = random.sample(range(len(dataset)), n)
    rows = []
    for idx in indices:
        sample = dataset[idx]
        row_text = f"Arabic: {sample['ar']}\nEnglish: {sample['en']}"
        rows.append(row_text)
    return "\n\n".join(rows)


class Sample(BaseModel):
    ar: str = Field(..., description="Arabic Translation sentence.")
    en: str = Field(..., description="English Translation sentence.")


class TranslationSamples(BaseModel):
    samples: list[Sample] = Field(
        ..., description="Arabic English Translation Samples."
    )


def save_samples(samples: list[Sample], output_path: str):
    df = pd.DataFrame([s.model_dump() for s in samples])
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    file_exists = os.path.exists(output_path)
    df.to_csv(
        output_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8",
    )


def gen_samples(dataset, n_examples, n_samples, n_iter, sleep_s, output_path):
    total = n_iter
    iteration = 0
    while n_iter > 0:
        examples = get_examples(dataset, n_examples)
        success = False
        for attempt in range(len(API_KEYS)):
            try:
                client = get_client()
                interaction = client.interactions.create(
                    model=MODEL_NAME,
                    input=f"Generate {n_samples} Translation Samples.",
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": TranslationSamples.model_json_schema(),
                    },
                    system_instruction=PROMPT.format(examples=examples),
                    generation_config={
                        "thinking_level": "medium",
                        "temperature": 0.5,
                        "top_p": 0.5,
                        "seed": random.randint(0, 2**31 - 1),
                    },
                )
                tmp = TranslationSamples.model_validate_json(
                    interaction.steps[-1].content[0].text
                )
                save_samples(tmp.samples, output_path)
                iteration += 1
                n_iter -= 1
                print(
                    f"[OK] iteration {iteration}/{total} "
                    f"| saved {len(tmp.samples)} rows"
                )
                success = True
                time.sleep(sleep_s)
                break
            except Exception as e:
                print(f"[ERROR] attempt {attempt + 1}: {e}")
                rotate_key()
        if not success:
            print("[WARN] iteration skipped")
            n_iter -= 1


if __name__ == "__main__":
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir="data")
    output_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    gen_samples(
        dataset=dataset,
        n_examples=10,
        n_samples=10,
        n_iter=100,
        sleep_s=60,
        output_path=f"./data/synthetic/{output_path}.csv",
    )
    print(f"[DONE] saved to {output_path}")
