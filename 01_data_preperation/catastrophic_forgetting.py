"""
Catastrophic Forgetting:
Training only on a narrow domain dataset causes the model to forget general capabilities.
Fix: Interleave the domain dataset with a general reasoning replay dataset.
"""

from datasets import Features, Value, interleave_datasets, load_dataset

# MOTIVATION: We want to avoid loading multi-gigabyte datasets into a system's Random Access Memory (RAM) because it triggers Out-Of-Memory (OOM) errors.
# Using the Streaming mode (`streaming=True`) for the download reads small chunks of data over HTTP from massive datasets hosted by Hugging Face. This style of downloading is referred to as downloading "lazily" in jargon.
# The data will be accessed as and when it is necessary during training, allowing later training scripts to start immediately without waiting for huge downloads.
domain_ds = load_dataset(
    "iknow-lab/open-materials-guide-2024", split="train", streaming=True
)
replay_ds = load_dataset(
    "nvidia/Nemotron-SFT-Science-v2", "rqa", split="train", streaming=True
)


def format_domain(row):
    # MOTIVATION: We want to replace missing data entries (`None`) with fallback text because empty values can crash text processing tools or accidentally teach the model to output the word "None".
    # Checking multiple column names and falling back to an empty string ensures we always get a valid piece of text. In data engineering jargon, this fallback chain is called "null coalescing".
    # Down the line in the workflow, every training step is guaranteed to receive clean text strings without unexpected crashes mid-training.
    target = str(row.get("Target") or row.get("Material_Name") or "Unknown Material")
    recipe = str(row.get("Synthesis_Recipe") or row.get("recipe") or "")
    return {"text": f"Target: {target}\nRecipe: {recipe}".strip()}


def format_replay(row):
    messages = row.get("messages") or []
    lines = [
        f"{m.get('role', 'user')}: {m.get('content') or ''}"
        for m in messages
        if isinstance(m, dict) and m.get("content")
    ]
    return {"text": "\n".join(lines).strip()}


def to_uniform_stream(ds, fn):
    # MOTIVATION: We cannot directly combine two datasets that have different column names or data formats because the dataset mixer requires every row from both sources to look identical in structure.
    # Here we delete all original columns (`remove_columns`) and force the single remaining `"text"` column to be stored as a standard text string (`Features`). In jargon, this is called "schema alignment" and "type casting".
    # Down the line, the training loop can read from the combined stream seamlessly without knowing or caring which original database a given example came from.
    cols = ds.column_names or (list(ds.features.keys()) if ds.features else None)
    ds = ds.map(fn, remove_columns=cols)
    if ds.features is not None:
        ds = ds.cast(Features({"text": Value("string")}))
    return ds


domain_clean = to_uniform_stream(domain_ds, format_domain)
replay_clean = to_uniform_stream(replay_ds, format_replay)

# MOTIVATION: We do not want to train the model exclusively on materials synthesis recipes because doing so overwrites its pre-existing knowledge of general science and math.
# Mixing 80% new materials examples with 20% general science examples reminds the model of its broader skills while it learns the new topic. In machine learning jargon, the forgetting problem is called "catastrophic forgetting", and mixing in old-style data is called "experience replay" or "dataset interleaving".
# As a result, when we evaluate the final model at the end of the workflow, it will be able to write lab recipes while still answering general reasoning questions accurately.
mixed_ds = interleave_datasets(
    [domain_clean, replay_clean],
    probabilities=[0.8, 0.2],
    seed=42,
    stopping_strategy="first_exhausted",
)

if __name__ == "__main__":
    for i, sample in enumerate(mixed_ds.take(3)):
        print(f"--- Sample {i + 1} ---")
        print(sample["text"][:200])
