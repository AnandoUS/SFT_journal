"""
Catastrophic Forgetting:
Training only on a narrow domain dataset causes the model to forget general capabilities.
Fix: Standardize both datasets into Conversational `messages` format and interleave the domain
stream with a general science replay stream.
"""

from datasets import Features, Value, interleave_datasets, load_dataset

SYSTEM_PROMPT = "You are a materials science assistant specializing in inorganic synthesis."

CONVERSATIONAL_FEATURES = Features(
    {
        "messages": [
            {
                "role": Value("string"),
                "content": Value("string"),
            }
        ]
    }
)

# MOTIVATION: We want to avoid loading multi-gigabyte datasets into a system's Random Access Memory (RAM) because it triggers Out-Of-Memory (OOM) errors.
# Using the Streaming mode (`streaming=True`) for the download reads small chunks of data over HTTP from massive datasets hosted by Hugging Face. This style of downloading is referred to as downloading "lazily" in jargon.
# CONTEXT ON INNER WORKINGS: Instead of downloading the entire dataset onto the hard drive before training begins, Python opens a live web connection and fetches just a few dozen recipes at a time into a small temporary memory buffer, discarding old batches as soon as the model finishes learning from them.
# The data will be accessed as and when it is necessary during training, allowing later training scripts to start immediately without waiting for huge downloads.
domain_ds = load_dataset(
    "iknow-lab/open-materials-guide-2024", split="train", streaming=True
)
replay_ds = load_dataset(
    "nvidia/Nemotron-SFT-Science-v2", "rqa", split="train", streaming=True
)


def format_domain(row):
    # MOTIVATION: We want to replace missing data entries (`None`) with fallback text before building our `system`, `user`, and `assistant` turns because empty values can crash text processing tools or accidentally teach the model to output the word `"None"`.
    # Checking multiple column names and falling back to an empty string ensures every message turn holds valid text. In data engineering jargon, this fallback chain is called "null coalescing".
    # Down the line in the workflow, every training step is guaranteed to receive clean conversational dictionaries without unexpected crashes mid-training.
    target = str(row.get("Target") or row.get("Material_Name") or "Unknown Material").strip()
    formula = str(row.get("Chemical_Formula") or "").strip()
    recipe = str(row.get("Synthesis_Recipe") or row.get("recipe") or "").strip()
    user_prompt = (
        f"Write a synthesis procedure for {target} ({formula})."
        if formula
        else f"Write a synthesis procedure for {target}."
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": recipe},
        ]
    }


def format_replay(row):
    raw_msgs = row.get("messages") or []
    clean_msgs = [
        {"role": str(m.get("role") or "user"), "content": str(m.get("content") or "").strip()}
        for m in raw_msgs
        if isinstance(m, dict) and str(m.get("content") or "").strip()
    ]
    return {"messages": clean_msgs}


def to_uniform_stream(ds, fn):
    # MOTIVATION: We cannot directly combine two datasets that have different column names or nested dictionary keys because the dataset mixer requires every row from both sources to share the exact same structure.
    # Here we delete all original columns (`remove_columns`), standardize every row into a `"messages"` list of `{"role", "content"}` dictionaries, and lock in the column types with `CONVERSATIONAL_FEATURES`. In jargon, this is called "schema alignment" and "type casting".
    # Down the line, `SFTTrainer` can read from the combined stream seamlessly and apply any model's chat template without caring which database an example came from.
    cols = ds.column_names or (list(ds.features.keys()) if ds.features else None)
    ds = ds.map(fn, remove_columns=cols)
    if ds.features is not None:
        ds = ds.cast(CONVERSATIONAL_FEATURES)
    return ds.filter(
        lambda r: any(m["role"] == "assistant" and m["content"] for m in r["messages"])
    )


domain_clean = to_uniform_stream(domain_ds, format_domain)
replay_clean = to_uniform_stream(replay_ds, format_replay)

# MOTIVATION: We do not want to train the model exclusively on materials synthesis recipes because doing so overwrites its pre-existing knowledge of general science and math.
# Mixing 80% new materials examples with 20% general science examples reminds the model of its broader skills while it learns the new topic. In machine learning jargon, the forgetting problem is called "catastrophic forgetting", and mixing in old-style data is called "experience replay" or "dataset interleaving".
# CONTEXT ON INNER WORKINGS: A language model stores general science skills and new materials facts inside the exact same shared weight tables. If every training step only adjusts those weights to improve materials recipes, the weights gradually drift away from values that general math and reasoning relied on—unless occasional general-science examples pull them back. In jargon, new updates overwriting shared weights is called "parameter interference" (or "weight drift").
# As a result, when we evaluate the final model at the end of the workflow, it will be able to write lab recipes while still answering general reasoning questions accurately.
mixed_ds = interleave_datasets(
    [domain_clean, replay_clean],
    probabilities=[0.8, 0.2],
    seed=42,
    stopping_strategy="first_exhausted",
)

if __name__ == "__main__":
    for i, sample in enumerate(mixed_ds.take(3)):
        print(f"--- Sample {i + 1} ({len(sample['messages'])} turns) ---")
        print(sample["messages"][-1]["role"], ":", sample["messages"][-1]["content"][:160])
