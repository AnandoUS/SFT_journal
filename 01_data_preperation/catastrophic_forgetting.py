"""
Catastrophic Forgetting:
Training only on a narrow domain dataset causes the model to forget general capabilities.
Fix: Interleave the domain dataset with a general reasoning replay dataset.
"""

from datasets import Features, Value, interleave_datasets, load_dataset

# MOTIVATION: Streaming mode (`streaming=True`) reads shards lazily over HTTP without
# downloading multi-gigabyte datasets to local disk before training starts.
domain_ds = load_dataset(
    "iknow-lab/open-materials-guide-2024", split="train", streaming=True
)
replay_ds = load_dataset(
    "nvidia/Nemotron-SFT-Science-v2", "rqa", split="train", streaming=True
)


def format_domain(row):
    # MOTIVATION: Coalesce nullable fields to strings so missing entries in raw tables
    # never inject `"None"` literals or crash downstream tokenizers.
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
    # MOTIVATION: `interleave_datasets` requires every stream to share the exact same
    # column names and PyArrow feature types; dropping original columns and casting
    # prevents schema mismatch errors when mixing datasets in later SFT stages.
    cols = ds.column_names or (list(ds.features.keys()) if ds.features else None)
    ds = ds.map(fn, remove_columns=cols)
    if ds.features is not None:
        ds = ds.cast(Features({"text": Value("string")}))
    return ds


domain_clean = to_uniform_stream(domain_ds, format_domain)
replay_clean = to_uniform_stream(replay_ds, format_replay)

# MOTIVATION: Experience Replay (mixing 80% domain data with 20% general reasoning data)
# anchors pre-trained reasoning weights during SFT so the model learns materials synthesis
# without suffering catastrophic forgetting on general math and logic tasks.
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
