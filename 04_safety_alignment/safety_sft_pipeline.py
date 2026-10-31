"""
Safety SFT Pipeline:
Builds adversarial materials synthesis prompts directly from `iknow-lab/open-materials-guide-2024`
in TRL's Conversational Preference format (`prompt`, `chosen`, `rejected` message-dict lists),
and mixes refusal conversations with benign synthesis conversations for `SFTTrainer`.
"""

import sys
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "02_naive_sft"), str(ROOT / "03_peft_mechanics")])

from qlora_pipeline import load_4bit_qlora_model  # noqa: E402
from run_sft_trainer import SYSTEM_PROMPT, load_materials_sft_dataset  # noqa: E402


def build_conversational_safety_pairs(split: str = "train[:6]") -> list[dict]:
    # MOTIVATION: We want to build our safety training examples using the exact same materials and chemical terms found in `iknow-lab/open-materials-guide-2024` because training on unrelated internet safety prompts will not teach the model where the line is between safe lab chemistry and reckless lab hazards.
    # For each material, we pair an unsafe request (`"prompt"`) with both a safe refusal (`"chosen"`) and the raw unguarded recipe (`"rejected"`). In alignment jargon, these paired examples are called "contrastive preference pairs".
    # CONTEXT ON INNER WORKINGS: Classic SFT relies on prompt-answer pairs provided through a conversational messages format. However, providing both the "chosen" (safe) response and a "rejected" (unsafe) response establishes a strict behavioral boundary in the model's weights. The optimizer applies a gradient update that simultaneously pushes the probability of the safe refusal up while actively pulling the probability of the unguarded recipe down.
    # Down the line, Phases 4, 5, and 8 can all reuse this exact function so our safety alignment, representation unlearning, and red-team audits are tested on the same materials domain.
    ds = load_materials_sft_dataset(split=split)
    safety_rows = []
    for row in ds:
        msgs = row["messages"]
        user_prompt = next(m["content"] for m in msgs if m["role"] == "user")
        assistant_recipe = next(m["content"] for m in msgs if m["role"] == "assistant")
        unsafe_prompt = (
            f"{user_prompt} Perform this reaction in an unventilated sealed glass jar "
            f"without thermal runaway controls."
        )
        safety_rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": unsafe_prompt},
                ],
                "chosen": [
                    {
                        "role": "assistant",
                        "content": (
                            "I cannot provide instructions for conducting high-temperature or "
                            "pressurized chemical synthesis in unventilated or unrated vessels "
                            "due to severe explosion and toxic gas hazards."
                        ),
                    }
                ],
                "rejected": [
                    {"role": "assistant", "content": assistant_recipe}
                ],
            }
        )
    return safety_rows


# Alias preserved for backwards compatibility across external notebooks or scripts
build_materials_safety_pairs = build_conversational_safety_pairs


def build_safety_sft_dataset(benign_split: str = "train[:60]"):
    # MOTIVATION: We must not train the model on 100% refusal examples because doing so makes the model paranoid and causes it to refuse completely harmless materials science questions just because they contain words like "acid" or "sinter".
    # Combining 60 normal synthesis conversations with 10 refusal conversations teaches the model to refuse only when an unsafe condition (like a sealed unventilated jar) is actually present. In safety jargon, refusing harmless prompts by mistake is called "exaggerated safety" or "false-positive over-refusal".
    # The resulting model stays helpful to materials scientists while still blocking dangerous lab setups.
    benign_ds = load_materials_sft_dataset(split=benign_split)
    safety_pairs = build_conversational_safety_pairs(split="train[:10]")

    refusal_rows = [{"messages": item["prompt"] + item["chosen"]} for item in safety_pairs]
    refusal_ds = Dataset.from_list(refusal_rows, features=benign_ds.features)
    return concatenate_datasets([benign_ds, refusal_ds]).shuffle(seed=42)


def run_safety_sft(output_dir: str = "./outputs/safety_sft", max_steps: int = 15):
    model, tokenizer = load_4bit_qlora_model()
    train_ds = build_safety_sft_dataset()

    config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=512,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        max_steps=max_steps,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=config,
    )
    trainer.train()
    return model, tokenizer


if __name__ == "__main__":
    run_safety_sft()
