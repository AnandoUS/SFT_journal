"""
Baseline Full-Parameter SFT (`SFTTrainer`):
Provides reusable model/dataset loaders standardized on the Conversational `messages` format
(`system`, `user`, `assistant` dictionaries) and runs full-parameter supervised fine-tuning
on a compact causal LM (`Qwen/Qwen2.5-0.5B`) using `iknow-lab/open-materials-guide-2024`.
"""

import torch
from datasets import Features, Value, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"
SYSTEM_PROMPT = "You are a materials science assistant specializing in inorganic synthesis."

DEFAULT_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

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


def load_model_and_tokenizer(model_id: str = DEFAULT_MODEL_ID, **model_kwargs):
    """Load causal LM and tokenizer with safe padding and ChatML template defaults for SFT."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # MOTIVATION: We need to define a blank filler token and place it at the right end of shorter sentences because batching sentences of different lengths requires making them the same length in a rectangular grid.
    # Many base models do not come with a dedicated filler token, so we reuse the end-of-sentence token (`eos_token`) and pad on the `"right"`. In jargon, this filler is called the "pad token", and putting it at the end is called "right-padding".
    # Putting filler tokens at the end (instead of the beginning) ensures the real words at the start of the sentence keep their natural word positions (0, 1, 2...) during training across all scripts in this repo.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs.setdefault("torch_dtype", dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    return model, tokenizer


def load_materials_sft_dataset(split: str = "train[:200]"):
    """Load and clean `iknow-lab/open-materials-guide-2024` into Conversational `{'messages': [...]}` rows."""
    # MOTIVATION: We standardize every raw materials row into `system`, `user`, and `assistant` dictionaries right inside our root loader because downstream trainers, alignment scripts, and audits all need clear speaker boundaries rather than hardcoded text prefixes.
    # Wrapping each turn in `{"role": ..., "content": ...}` produces a model-agnostic dataset that `SFTTrainer` and `tokenizer.apply_chat_template()` natively understand. In jargon, this is the "Conversational Messages format".
    # Down the line, Phases 3, 4, 5, 7, and 8 can all reuse this exact dataset loader seamlessly across Qwen, Llama, or Gemma model families.
    ds = load_dataset("iknow-lab/open-materials-guide-2024", split=split)
    ds = ds.filter(lambda r: bool((r.get("Synthesis_Recipe") or r.get("recipe") or "").strip()))

    def format_row(row):
        target = str(row.get("Target") or row.get("Material_Name") or "target material").strip()
        formula = str(row.get("Chemical_Formula") or "").strip()
        recipe = str(row.get("Synthesis_Recipe") or row.get("recipe") or "").strip()
        prompt = (
            f"Write a synthesis procedure for {target} ({formula})."
            if formula
            else f"Write a synthesis procedure for {target}."
        )
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": recipe},
            ]
        }

    return ds.map(
        format_row, remove_columns=ds.column_names, features=CONVERSATIONAL_FEATURES
    )


def run_baseline_sft(output_dir: str = "./outputs/naive_sft", max_steps: int = 20):
    model, tokenizer = load_model_and_tokenizer()
    train_ds = load_materials_sft_dataset(split="train[:200]")

    config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=512,
        per_device_train_batch_size=4,
        # MOTIVATION: We want the stability of training on a larger batch of examples at once without buying a bigger Graphics Processing Unit (GPU) because fitting many long recipes into GPU Video RAM (VRAM) at the same time causes Out-Of-Memory (OOM) crashes.
        # Processing 4 examples at a time (`per_device_train_batch_size=4`) and adding up the corrections over 2 rounds (`gradient_accumulation_steps=2`) before updating the model gives the exact same mathematical update as a batch of 8. In jargon, this is called "gradient accumulation".
        # CONTEXT ON INNER WORKINGS: While gradient tensor sizes are independent of input size, the self-attention matrices used to calculate them during backpropagation scale quadratically with the length of the recipe (known as sequence length T) and are stored after the forward pass.
        # Gradient accumulation allows for clearing out the intermediate self-attention matrices for micro-batches to prevent OOM errors and lets us train with stable, large-batch updates even on a modest consumer GPU.
        gradient_accumulation_steps=2,
        # MOTIVATION: We want to avoid storing every single intermediate layer's calculations in GPU Video RAM (VRAM) during the forward pass because those saved calculations quickly fill up all available memory on long texts.
        # Turning on `gradient_checkpointing` throws away most intermediate calculations and recalculates them on the fly when updating weights. In jargon, this trade-off is called "activation recomputation" or "gradient checkpointing".
        # CONTEXT ON INNER WORKINGS: Normally, as a prompt passes forward through all 24+ layers of a model, the GPU keeps every layer's intermediate scratch-work in memory so it can compute weight updates on the way back. Gradient checkpointing deletes most of that scratch-work during the forward pass and quickly recalculates one layer at a time during the backward update pass. In jargon, these saved intermediate layer outputs are called "activations".
        # Training takes about 20% longer in wall-clock time, but uses up to 60% less GPU memory, allowing much longer synthesis recipes to fit without crashing.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # MOTIVATION: We do not want to hit the model with the full learning speed on the very first step because the optimizer has not yet calibrated the average size of the updates, which can cause wild, destructive jumps in the model's weights.
        # Starting with a tiny learning speed for the first 5% of steps (`warmup_ratio=0.05`) and then smoothly tapering it down along a curve (`"cosine"`) keeps updates gentle. In jargon, this speed profile is called a "cosine learning rate schedule with linear warmup".
        # CONTEXT ON INNER WORKINGS: Modern optimizers automatically scale each weight's update based on a running average of its recent update sizes. On step 1, the optimizer has no history yet, so taking a full-speed step before that running average settles can cause a huge, erratic jump that damages pre-trained knowledge. In jargon, these running averages are called "optimizer moments" (used by the "AdamW" optimizer).
        # This prevents early loss spikes and helps the model settle smoothly into a good solution at the end of training.
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        # MOTIVATION: We want to prevent the model from memorizing the exact wording of our small training set word-for-word because rote memorization makes the model brittle when users phrase questions slightly differently.
        # Setting `neftune_noise_alpha=5.0` adds a tiny amount of random jitter to the word representations during training. In jargon, this technique is called "NEFTune" (Noisy Embeddings Fine-Tuning).
        # The model is forced to learn the underlying meaning of the synthesis steps rather than memorizing exact phrasing, producing more natural answers during evaluation.
        neftune_noise_alpha=5.0,
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
    run_baseline_sft()
