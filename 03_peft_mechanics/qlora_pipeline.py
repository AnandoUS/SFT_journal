"""
QLoRA Pipeline (4-bit NF4 Quantization + rsLoRA / DoRA Adapters):
Quantizes the frozen base model to 4-bit NormalFloat (NF4) via `bitsandbytes` and attaches
trainable LoRA adapters across all linear layers. Consumes Conversational `messages` datasets
directly from `load_materials_sft_dataset` in Phase 2.
"""

import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "02_naive_sft"))

from run_sft_trainer import (  # noqa: E402
    DEFAULT_CHATML_TEMPLATE,
    DEFAULT_MODEL_ID,
    load_materials_sft_dataset,
)


def load_4bit_qlora_model(
    model_id: str = DEFAULT_MODEL_ID,
    r: int = 16,
    lora_alpha: int = 32,
    add_domain_tokens: bool = False,
):
    """Load a 4-bit NF4 quantized base model and attach Rank-Stabilized LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
    tokenizer.padding_side = "right"

    if torch.cuda.is_available():
        # MOTIVATION: We want to compress the frozen base model's weights from 16-bit numbers down to 4-bit numbers because storing a full 16-bit model takes four times as much GPU Video RAM (VRAM).
        # Using the `"nf4"` (4-bit NormalFloat) format stores weights compactly in 4 bits and unpacks them into 16-bit numbers (`bfloat16`) only for the split second when a layer is doing math. In jargon, compressing weights to 4-bit precision while training LoRA adapters on top is called "QLoRA" (Quantized LoRA), and compressing the compression scales themselves is called "double quantization".
        # CONTEXT ON INNER WORKINGS: Most weights in a trained neural network cluster tightly around zero in a bell-curve shape, so `"nf4"` places more of its 16 possible 4-bit rounding values near zero where accuracy matters most. During math steps, only one small chunk of weights at a time is unpacked to 16-bit inside the GPU's fast on-chip memory, so the full 16-bit model never sits in main GPU memory. In hardware jargon, that fast on-chip memory is called "SRAM", and unpacking weights only during math is called "on-the-fly dequantization".
        # This allows us to fine-tune models in Phases 3, 4, and 5 on a single GPU that would otherwise require an expensive multi-GPU server.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

    # MOTIVATION: If we add new custom domain tags (like `<RECIPE>`) to the vocabulary, we must expand the model's word-lookup table and keep it trainable because new tags start out with no learned meaning.
    # Calling `resize_token_embeddings` adds new rows for the tags (rounded up to a multiple of 64 so GPU math hardware runs fast), and `modules_to_save` tells LoRA not to freeze the word-lookup table. In jargon, this is called "embedding resizing" and "unfreezing input/output embeddings".
    # Down the line, the model learns exact representations for our custom tags instead of breaking them into random sub-letters.
    extra_modules = None
    if add_domain_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<RECIPE>", "</RECIPE>"]})
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        extra_modules = ["embed_tokens", "lm_head"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # MOTIVATION: We want to attach LoRA adapters to every linear matrix in the transformer (both the attention layers and the feed-forward MLP layers) because adapting only the attention layers leaves out over half of the model's thinking capacity.
        # Setting `target_modules="all-linear"` automatically finds and wraps every eligible matrix in the network. In jargon, this is called "all-linear LoRA targeting".
        # Although it adds slightly more trainable parameters than attention-only LoRA, it closely matches the quality of full-model fine-tuning in downstream evaluations.
        target_modules="all-linear",
        # MOTIVATION: When we increase the LoRA bottleneck size `r` to give the adapter more capacity, standard LoRA divides updates by `r`, which makes updates too tiny and causes learning to stall.
        # Turning on `use_rslora=True` divides by the square root of `r` (`alpha / sqrt(r)`) instead, keeping update sizes healthy even at larger ranks. In jargon, this is called "Rank-Stabilized LoRA (rsLoRA)", while `use_dora` enables "Weight-Decomposed LoRA (DoRA)".
        # CONTEXT ON INNER WORKINGS: When `B @ A` multiplies across a wider bottleneck `r`, it adds `r` numbers together for every output. By basic statistics, adding `r` random numbers together makes their typical spread grow by the square root of `r` (`sqrt(r)`), not by `r`—so dividing by `sqrt(r)` keeps the adapter's output strength steady as `r` grows.
        # We can freely increase `r` (e.g., from 8 to 32 or 64) for harder tasks later without having to re-tune our learning rate from scratch.
        use_rslora=True,
        use_dora=False,
        modules_to_save=extra_modules,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def run_qlora_sft(output_dir: str = "./outputs/qlora_sft", max_steps: int = 20):
    model, tokenizer = load_4bit_qlora_model()
    model.print_trainable_parameters()

    train_ds = load_materials_sft_dataset(split="train[:200]")
    config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
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
    run_qlora_sft()
