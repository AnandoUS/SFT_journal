"""
Direct Preference Optimization (DPO):
Aligns model behavior on Conversational `(prompt, chosen, rejected)` message-list triples
derived from `iknow-lab/open-materials-guide-2024` using dual LoRA adapters (`dpo_train` vs.
frozen `sft_ref`) on a shared 4-bit base model without training a separate reward model:

    L_DPO = -log sigma( beta * [log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x))] )

Here 
$y_w$ (Winning): The safe refusal response (the chosen text).
$y_l$ (Losing): The hazardous, unguarded recipe (the rejected text).

To minimize the loss, the optimizer must maximize the entire value inside the sigmoid function ($\sigma$). 
This means simultaneously pull up the probability of the target refusal words and push down the probability of the hazardous recipe (rejected).
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "03_peft_mechanics"), str(ROOT / "04_safety_alignment")])

from qlora_pipeline import load_4bit_qlora_model  # noqa: E402
from safety_sft_pipeline import build_conversational_safety_pairs  # noqa: E402


def run_dpo_alignment(
    output_dir: str = "./outputs/dpo_safety",
    max_steps: int = 10,
    sft_adapter_path: Optional[str] = None,
):
    # MOTIVATION: DPO normally requires keeping two full copies of the model in GPU Video RAM (VRAM) at the same time (the active model being trained and a frozen starting copy for comparison) because the math compares new probabilities against original probabilities.
    # Instead of duplicating the whole 4-bit base model, we attach two tiny LoRA adapters—a frozen `"sft_ref"` copy from Phase 3 and a trainable `"dpo_train"` copy—and set `ref_model=None` so `DPOTrainer` switches between them. In jargon, this is called "PEFT-based reference sharing".
    # CONTEXT ON INNER WORKINGS: For each training example, `DPOTrainer` runs the conversation through the shared 4-bit base model twice on the same GPU: first with the trainable `"dpo_train"` adapter active to get the new probabilities, and second with the frozen `"sft_ref"` adapter active to read the original starting probabilities.
    # This trick cuts GPU memory usage nearly in half, allowing DPO to run on hardware that can only fit a single model.
    model, tokenizer = load_4bit_qlora_model(adapter_name="sft_ref")
    if sft_adapter_path is not None:
        model.load_adapter(sft_adapter_path, adapter_name="sft_ref", is_trainable=False)
        model.load_adapter(sft_adapter_path, adapter_name="dpo_train", is_trainable=True)
    else:
        model.add_adapter("dpo_train", model.peft_config["sft_ref"])
        with torch.no_grad():
            cloned = {
                k.replace(".sft_ref.", ".dpo_train."): v.clone()
                for k, v in model.state_dict().items()
                if ".sft_ref." in k
            }
            model.load_state_dict(cloned, strict=False)
    model.set_adapter("dpo_train")

    dpo_ds = Dataset.from_list(build_conversational_safety_pairs(split="train[:12]"))

    config = DPOConfig(
        output_dir=output_dir,
        model_adapter_name="dpo_train",
        ref_adapter_name="sft_ref",
        # MOTIVATION: While we want the model to prefer the safe answer (`chosen`) over the unsafe answer (`rejected`), we do not want it to drift so far from its original speaking style that it starts outputting gibberish.
        # Setting `beta=0.1` acts like a rubber band that penalizes the model if its probabilities stray too far from the starting model. In alignment jargon, this training method is called "Direct Preference Optimization (DPO)", and `beta` is the "KL-divergence regularization strength".
        # CONTEXT ON INNER WORKINGS: At every step, DPO checks how much more likely the model makes the safe refusal versus the unsafe recipe, compared against how likely those two answers were before DPO started. If the model tries to boost the refusal by wrecking its normal grammar, its probabilities drift too far from the starting model and `beta` applies a braking penalty. In alignment jargon, scoring answers by comparing new versus starting probabilities is called an "implicit reward".
        # A balanced `beta` gives us reliable safety refusals while keeping the model's chemistry explanations fluent and natural.
        beta=0.1,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        max_steps=max_steps,
        max_length=512,
        max_prompt_length=256,
        logging_steps=2,
        save_strategy="no",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=dpo_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    return model, tokenizer

if __name__ == "__main__":
    run_dpo_alignment()
