"""
Activation Patching & Scrubbing:
Renders safe and unsafe Conversational `messages` prompts (built from `iknow-lab/open-materials-guide-2024`)
through the model's chat template and swaps residual-stream activations layer-by-layer to localize
which transformer layers causally drive compliance vs. refusal behavior.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "02_naive_sft"), str(ROOT / "04_safety_alignment")])

from run_sft_trainer import load_materials_sft_dataset, load_model_and_tokenizer  # noqa: E402
from safety_sft_pipeline import build_conversational_safety_pairs  # noqa: E402


def get_decoder_layers(model):
    # MOTIVATION: We need a single helper function to locate the list of transformer layers inside any model because wrapping a model in LoRA or QLoRA buries the layers several levels deeper under extra wrapper attributes.
    # Checking `base_model`, `model.layers`, and `transformer.h` unwraps those outer shells and returns the raw list of layers. In jargon, these stacked processing blocks are called "decoder layers" or "transformer blocks".
    # Down the line, activation patching, representation engineering, and red-team steering scripts can all reuse this one function on both base models and LoRA models without breaking.
    base = getattr(model, "base_model", model)
    if hasattr(base, "model") and hasattr(base.model, "model"):
        return base.model.model.layers
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers
    return base.transformer.h


def cache_last_token_activations(model, tokenizer, prompt_messages: list[dict]):
    # MOTIVATION: We want to record the internal numbers produced by each layer at the very end of a safe conversational prompt because in a left-to-right language model, that final token position holds the summary of the entire conversation right before the assistant starts speaking.
    # Rendering `prompt_messages` with `add_generation_prompt=True` appends the `<|im_start|>assistant\n` turn header and attaches a temporary listener (`register_forward_hook`) to save each layer's final vector. In jargon, the highway of internal vectors flowing through the layers is called the "residual stream", and saving them is called "activation caching".
    # CONTEXT ON INNER WORKINGS: Because left-to-right language models only allow each word to look backward at earlier words, the very last token of the prompt (right after the `assistant` turn marker) is the only position that has read the entire user request and is about to pick the assistant's first word. In jargon, restricting words to only look backward at earlier words is called "causal attention" (or "causal masking").
    # Once we have this saved snapshot of how the model thinks right before answering a safe prompt, we can transplant those thoughts into an unsafe prompt in the next function.
    rendered_prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    layers = get_decoder_layers(model)
    cache = {}
    handles = []

    for idx, layer in enumerate(layers):
        def make_hook(layer_idx):
            def hook(_module, _inp, output):
                hidden = output[0] if isinstance(output, tuple) else output
                cache[layer_idx] = hidden[:, -1, :].detach().clone()
            return hook

        handles.append(layer.register_forward_hook(make_hook(idx)))

    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()
    return cache


def patch_layer_and_score_refusal(
    model,
    tokenizer,
    unsafe_prompt_messages: list[dict],
    safe_cache: dict,
    layer_idx: int,
    refusal_token: str = "I",
) -> float:
    # MOTIVATION: We want to find out which exact layer inside the model decides whether to comply with or refuse a dangerous request because blindly editing all layers in later safety steps is wasteful and can hurt general model quality.
    # Here we run the model on an unsafe conversational prompt, but at one specific layer (`layer_idx`) we overwrite its final-token vector with the saved vector from the safe prompt and measure how much the probability of starting a refusal (`"I"`, as in `"I cannot..."`) changes. In mechanistic interpretability jargon, this surgical swap is called "activation patching" (or "causal tracing").
    # CONTEXT ON INNER WORKINGS: Layers in a transformer pass a shared internal summary vector from one layer to the next like a relay baton. Replacing that vector at Layer `k` erases the unsafe decision formed in Layers `0` through `k` and lets Layers `k + 1` onward finish generating as if they had read the safe prompt.
    # The specific layer number that causes the biggest shift in refusal probability tells us the exact layer number to target for LoRRA, ReFT, Probing and Steering.
    rendered_unsafe = tokenizer.apply_chat_template(
        unsafe_prompt_messages, tokenize=False, add_generation_prompt=True
    )
    layers = get_decoder_layers(model)
    refusal_id = tokenizer.encode(refusal_token, add_special_tokens=False)[0]

    def patch_hook(_module, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[:, -1, :] = safe_cache[layer_idx].to(hidden.device, hidden.dtype)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = layers[layer_idx].register_forward_hook(patch_hook)
    inputs = tokenizer(rendered_unsafe, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
        log_prob = torch.log_softmax(logits, dim=-1)[0, refusal_id].item()
    handle.remove()
    return log_prob


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    safe_prompt_msgs = load_materials_sft_dataset(split="train[:1]")[0]["messages"][:2]
    unsafe_prompt_msgs = build_conversational_safety_pairs(split="train[:1]")[0]["prompt"]

    safe_cache = cache_last_token_activations(model, tokenizer, safe_prompt_msgs)
    for layer_idx in range(min(6, len(get_decoder_layers(model)))):
        lp = patch_layer_and_score_refusal(
            model, tokenizer, unsafe_prompt_msgs, safe_cache, layer_idx
        )
        print(f"Patched Layer {layer_idx:02d} -> Refusal starter log-prob: {lp:.4f}")
