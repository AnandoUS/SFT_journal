import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base_layer
        # MOTIVATION: We want to lock the original weight matrix (`W_0`) so it cannot change during training because updating billions of original weights requires storing huge optimizer tracking tables in GPU Video RAM (VRAM).
        # Setting `requires_grad = False` tells PyTorch not to compute or store updates for the original layer. In jargon, locking weights so they stay fixed is called "freezing the base model".
        # Because the base weights never change, we save massive amounts of memory and can swap different small trained adapters in and out on top of the same base model later.
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        # MOTIVATION: We want to represent the weight update using two skinny matrices (`A` and `B`, with a tiny bottleneck width `r=8`) instead of one giant square matrix because two skinny matrices have less than 2% as many numbers to train.
        # We fill `A` with small random numbers and fill `B` with exact zeros so that at the start of training (`B @ A == 0`), our side-branch adds nothing and the model behaves identically to the untouched base model. In jargon, this bottleneck trick is called "Low-Rank Adaptation (LoRA)", and starting with `B=0` is called "zero-initialization".
        # Starting as a clean no-op prevents random initial adapter weights from scrambling the base model's knowledge on step 1.
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return self.base(x) + delta

    def merge_weights(self) -> nn.Linear:
        # MOTIVATION: Once training is finished, we do not want to run both the original matrix and the side-branch matrices separately during live use because running two branches adds extra math delay to every generated word.
        # Multiplying `B @ A` and adding the result directly into the original weight matrix `W_0` bakes our learned changes permanently into the main layer. In jargon, this is called "merging adapter weights".
        # When we deploy the model at the end of the workflow, it runs at the exact same speed as the original base model with zero extra latency.
        with torch.no_grad():
            self.base.weight.add_((self.lora_B @ self.lora_A) * self.scaling)
        return self.base

d_in, d_out, rank = 1024, 1024, 8
linear = nn.Linear(d_in, d_out, bias=False)
lora_layer = LoRALinear(linear, r=rank, alpha=16.0)

total_params = sum(p.numel() for p in lora_layer.parameters())
trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
print(f"Total params: {total_params:,} | Trainable LoRA params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

x = torch.randn(2, 16, d_in)
assert torch.allclose(linear(x), lora_layer(x)), "At init (B=0), LoRA output must equal base output"
        return self.base
