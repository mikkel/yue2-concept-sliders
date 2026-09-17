"""Map our native particle projections to ComfyUI's fused YuE2 projections."""
from types import SimpleNamespace

import torch
from torch import nn
import comfy.model_prefetch

from .slider_runtime import ParticleSlider, PROJECTIONS


class ShapeLinear(nn.Linear):
    """Shape-only host for the native loader; its forward yields zero base output."""
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs, bias=False, device="meta")

    def forward(self, x):
        return x.new_zeros((*x.shape[:-1], self.out_features))


def load_particles(config, path):
    # Reuse the native loader's complete format/shape/finite validation and
    # exact ParticleProjection math, without allocating another base model.
    fields = ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size")
    host = nn.Module()
    host.register_parameter("device_anchor", nn.Parameter(torch.empty(0)))
    host.config = SimpleNamespace(model_type="yue2", **{k: getattr(config, k) for k in fields})
    host.model = nn.Module()
    host.model.layers = nn.ModuleList()
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    for _ in range(config.num_hidden_layers):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for name, inputs, outputs in (("q_proj", config.hidden_size, q),
                                     ("k_proj", config.hidden_size, kv),
                                     ("v_proj", config.hidden_size, kv),
                                     ("o_proj", q, config.hidden_size)):
            setattr(layer.self_attn, name, ShapeLinear(inputs, outputs))
        host.model.layers.append(layer)
    with torch.random.fork_rng(devices=[]):
        network, _ = ParticleSlider.load(host, path)
    return network


def projection_forward(original, branches):
    def forward(x, *args, **kwargs):
        output = original(x, *args, **kwargs)
        if len(branches) == 1:
            return output + branches[0](x).to(output)
        pieces = output.split([b.lora_up.out_features for b in branches], dim=-1)
        return torch.cat([piece + branch(x).to(piece)
                          for piece, branch in zip(pieces, branches)], dim=-1)
    return forward


class ParticleGeneration:
    """A clone-local ModelPatcher object patch, active only during AR sampling."""
    def __init__(self, host, original, network, strength):
        self.host, self.original = host, original
        self.network, self.strength = network, strength

    def __call__(self, *args, **kwargs):
        originals = []
        try:
            self.network.to(device=self.host.execution_device, dtype=torch.float32)
            for branch in self.network.adapters.values():
                object.__setattr__(branch, "_particles", self.network.particles)
            with self.network.scaled(self.strength):
                for index, layer in enumerate(self.host.model.layers):
                    branches = {name: self.network.adapters[f"model-layers-{index}-self_attn-{name}"]
                                for name in PROJECTIONS}
                    attention = layer.self_attn
                    if self.host.config.merged_qkv:
                        targets = [(attention.qkv_proj, [branches[n] for n in PROJECTIONS[:3]]),
                                   (attention.o_proj, [branches["o_proj"]])]
                    else:
                        targets = [(getattr(attention, n), [branches[n]]) for n in PROJECTIONS]
                    for module, parts in targets:
                        originals.append((module, module.forward))
                        module.forward = projection_forward(module.forward, parts)
                return self.original(*args, **kwargs)
        finally:
            # ComfyUI captures decode graphs; discard those before releasing
            # particle tensors or running the frozen acoustic prefix pass.
            comfy.model_prefetch.cleanup_prefetch_queues()
            for module, original in reversed(originals):
                module.forward = original
            self.network.to("cpu")
