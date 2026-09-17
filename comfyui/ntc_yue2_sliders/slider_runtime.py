"""Native YuE2 particle-slider inference classes, extracted from training source.
See source-provenance.json for exact file hashes. No adapter math is rewritten.
"""
from __future__ import annotations
from contextlib import contextmanager
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
FORMAT = "conceptmod-yue2-routed-particle-ar-v1"
UPSTREAM_REVISION = "ef1936f2ee39fe8de486a0f47a481c95f8d4da87"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=None,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv1d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
            self.lora_down = nn.Conv1d(
                in_dim,
                self.lora_dim,
                org_module.kernel_size,
                org_module.stride,
                org_module.padding,
                bias=False,
            )
            self.lora_up = nn.Conv1d(self.lora_dim, out_dim, 1, bias=False)

        elif "Conv" in org_module.__class__.__name__:  # 一応
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        #lora_init_weights_advanced(self.lora_down, self.lora_up, 5e-2, sparsity=0.9)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.seq_gain = None  # optional 1-D envelope over the sequence axis
        self.seq_gain_mode = "stretch"  # stretch | prefix
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _seq_gain_broadcast(self, x):
        gain = self.seq_gain
        if gain is None:
            return 1.0
        if x.dim() == 3:
            seq_dim, seq_len = 1, x.shape[1]
        elif x.dim() == 2:
            seq_dim, seq_len = 0, x.shape[0]
        else:
            return 1.0
        g = gain.to(device=x.device, dtype=x.dtype).reshape(-1)
        if g.numel() == 1:
            return g
        mode = getattr(self, "seq_gain_mode", "stretch")
        if mode == "prefix":
            if g.numel() >= seq_len:
                g = g[:seq_len]
            else:
                g = torch.cat((g, g[-1].expand(seq_len - g.numel())))
        elif g.numel() != seq_len:
            g = torch.nn.functional.interpolate(
                g.float().view(1, 1, -1),
                size=seq_len,
                mode="linear",
                align_corners=True,
            ).to(dtype=x.dtype).view(-1)
        shape = [1] * x.dim()
        shape[seq_dim] = seq_len
        return g.reshape(shape)

    def forward(self, x):
        # LoRA may be fp32/cuda while the host module is bf16 or CPU-offloaded.
        weight = self.lora_down.weight
        x_lora = x.to(device=weight.device, dtype=weight.dtype)
        delta = self.lora_up(self.lora_down(x_lora)).to(device=x.device, dtype=x.dtype)
        gain = self.multiplier * self.scale * self._seq_gain_broadcast(x)
        return self.org_forward(x) + delta * gain

def sound_only(text: str) -> str:
    """Reject named-reference syntax; prompts must describe sound directly."""
    if re.search(r"\b(?:in the style of|inspired by|sounds like)\b|\S\s+[—–]\s+meaning\b", text, re.I):
        raise ValueError("Describe instruments, voice, room and timing; remove named references")
    return text

def architecture(model) -> dict:
    config = model.config
    if getattr(config, "model_type", None) != "yue2":
        raise ValueError("Expected the official YuE2 AR–NAR model")
    return {key: getattr(config, key) for key in (
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size",
    )}

def attention_targets(model) -> dict[str, nn.Linear]:
    architecture(model)
    targets = {}
    for i, layer in enumerate(model.model.layers):
        for name in PROJECTIONS:
            module = getattr(layer.self_attn, name, None)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"Unsupported YuE2 projection: layer {i} {name}")
            targets[f"model.layers.{i}.self_attn.{name}"] = module
    if len(targets) != model.config.num_hidden_layers * len(PROJECTIONS):
        raise ValueError("YuE2 attention layout does not match its configuration")
    return targets

class _Adapter(LoRAModule):
    def forward(self, x):
        if self.multiplier == 0:
            return self.org_forward(x)
        return super().forward(x)

class YuE2Slider(nn.Module):

    def __init__(self, model, rank=8, alpha=8.0):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('rank and alpha must be positive')
        targets = attention_targets(model)
        if any((isinstance(getattr(m.forward, '__self__', None), _Adapter) for m in targets.values())):
            raise ValueError('A YuE2 slider is already attached')
        self.rank, self.alpha = (rank, float(alpha))
        self.architecture = architecture(model)
        self.target_names = list(targets)
        model.requires_grad_(False)
        self.adapters = nn.ModuleDict()
        for path, module in targets.items():
            key = path.replace('.', '-')
            adapter = _Adapter(key, module, multiplier=0.0, lora_dim=rank, alpha=alpha)
            adapter.apply_to()
            self.adapters[key] = adapter
        self.to(device=next(model.parameters()).device, dtype=torch.float32)

    @contextmanager
    def scaled(self, scale):
        if not math.isfinite(scale):
            raise ValueError('Slider scale must be finite')
        previous = [a.multiplier for a in self.adapters.values()]
        for adapter in self.adapters.values():
            adapter.multiplier = float(scale)
        try:
            yield
        finally:
            for adapter, value in zip(self.adapters.values(), previous):
                adapter.multiplier = value

def mlp(inputs, outputs, width):
    layers = []
    for n in (inputs, width, width):
        layers.extend((nn.Linear(n, width), nn.LeakyReLU(.2)))
    layers.append(nn.Linear(width, outputs))
    return nn.Sequential(*layers)

class RoutedMLP(nn.Module):
    """Every input uses the same softmax route at train and inference time."""
    def __init__(self, inputs, outputs, z_dim=4, width=48, router_width=16):
        super().__init__()
        self.router = mlp(inputs, z_dim, router_width)
        self.net = mlp(inputs + z_dim, outputs, width)

    def forward(self, x, particles):
        q = self.router(x)
        weights = (q @ particles.T / math.sqrt(particles.shape[1])).softmax(-1)
        z = weights @ particles
        return self.net(torch.cat((x, z), dim=-1))

shared = SimpleNamespace(RoutedMLP=RoutedMLP)

class ParticleProjection(_Adapter):
    def __init__(self, name, module, particles, rank, alpha):
        super().__init__(name, module, multiplier=0., lora_dim=rank, alpha=alpha)
        # The cloud belongs to the root exactly once, not to every projection.
        object.__setattr__(self, '_particles', particles)
        self.bridge = shared.RoutedMLP(rank, rank)

    def forward(self, x):
        if self.multiplier == 0:
            return self.org_forward(x)
        features = self.lora_down(x.to(device=self.lora_down.weight.device, dtype=torch.float32))
        routed = self.bridge(features, self._particles)
        delta = self.lora_up(routed).to(device=x.device, dtype=x.dtype)
        return self.org_forward(x) + delta * (self.multiplier * self.scale)

class ParticleSlider(YuE2Slider):
    def __init__(self, model, rank=8, alpha=8.):
        nn.Module.__init__(self)
        if rank != 8 or alpha != 8.:
            raise ValueError('The experimental particle format pins rank/alpha 8')
        targets = attention_targets(model)
        if any(isinstance(getattr(m.forward, '__self__', None), _Adapter) for m in targets.values()):
            raise ValueError('A YuE2 slider is already attached')
        self.rank, self.alpha = rank, float(alpha)
        self.architecture, self.target_names = architecture(model), list(targets)
        model.requires_grad_(False)
        device = next(model.parameters()).device
        self.particles = nn.Parameter(torch.randn(128, 4, device=device))
        self.adapters = nn.ModuleDict()
        for name, module in targets.items():
            key = name.replace('.', '-')
            adapter = ParticleProjection(key, module, self.particles, rank, alpha)
            adapter.apply_to()
            self.adapters[key] = adapter
        self.to(device=device, dtype=torch.float32)

    def save(self, path, metadata, *, state=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(metadata, format=FORMAT, rank=self.rank, alpha=self.alpha,
            architecture=self.architecture, targets=self.target_names,
            particles=128, particle_dim=4, bridge_width=48, router_width=16,
            upstream_revision=UPSTREAM_REVISION)
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False); sound_only(payload)
        state = self.state_dict() if state is None else state
        if state.keys() != self.state_dict().keys() or any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Invalid particle-slider export')
        save_file({k: v.detach().float().cpu().contiguous() for k, v in state.items()},
                  str(path), metadata={'conceptmod': payload})
        path.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')

    @classmethod
    def load(cls, model, path):
        with safe_open(str(path), framework='pt', device='cpu') as f:
            record = json.loads((f.metadata() or {}).get('conceptmod', '{}'))
        sound_only(json.dumps(record, ensure_ascii=False))
        required = dict(format=FORMAT, rank=8, alpha=8., particles=128, particle_dim=4,
                        bridge_width=48, router_width=16)
        if any(record.get(k) != v for k, v in required.items()) or record.get('dummy'):
            raise ValueError('Incompatible routed-particle checkpoint')
        if record.get('architecture') != architecture(model) or record.get('targets') != list(attention_targets(model)):
            raise ValueError('Particle-slider architecture/targets differ')
        state = load_file(str(path), device='cpu')
        expected = {'particles': (128, 4)}
        for name, module in attention_targets(model).items():
            prefix = 'adapters.' + name.replace('.', '-')
            expected.update({prefix + '.lora_down.weight': (8, module.in_features),
                prefix + '.lora_up.weight': (module.out_features, 8), prefix + '.alpha': ()})
            for branch, inputs, outputs, width in [('router', 8, 4, 16), ('net', 12, 8, 48)]:
                sizes = [inputs, width, width, width, outputs]
                for j, (a, b) in enumerate(zip(sizes, sizes[1:])):
                    expected[f'{prefix}.bridge.{branch}.{j*2}.weight'] = (b, a)
                    expected[f'{prefix}.bridge.{branch}.{j*2}.bias'] = (b,)
        if state.keys() != expected.keys() or any(tuple(state[k].shape) != shape for k, shape in expected.items()):
            raise ValueError('Incomplete routed-particle tensors')
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError('Non-finite routed-particle tensors')
        if any(float(v) != 8. for k, v in state.items() if k.endswith('.alpha')):
            raise ValueError('Particle-slider alpha differs')
        network = cls(model); network.load_state_dict(state, strict=True)
        return network, record
