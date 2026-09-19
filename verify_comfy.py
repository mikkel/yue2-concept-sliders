"""CPU integration check against an actual current ComfyUI checkout.

Run: python verify_comfy.py /path/to/ComfyUI /path/to/slider-snapshot
"""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace

if len(sys.argv) != 3:
    raise SystemExit("Usage: python verify_comfy.py COMFYUI_CHECKOUT SLIDER_SNAPSHOT")

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(sys.argv[2]).resolve()
COMFY = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(COMFY), str(HERE / "comfyui")]
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options
comfy.options.enable_args_parsing()

import torch
from torch import nn
from safetensors.torch import load_file, save_file
import folder_paths
from comfy.text_encoders.yue2 import YuE2TEModel
from comfy.model_patcher import ModelPatcher
from ntc_yue2_sliders import YuE2ConceptSlider
from ntc_yue2_sliders.adapter import ParticleGeneration, load_particles
from ntc_yue2_sliders.slider_runtime import ParticleSlider, PROJECTIONS

torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.manual_seed(71)
device = torch.device("cpu")
tiny = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1,
            merged_mlp=False, fixed_kv=False)
te = YuE2TEModel(device=device, dtype=torch.float32, config=tiny)
for name, parameter in te.named_parameters():
    nn.init.normal_(parameter, std=.03)
    if "norm" in name and name.endswith("weight"):
        nn.init.ones_(parameter)


def reference_host(te):
    host = nn.Module()
    fields = ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size")
    host.config = SimpleNamespace(model_type="yue2", **{k: getattr(te.config, k) for k in fields})
    host.model = nn.Module()
    host.model.layers = nn.ModuleList()
    for layer in te.model.layers:
        native = nn.Module()
        native.self_attn = nn.Module()
        weights = layer.self_attn.qkv_proj.weight.detach().split([256, 128, 128])
        for name, weight in zip(PROJECTIONS, [*weights, layer.self_attn.o_proj.weight]):
            projection = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            projection.weight.data.copy_(weight)
            setattr(native.self_attn, name, projection)
        host.model.layers.append(native)
    return host


class ClipFixture:
    def __init__(self, te, patcher=None):
        self.cond_stage_model = te
        self.patcher = patcher or ModelPatcher(te, device, device)

    def clone(self):
        return ClipFixture(self.cond_stage_model, self.patcher.clone())


with tempfile.TemporaryDirectory(prefix="yue2-comfy-test-") as temp:
    path = Path(temp) / "test.safetensors"
    reference = reference_host(te)
    native = ParticleSlider(reference)
    for branch in native.adapters.values():
        nn.init.normal_(branch.lora_up.weight, std=.05)
    native.save(path, {"purpose": "tiny CPU integration fixture"})
    folder_paths.add_model_folder_path("loras", temp)
    clip = ClipFixture(te)
    node = YuE2ConceptSlider()
    assert node.apply(clip, path.name, 0)[0] is clip
    x = torch.randn(2, 3, 16)
    x_o = torch.randn(2, 3, 256)
    baseline = [(layer.self_attn.qkv_proj(x), layer.self_attn.o_proj(x_o)) for layer in te.model.layers]
    original_methods = [(layer.self_attn.qkv_proj.forward, layer.self_attn.o_proj.forward) for layer in te.model.layers]
    comparisons = 0
    for strength in (.5, 1):
        def compare_projections():
            with native.scaled(strength):
                for i, layer in enumerate(te.model.layers):
                    source = reference.model.layers[i].self_attn
                    expected = torch.cat([getattr(source, n)(x) for n in PROJECTIONS[:3]], dim=-1)
                    torch.testing.assert_close(layer.self_attn.qkv_proj(x), expected, atol=1e-7, rtol=1e-6)
                    torch.testing.assert_close(layer.self_attn.o_proj(x_o), source.o_proj(x_o), atol=1e-7, rtol=1e-6)
                    assert not torch.equal(layer.self_attn.qkv_proj(x), baseline[i][0])
        ParticleGeneration(te, compare_projections, load_particles(te.config, path), strength)()
        comparisons += 2 * len(te.model.layers)
        for layer, methods, outputs in zip(te.model.layers, original_methods, baseline):
            assert layer.self_attn.qkv_proj.forward == methods[0]
            assert layer.self_attn.o_proj.forward == methods[1]
            assert torch.equal(layer.self_attn.qkv_proj(x), outputs[0])

    # Real ComfyUI ModelPatcher clone activation, AR generation, and acoustic pass.
    patched = node.apply(clip, path.name, 1)[0]
    assert not clip.patcher.object_patches
    assert "_generate" in patched.patcher.object_patches
    original_generate = te._generate
    acoustic_before = te._acoustic_conditioning([151643, 151851], [151900, 151901], torch.float32)
    patched.patcher.patch_model(load_weights=False)
    ids, truncated = te._generate([151643, 151851], 7, 2, "semantic", torch.float32,
        temperature=0, top_p=.95, top_k=100, repetition_penalty=1.2, penalty_window=50, min_tokens=2)
    assert len(ids) == 2 and truncated
    acoustic_after = te._acoustic_conditioning([151643, 151851], [151900, 151901], torch.float32)
    assert torch.equal(acoustic_before[0], acoustic_after[0])
    assert acoustic_before[1] == acoustic_after[1]
    patched.patcher.unpatch_model(unpatch_weights=False)
    assert te._generate == original_generate
    for layer, methods in zip(te.model.layers, original_methods):
        assert layer.self_attn.qkv_proj.forward == methods[0]
        assert layer.self_attn.o_proj.forward == methods[1]

    def fail():
        raise RuntimeError("intentional cancellation fixture")
    try:
        ParticleGeneration(te, fail, load_particles(te.config, path), 1)()
    except RuntimeError as exc:
        assert "intentional" in str(exc)
    else:
        raise AssertionError("Expected cancellation")
    for layer, methods in zip(te.model.layers, original_methods):
        assert layer.self_attn.qkv_proj.forward == methods[0]
        assert layer.self_attn.o_proj.forward == methods[1]

    # Reject missing particle tensors using the native checkpoint validator.
    from safetensors import safe_open
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    state = load_file(path)
    del state["particles"]
    bad = Path(temp) / "missing-cloud.safetensors"
    save_file(state, bad, metadata=metadata)
    try:
        load_particles(te.config, bad)
    except ValueError as exc:
        assert "Incomplete" in str(exc)
    else:
        raise AssertionError("Accepted an incomplete checkpoint")

    # Validate all real released checkpoints against the real ComfyUI config.
    full = YuE2TEModel(device="meta", dtype=torch.bfloat16)
    catalog = json.loads((HERE / "catalog.json").read_text())
    for spec in catalog["sliders"]:
        network = load_particles(full.config, WEIGHTS / spec["weights"])
        assert len(network.adapters) == 112
        assert network.particles.shape == (128, 4)

result = dict(projection_comparisons=comparisons, real_checkpoints=16,
    zero_bypass=True, clone_isolation=True, real_comfy_ar=True,
    acoustic_conditioning_unchanged=True, exception_restoration=True,
    rejects_missing_particles=True, device="CPU", comfy_checkout=str(COMFY))
(HERE / "verification").mkdir(exist_ok=True)
(HERE / "verification/comfyui.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
