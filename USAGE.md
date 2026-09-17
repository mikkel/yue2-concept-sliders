# Native YuE2 slider loading

Download the release with `huggingface_hub.snapshot_download("ntc-ai/yue2-concept-sliders")`. Every entry in `catalog.json` points to its native checkpoint, sidecar, prompt pairs and listening examples.

Use the official [YuE2 runtime](https://github.com/multimodal-art-projection/YuE) at revision `ef1936f2ee39fe8de486a0f47a481c95f8d4da87` (version 0.1.6). Its tested inference dependencies include PyTorch 2.10.0 and Transformers 4.57.6. Install in a separate environment following the upstream instructions. The included `slider_runtime.py` needs only PyTorch and safetensors; it does not import training code. A working, pinned deployment is also provided in the [Space source](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders/tree/main).

Save the following script beside the downloaded `slider_runtime.py`, with your own `lyrics.txt` containing section labels such as `[verse]` and `[chorus]`:

```python
from pathlib import Path
import json
import torch
from huggingface_hub import hf_hub_download
from yue2 import YuE2Pipeline
from yue2.protocol import SongRequest
from slider_runtime import YuE2Slider, attention_targets

repo = "ntc-ai/yue2-concept-sliders"
catalog = json.loads(Path(hf_hub_download(repo, "catalog.json")).read_text())
entry = next(s for s in catalog["sliders"] if s["id"] == "metal")
path = hf_hub_download(repo, entry["weights"])

with YuE2Pipeline.from_pretrained(
    catalog["base_model"], vae=catalog["vae"],
    revision=catalog["base_revision"], vae_revision=catalog["vae_revision"],
    device="cuda", backend="torch", quantization="none",
    offload_ar=False, memory_budget_gib=24,
) as pipe, torch.inference_mode():
    model = pipe._load_model()
    targets = attention_targets(model)
    originals = {name: module.forward for name, module in targets.items()}
    try:
        adapter, metadata = YuE2Slider.load(model, path)
        assert metadata["model_identity"] == pipe.weights["mot"]
        adapter.eval().requires_grad_(False)
        request = SongRequest(
            style="Piano chords, bass and a light drum kit. One clear lead singer.",
            lyrics=Path("lyrics.txt").read_text(), seed=1709, cot="off",
        )
        with adapter.scaled(1.0):
            plan = pipe.plan(request=request)
            semantic = pipe.generate_semantic(plan)
    finally:
        # Remove hooks before acoustic synthesis and on every error path.
        for name, module in targets.items():
            module.forward = originals[name]
    latents = pipe.synthesize(semantic)
    audio = pipe.decode(latents)
    import soundfile as sf
    sf.write("metal-on.wav", audio, 48000, subtype="PCM_24")
```

For Off/On comparisons, repeat with the same request and seed at scales `0` and `1`. Scale `0.5` interpolates the adapter residual; negative strength is not a trained opposite. The example uses native automatic stopping with the upstream 9,000-token guard. For an optional approximately 20-second maximum, pass `sampling={"max_tokens": 500, "min_tokens": 200}` to `generate_semantic`; that cap can cut a musical phrase short.

Use `backend="torch-eager"` for the simple reference path, or `backend="torch"` for the upstream CUDA-graph path now tested in the Space. Keep `quantization="none"` and `offload_ar=False`. Adapter hooks must be active during AR generation, including graph capture, and removed before NAR synthesis. For the original particle checkpoints, standard LoRA and PEFT loaders do not implement the routed MLP or shared particle cloud. Do not drop those tensors or merge the checkpoint into the base model.

## Experimental ordinary LoRAs

Download the native archive from the [distillation release](https://github.com/mikkel/yue2-concept-sliders/releases/tag/distilled-rank8-20260917).
Use the current `slider_runtime.py` and the same example above, replacing the
checkpoint path with your local `metal_distilled_refined_rank8.safetensors`.
`YuE2Slider.load` validates and loads both the original particle format and the
ordinary LoRA format. Keep the adapter removed before acoustic synthesis for
AR-only comparisons. See [DISTILLATION.md](DISTILLATION.md) for approximation
quality and the broader acoustic-prefix scope of standard ComfyUI LoRA loading.

## ComfyUI

Install our [YuE2 Concept Slider custom node](comfyui/ntc_yue2_sliders/README.md), then connect it between the native YuE2 checkpoint's CLIP output and YuE2 Generate Music. Use the original files under `weights/particle-1200-v1/`; no conversion is needed. The [ZIP](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/comfyui/ntc_yue2_sliders.zip) includes the node and a complete workflow. Original particle files require this custom node. The distilled ordinary files instead use standard Load LoRA and the experimental release’s workflow. The node passes native-math and actual ComfyUI AR/patcher integration checks on CPU; full GPU audio workflows have not been validated.

Use sound descriptions in captions and original lyric sheets. The released prompts and checkpoint metadata were checked for named music references.
