# YuE2 Particle Slider for ComfyUI

Load our native YuE2 routed-particle checkpoints with **YuE2 Particle Slider (ntc-ai)**. The node preserves every router, MLP and the shared particle cloud. It does not convert or merge the adapters into ordinary LoRA weights.

## Install

For GitHub installation, clone `https://github.com/mikkel/yue2-concept-sliders.git` directly into `ComfyUI/custom_nodes/` and restart. The repository root registers this node. Use either that checkout or the ZIP method below; installing both registers duplicate nodes.

1. Update ComfyUI to a version with its built-in **YuE2 Generate Music** node. This package targets the official ComfyUI YuE2 integration.
2. [Download the custom-node ZIP](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/comfyui/ntc_yue2_sliders.zip) and extract its `ntc_yue2_sliders` folder into `ComfyUI/custom_nodes/`.
3. Download the native slider files from [weights/particle-gmix-1600-v2](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/weights/particle-gmix-1600-v2) into `ComfyUI/models/loras/yue2/`. The `.safetensors` file is sufficient; metadata is embedded.
4. Restart ComfyUI. No extra Python packages are required beyond current ComfyUI.

Use the [official ComfyUI YuE2 checkpoint](https://huggingface.co/Comfy-Org/YuE2), `yue2_3b_bf16.safetensors`, in `ComfyUI/models/checkpoints/`.

## Connect

Insert the slider node on the **CLIP** connection:

```text
Load Checkpoint (YuE2) · CLIP
    → YuE2 Particle Slider (ntc-ai) · CLIP
        → YuE2 Generate Music · clip
```

Select a native checkpoint such as `yue2/female_step1600.safetensors`. Strength `0` bypasses the node exactly, `0.5` is intermediate and `1` is the trained positive endpoint. Leave the existing MODEL, VAE, sampler and audio-output connections in your YuE2 workflow as they are. A complete workflow is included as `workflow.json`; drag it into ComfyUI after installing the node and weights. Its sampler and audio connections follow the [official YuE2 template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/audio_yue2_text2music.json), with our original release caption and lyrics. The upstream template's MIT notice is included in `WORKFLOW_LICENSE`.

Start with an empty ABC input to match the release's off-mode examples. YuE2 Generate Music chooses its ending naturally, up to its `max_duration` guard; leave that at `360` for the native six-minute guard. Connect its returned seconds to Empty YuE2 Latent Audio. If using Generate ABC, the same patched CLIP can feed that node too.

Describe sound through instruments, voice, room and timing. Avoid real music names in style and lyrics.

## Compatibility and checks

The node uses ComfyUI's clone-local ModelPatcher object patches. During AR generation it adds the native Q, K and V particle outputs to the corresponding slices of the fused QKV projection, and patches O separately. It restores all projection functions before acoustic prefix conditioning, including on exceptions. All adapter arithmetic remains FP32 and is cast back to the host output dtype.

The original `ParticleProjection` and complete checkpoint validator are included in `slider_runtime.py`; `source-provenance.json` records their source hashes. Shape-only meta-device projections allow validation without loading a second base model. The base weights remain under ComfyUI's normal device/offload management; the small adapter moves to the execution device for AR and back to CPU afterward.

Verified against ComfyUI commit `387f98aa2822f684b8597959a52a467d88cc4806`: native projection parity at strengths 0.5 and 1, exact zero bypass, real ComfyUI AR execution and ModelPatcher activation/restoration, unchanged acoustic conditioning, exception cleanup, malformed-checkpoint rejection, and validation of all 16 published checkpoints. These integration checks ran on CPU. Full GPU audio workflows, quantized bases, and third-party YuE2 node packs have not been validated.

Standard **Load LoRA** cannot apply these nonlinear checkpoints. Use this node with the original native weights. The code is MIT; adapter weights and the base model retain CC BY-NC 4.0 terms.
