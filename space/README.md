---
title: YuE2 Concept Sliders
emoji: 🎚️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
short_description: Shape voice and genre with 16 native music sliders
python_version: "3.12"
startup_duration_timeout: 1h
license: cc-by-nc-4.0
models:
- ntc-ai/yue2-concept-sliders
- m-a-p/YuE2-3B
- m-a-p/YuE2-Vae
---

# YuE2 concept sliders

Generate a same-caption, same-lyrics, same-seed Off/On comparison with any of 16 voice and genre controls. Strength 0 bypasses the adapter; +1 is the trained positive endpoint. Browse all 64 release comparisons without consuming GPU quota in the listening tab.

[Model repo, weights and samples](https://huggingface.co/ntc-ai/yue2-concept-sliders) · [Native loading](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/USAGE.md) · [MiniMax Music 3 demo](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders)

These are the final EMA checkpoints from the 1,200-update routed-particle campaign. The native nonlinear adapters run only during AR planning/composition. The acoustic stage and VAE use frozen base weights. They require the included native loader and cannot be applied using a generic LoRA node.

Song length defaults to **Automatic**: the native YuE2 sampler decides when to emit its ending token. No seconds-based cap is applied in this mode. Choose **Limited** to set an optional maximum duration. Off and On use the same caption, lyrics and seed, but can end at different times.

YuE2 runs in BF16 through its native PyTorch CUDA-graph decoder. Graph capture includes the native particle projections; the adapter is removed before acoustic synthesis. Both the base model and VAE are loaded at startup for ZeroGPU. Requests are serialized; output paths are unique. Automatic mode retains the upstream 9,000-token safety guard (about six minutes). The GPU reservation is estimated separately and capped at 120 seconds **after** the Spaces hardware duration multiplier; it never changes the sampling limit. Long requests remain subject to ZeroGPU's execution-time and quota limits. Generation details report the active limit, semantic timing and any truncation. A live automatic-generation check produced a naturally ended 58.20-second song in 28.23 seconds with a 34-second scheduler request. Audio quality and lyric preservation are not guaranteed by the training diagnostics.

`yue2/` is the unmodified upstream inference package from revision `ef1936f2ee39fe8de486a0f47a481c95f8d4da87`, bundled with its licenses. Bundling avoids installing the upstream wheel's exact Hub pin into the platform-managed environment. Gradio is pinned to 6.17.3, the newest release compatible with the Hub range required by YuE2's tested Transformers 4.57.6; Gradio 6.18 and later require Hub 1.x. `slider_runtime.py` contains the exact native inference classes extracted from training source; `source-provenance.json` records their source hashes.

Base checkpoints: [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) and [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae). Model terms are in `upstream-licenses/MODEL_LICENSE`; upstream inference code uses Apache-2.0. Adapter checkpoints use CC BY-NC 4.0.
