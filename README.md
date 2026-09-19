# YuE2 Concept Sliders v2

Sixteen voice and genre controls for **YuE2-3B**. Version 2 publishes the final
**1,600-update EMA gmix teachers**, newly distilled rank-8 ordinary LoRAs,
ComfyUI exports and matched listening comparisons.

[Download v2](https://github.com/mikkel/yue2-concept-sliders/releases/tag/v2) · [Training math](MATH.md) · [Distillation method and measurements](DISTILLATION.md) · [Native usage](USAGE.md) · [Live Space — v1 particles](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders)

## Choose an adapter

| V2 option | Downloads | ComfyUI loading |
|---|---|---|
| Native routed particles | [All 16 teachers](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/weights/particle-gmix-1600-v2) | [YuE2 Concept Slider custom node](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/comfyui/ntc_yue2_sliders/README.md) |
| Ordinary rank-8 distills | [ComfyUI files](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/distilled-rank8-v2/comfyui) · [Native files](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/distilled-rank8-v2/native) | Standard **Load LoRA**, MODEL **0**, CLIP **1** |

[Standard LoRA workflow](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/workflow.json?download=true) · [Native particle workflow](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/comfyui/ntc_yue2_sliders/workflow.json?download=true) · [Download all ComfyUI distills](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/yue2-v2-comfyui-loras.zip?download=true)

Place files ending in `_comfyui.safetensors` in `ComfyUI/models/loras/`.
Strength 0 is Off; 1 is the trained positive endpoint; 0.5 is intermediate.
Negative strengths are unsupported. The ordinary students approximate the
nonlinear teachers. QKV is fused to rank 24; O remains rank 8.

## What changed in v2

The inference branch still routes through 128 learned four-dimensional particles
shared across 112 AR attention projections. Training now uses 512 seedbank
sources per control, 32-token shared histories, paired-edit normalization and
an eight-token, width-48 global-mix critic with four attention heads. Each
critic token mixes the complete hidden state. The bounded critic score,
paired shared noise, lazy gradient cap and particle variance/covariance penalty
define the adversarial game.

The catalog records the actual schedules: Metal retains the longer anneal;
Pop and Hip-Hop hold noise at 1; the remaining controls use the per-run 1.3×
edit-RMS hold. The final noise does not reach 0.03. Female and Male use the
retained expanded-cue h13 runs. All 16 are final EMA checkpoints at update
1600. This is a fixed-budget release; no listening-quality selection is claimed.
See [the full equations](MATH.md) and [training traces](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/evidence/particle-gmix-1600-v2).

## Controls and downloads

| Control | Sound | Particles | Ordinary LoRA |
|---|---|---|---|
| Female | One adult female lead with clear melodic phrasing | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/female_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/female_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/female_v2_distilled_rank8.safetensors?download=true) |
| Male | One adult male lead with clear melodic phrasing | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/male_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/male_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/male_v2_distilled_rank8.safetensors?download=true) |
| Pop | Clear hooks, crisp drums and a polished chorus | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/pop_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/pop_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/pop_v2_distilled_rank8.safetensors?download=true) |
| Hip-Hop | Rapped verses, deep sub bass and nimble hats | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/hiphop_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/hiphop_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/hiphop_v2_distilled_rank8.safetensors?download=true) |
| R&B | Warm keys, deep pocket and fluid vocal phrasing | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/rnb_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/rnb_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/rnb_v2_distilled_rank8.safetensors?download=true) |
| Indie Rock | Chiming guitars, moving bass and a human drum kit | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/indie-rock_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/indie-rock_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/indie-rock_v2_distilled_rank8.safetensors?download=true) |
| Pop Punk | Palm-muted power chords and driving chorus drums | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/pop-punk_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/pop-punk_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/pop-punk_v2_distilled_rank8.safetensors?download=true) |
| Metal | Heavy guitar riffs, tight kicks and big melodic choruses | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/metal_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/metal_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/metal_v2_distilled_rank8.safetensors?download=true) |
| Country | Acoustic strum, twangy fills and an easy backbeat | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/country_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/country_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/country_v2_distilled_rank8.safetensors?download=true) |
| Acoustic Folk | Fingerpicked strings and a warm small-room performance | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/acoustic-folk_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/acoustic-folk_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/acoustic-folk_v2_distilled_rank8.safetensors?download=true) |
| House | Steady club kick, offbeat hats and a rolling bass line | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/house_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/house_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/house_v2_distilled_rank8.safetensors?download=true) |
| Disco Funk | Elastic bass, clipped guitar and bright dance-floor strings | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/disco-funk_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/disco-funk_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/disco-funk_v2_distilled_rank8.safetensors?download=true) |
| K-pop | Sharp synth hooks, tight edits and a big chorus lift | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/kpop_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/kpop_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/kpop_v2_distilled_rank8.safetensors?download=true) |
| Reggaeton | Dembow drums, rounded sub bass and clipped melodic hooks | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/reggaeton_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/reggaeton_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/reggaeton_v2_distilled_rank8.safetensors?download=true) |
| Afrobeats | Interlocking percussion, melodic bass and buoyant guitar | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/afrobeats_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/afrobeats_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/afrobeats_v2_distilled_rank8.safetensors?download=true) |
| Lo-fi | Soft swung drums, mellow keys and gentle tape warmth | [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/weights/particle-gmix-1600-v2/lofi_step1600.safetensors?download=true) | [ComfyUI](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/comfyui/lofi_v2_distilled_rank8_comfyui.safetensors?download=true) · [Native](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/distilled-rank8-v2/native/lofi_v2_distilled_rank8.safetensors?download=true) |

## Listen to v2

[Open the matched v2 comparisons on Hugging Face](https://huggingface.co/ntc-ai/yue2-concept-sliders#listen-to-v2) · [Full samples](SAMPLES.md).
The model card has 48 audio players, with **Off**, **Particles** and **Distill**
in each row. Each group keeps caption, lyrics and seed fixed. Distill applies
the ordinary LoRA during composition. Original recordings and sidecars remain
in the sample folders. These are approximately 20-second native GPU excerpts;
full ComfyUI GPU audio generation remains unvalidated.

## Reproduction and previous versions

[Release tools](release-tools/README.md) create the ordinary distills from the
versioned teacher catalog, then convert and validate the exports. The
[v2 manifest](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/particle-gmix-1600-v2/release-manifest.json) records file hashes;
each student records its exact teacher hash. Training and evaluation prompts
contain sound descriptions. Weights retain CC BY-NC 4.0 terms.

[V1 particle weights](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/weights/particle-1200-v1) · [V1 ordinary LoRAs](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/distilled-rank8-v1) · [V1 README and math at their original revision](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/f7c2e7ab024b6505f009ea9c6f7deb36e3373bc0).
The live Space continues to run its pinned v1 particle deployment.
