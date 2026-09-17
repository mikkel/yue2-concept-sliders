"""YuE2 concept sliders: native routed-particle adapters on AR composition."""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces
import torch

from contextlib import contextmanager
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import tempfile
import time

import gradio as gr
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from yue2 import YuE2Pipeline
from yue2.modeling_vae import YuE2VAE
from yue2.protocol import SongRequest

from slider_runtime import ParticleSlider, attention_targets, sound_only
from sound_guard import validate_sound

REPO = "ntc-ai/yue2-concept-sliders"
RELEASE = json.loads(Path("catalog.json").read_text())
CATALOG = {s["id"]: s for s in RELEASE["sliders"]}
MODEL_REVISION = os.environ.get("SLIDER_REVISION", "fad542cb5ff2a0b09339a289398cca291e5cae7a")
MAX_SEED = 2**31 - 1

print("Loading pinned YuE2 base and audio decoder...", flush=True)
# Construct on CPU to avoid the upstream CUDA memory-limit call during ZeroGPU
# startup. The actual modules are placed eagerly on CUDA for ZeroGPU packing.
PIPE = YuE2Pipeline.from_pretrained(
    RELEASE["base_model"], vae=RELEASE["vae"], revision=RELEASE["base_revision"],
    vae_revision=RELEASE["vae_revision"], device="cpu", backend="torch",
    quantization="none", offload_ar=False, memory_budget_gib=40, progress=False,
)
MODEL = PIPE._load_model().eval().requires_grad_(False).to("cuda")
VAE = YuE2VAE.from_pretrained(PIPE.vae_dir, decoder_only=True, device="cpu",
                            local_files_only=True).eval().requires_grad_(False).to("cuda")
PIPE.device = torch.device("cuda")
PIPE._vae = VAE
FILES = {sid: hf_hub_download(REPO, item["weights"], revision=MODEL_REVISION)
         for sid, item in CATALOG.items()}
for sid, path in FILES.items():
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["conceptmod"])
    if metadata["model_identity"] != PIPE.weights["mot"]:
        raise ValueError(f"Base model identity does not match {sid}")
    validate_sound(json.dumps(metadata))
print(f"YuE2 base, decoder and {len(FILES)} native sliders ready.", flush=True)


@contextmanager
def applied_slider(concept: str, strength: float):
    targets = attention_targets(MODEL)
    originals = {name: module.forward for name, module in targets.items()}
    try:
        if strength == 0:
            yield
        else:
            network, _ = ParticleSlider.load(MODEL, FILES[concept])
            network.eval().requires_grad_(False)
            with network.scaled(strength):
                yield
    finally:
        for name, module in targets.items():
            module.forward = originals[name]


def render(concept, strength, caption, lyrics, seconds, steps, seed):
    request = SongRequest(style=caption, lyrics=lyrics, cot="off", seed=seed)
    sampling = None
    max_tokens = PIPE.generation_config.semantic.max_tokens
    if seconds is not None:
        max_tokens = min(max_tokens, int(seconds * 25))
        sampling = {"max_tokens": max_tokens, "min_tokens": min(200, max_tokens)}
    old_config = PIPE.generation_config
    PIPE.generation_config = replace(old_config, ode_steps=steps)
    try:
        start = time.perf_counter()
        print(json.dumps(dict(event="render_start", strength=strength,
                              max_tokens=max_tokens)), flush=True)
        with applied_slider(concept, strength):
            # Upstream graph capture calls the unfused projections with our
            # particle hooks active; each graph is closed before restoration.
            plan = PIPE.plan(request=request)
            semantic = PIPE.generate_semantic(plan, sampling=sampling)
        print(json.dumps(dict(event="semantic_ready", strength=strength,
            tokens=len(semantic.tokens), truncated=semantic.truncated,
            timing=semantic.timing)), flush=True)
        # Restore every AR projection before the frozen acoustic stage.
        latents = PIPE.synthesize(semantic)
        # Same native tiled VAE path as PIPE.decode, keeping both small models
        # resident to avoid offload/reload between the two comparison renders.
        z = torch.as_tensor(latents, dtype=torch.float32).T.unsqueeze(0)
        decoded = VAE.decode_tiled(z, core_frames=PIPE.vae_core_frames,
                                  halo_frames=16, output_device="cpu")
        if not torch.isfinite(decoded).all():
            raise RuntimeError("YuE2 decoder returned non-finite audio")
        audio = decoded[0].float().clamp(-1, 1).T.contiguous().numpy()
        if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) <= 1e-6:
            raise RuntimeError("YuE2 returned empty, silent or non-finite audio")
        path = Path(tempfile.mkdtemp(prefix="yue2-slider-")) / "audio.wav"
        sf.write(path, audio, 48000, subtype="PCM_16")
        return str(path), dict(duration_seconds=round(len(audio) / 48000, 3),
                              duration_mode="automatic" if seconds is None else "limited",
                              max_tokens=max_tokens, semantic_tokens=len(semantic.tokens),
                              semantic_truncated=semantic.truncated,
                              semantic_timing=semantic.timing,
                              elapsed_seconds=round(time.perf_counter() - start, 2))
    finally:
        PIPE.generation_config = old_config


def gpu_duration(concept, strength, caption, lyrics, seconds=30, steps=16,
                 seed=1709, randomize_seed=False, compare=True,
                 duration_mode="Automatic", *args, **kwargs):
    count = 2 if compare and float(strength) != 0 else 1
    if duration_mode == "Automatic":
        # Scheduling estimate only: never convert it into a sampling limit.
        words = sum(len(line.split()) for line in lyrics.splitlines()
                    if line.strip() and not line.strip().startswith("["))
        estimated_seconds = max(30, words * .7 + 15)
    else:
        estimated_seconds = float(seconds)
    # CUDA graph decode batches the CFG branches and removes per-token
    # Python dispatch. Allow capture overhead for each independent song.
    # Live automatic On: 58.20 s audio / 16 steps in 28.23 s GPU time.
    estimate = count * (15 + estimated_seconds * .6 + float(steps) * .1)
    # Spaces 0.51 applies a hardware duration factor (1.5 on the current
    # large GPU) *after* this callback. Bound the final scheduler request,
    # not just the value returned here. This never limits semantic tokens.
    from datetime import timedelta
    from spaces.zero.client import get_duration_seconds
    factor = float(get_duration_seconds(timedelta(seconds=1), None))
    budget = min(120, max(30, math.ceil(estimate)))
    declared = max(1, math.floor(budget / factor))
    print(json.dumps(dict(event="gpu_reservation", duration_mode=duration_mode,
        comparisons=count, declared_seconds=declared, duration_factor=factor,
        requested_seconds=round(declared * factor))), flush=True)
    return declared


@spaces.GPU(duration=gpu_duration)
@torch.inference_mode()
def generate(concept: str, strength: float, caption: str, lyrics: str,
             seconds: float = 30, steps: int = 16, seed: int = 1709,
             randomize_seed: bool = False, compare: bool = True,
             duration_mode: str = "Automatic",
             progress=gr.Progress()):
    """Generate a YuE2 song with a voice or genre slider, optionally paired
    with a same-caption, same-lyrics, same-seed base render. Strength ranges
    from zero (base) to one (trained positive endpoint). Automatic duration
    uses the native stopping behavior and ignores seconds. Limited duration
    uses seconds as an upper bound, which can cut a phrase short.
    Returns Off/On WAVs and details, including whether the token guard was hit.
    """
    if concept not in CATALOG:
        raise gr.Error("Choose a control from the catalog.")
    if not math.isfinite(float(strength)) or not 0 <= float(strength) <= 1:
        raise gr.Error("Strength must be between 0 and 1.")
    if duration_mode not in ("Automatic", "Limited"):
        raise gr.Error("Choose Automatic or Limited song length.")
    if duration_mode == "Limited" and (
        not math.isfinite(float(seconds)) or not 2 <= float(seconds) <= 360
    ):
        raise gr.Error("Choose a duration limit between 2 and 360 seconds.")
    duration = None if duration_mode == "Automatic" else float(seconds)
    if not 4 <= int(steps) <= 32:
        raise gr.Error("Audio steps must be between 4 and 32.")
    caption, lyrics = caption.strip(), lyrics.strip()
    if not caption or not lyrics or len(caption) > 2500 or len(lyrics) > 4000:
        raise gr.Error("Enter a sound description (up to 2,500 characters) and lyrics (up to 4,000).")
    try:
        validate_sound(caption + "\n" + lyrics)
        sound_only(caption + "\n" + lyrics)
    except ValueError as exc:
        raise gr.Error("Describe instruments, voice, room and timing; remove named music references.") from exc
    seed = random.randrange(MAX_SEED) if randomize_seed else int(seed)
    if not 0 <= seed <= MAX_SEED:
        raise gr.Error("Seed must be between 0 and 2147483647.")
    start = time.perf_counter()
    off, off_details = None, None
    if compare or float(strength) == 0:
        progress(.05, desc="Composing the base song")
        off, off_details = render(concept, 0, caption, lyrics, duration, int(steps), seed)
    if float(strength) == 0:
        on, on_details = off, off_details
    else:
        progress(.5 if compare else .05, desc=f"Composing with {CATALOG[concept]['label']}")
        on, on_details = render(concept, float(strength), caption, lyrics, duration, int(steps), seed)
    elapsed = round(time.perf_counter() - start, 2)
    print(json.dumps(dict(event="generated", concept=concept, strength=strength, seed=seed,
                          elapsed_seconds=elapsed, off=off_details, on=on_details)), flush=True)
    progress(1, desc="Ready to listen")
    return off, on, seed, dict(concept=concept, strength=strength, seed=seed,
        duration_mode=duration_mode.lower(), elapsed_seconds=elapsed, off=off_details, on=on_details,
        note=("Same caption, lyrics and seed. YuE2 chooses when each song ends."
              if duration is None else
              "Same caption, lyrics and seed. The duration limit can cut a phrase short."))


def duration_controls(mode: str) -> dict:
    """Show the optional duration limit only when Limited is selected."""
    return gr.update(visible=mode == "Limited")


def example_for(concept: str) -> tuple[str, str, str]:
    """Return the held-out neutral caption and original lyric sheet for a control."""
    spec = CATALOG[concept]
    sample = spec["samples"][0]
    return spec["description"], sample["caption"], sample["lyrics"]


def listening_pair(concept: str, comparison: str) -> tuple[str, str, str]:
    """Return pre-rendered Off and On examples and their matched seed and prompt."""
    index = int(comparison)
    if concept not in CATALOG or index not in range(4):
        raise gr.Error("Choose a control and comparison.")
    spec, sample = CATALOG[concept], CATALOG[concept]["samples"][index]
    from shutil import copyfile
    output = Path(tempfile.mkdtemp(prefix="yue2-listening-"))
    paths = []
    for side in ("off", "on"):
        cached = hf_hub_download(REPO, sample[side], revision=MODEL_REVISION)
        path = output / f"{side}.mp3"
        copyfile(cached, path)
        paths.append(str(path))
    note = (f"**{spec['label']} · step 1200 · held-out row {sample['row'] + 1} · seed {sample['seed']}**\n\n"
            "Off 0 / On +1. Same caption and lyrics. Approximately 20-second excerpts.\n\n"
            f"**Caption**\n\n{sample['caption']}\n\n**Lyrics**\n\n{sample['lyrics']}")
    return paths[0], paths[1], note


choices = [(s["label"], sid) for sid, s in CATALOG.items()]
default = "female"
blurb, caption_default, lyrics_default = example_for(default)
CSS = """
.gradio-container {max-width: 1140px !important; margin: auto;}
#hero {padding: 24px 0 14px;}
#hero h1 {font-size: clamp(2rem, 5vw, 3.4rem); letter-spacing: -.04em;}
#generate {min-height: 52px; font-size: 1.1rem;}
footer {display:none !important;}
"""
with gr.Blocks(title="YuE2 Concept Sliders") as demo:
    gr.Markdown("""# YuE2 · Concept Sliders
**One prompt. Sixteen ways to move the sound.**

Shape the voice or arrangement with a slider. Compare the same caption, lyrics and seed with the control off and on.

[Weights & listening examples](https://huggingface.co/ntc-ai/yue2-concept-sliders) · [MiniMax Music 3 demo](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders)
""", elem_id="hero")
    with gr.Tab("Create a comparison"):
        with gr.Row():
            with gr.Column(scale=1):
                concept = gr.Dropdown(choices, value=default, label="Control")
                description = gr.Markdown(blurb)
                strength = gr.Slider(0, 1, value=1, step=.05, label="Strength · 0 off / 1 on")
                caption = gr.Textbox(value=caption_default, label="Sound description", lines=9,
                    info="Describe instruments, voice, room and timing. Leave out real music names.")
                lyrics = gr.Textbox(value=lyrics_default, label="Lyrics", lines=10)
            with gr.Column(scale=1):
                with gr.Row():
                    off = gr.Audio(label="Off · base YuE2", type="filepath")
                    on = gr.Audio(label="On · your control", type="filepath")
                compare = gr.Checkbox(True, label="Generate both Off and On with the same seed")
                duration_mode = gr.Radio(["Automatic", "Limited"], value="Automatic",
                    label="Song length", info="Automatic lets YuE2 choose when the song ends.")
                seconds = gr.Slider(2, 360, value=30, step=1,
                    label="Duration limit (seconds)", visible=False)
                with gr.Accordion("Generation settings", open=False):
                    steps = gr.Slider(4, 32, value=16, step=4, label="Audio rendering steps")
                    seed = gr.Number(1709, precision=0, label="Seed", minimum=0, maximum=MAX_SEED)
                    randomize_seed = gr.Checkbox(False, label="Randomize seed")
                button = gr.Button("Generate comparison", variant="primary", elem_id="generate")
                gr.Markdown("Off and On may have different lengths. Changing the slider can change the composition and phrasing.")
                with gr.Accordion("Generation details", open=False):
                    details = gr.JSON()
        concept.change(example_for, concept, [description, caption, lyrics], api_name="example_for")
        duration_mode.change(duration_controls, duration_mode, seconds, api_name=False)
        inputs = [concept, strength, caption, lyrics, seconds, steps, seed, randomize_seed, compare, duration_mode]
        outputs = [off, on, seed, details]
        button.click(generate, inputs, outputs, api_name="generate", concurrency_limit=1, concurrency_id="gpu")
        gr.Examples([[sid, 1, CATALOG[sid]["samples"][0]["caption"], CATALOG[sid]["samples"][0]["lyrics"]]
                     for sid in ("female", "metal", "house", "lofi")],
                    inputs=inputs[:4], outputs=outputs, fn=generate,
                    cache_examples=True, cache_mode="lazy", label="Try a held-out prompt")
    with gr.Tab("Listen to the release"):
        gr.Markdown("Browse all 64 matched Off/On pairs. These recordings play without using GPU quota.")
        with gr.Row():
            listen_concept = gr.Dropdown(choices, value=default, label="Control")
            listen_index = gr.Dropdown([("Prompt 1 · seed 1709", "0"), ("Prompt 1 · seed 2903", "1"),
                                       ("Prompt 2 · seed 1709", "2"), ("Prompt 2 · seed 2903", "3")], value="0", label="Comparison")
            listen_button = gr.Button("Load comparison")
        with gr.Row():
            sample_off = gr.Audio(label="Off · 0", type="filepath")
            sample_on = gr.Audio(label="On · +1", type="filepath")
        sample_notes = gr.Markdown()
        listen_button.click(listening_pair, [listen_concept, listen_index],
                            [sample_off, sample_on, sample_notes], api_name="listening_pair")
    gr.Markdown("16 experimental controls · Final EMA checkpoints at 1,200 updates · [Native loading guide](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/USAGE.md)")

demo.queue(default_concurrency_limit=1, max_size=12).launch(
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue"), css=CSS, mcp_server=True)
