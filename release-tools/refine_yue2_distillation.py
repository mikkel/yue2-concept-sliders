#!/usr/bin/env python3
"""Refine distilled LoRAs on teacher hidden states and render held-out audio."""
from __future__ import annotations
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import time

import distill_yue2_particles as d
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_hidden(model, ids):
    tokens = torch.tensor([ids], device=next(model.parameters()).device)
    positions = torch.arange(len(ids), device=tokens.device)[None]
    backbone = model.model
    x = backbone.embed_tokens(tokens)
    cos, sin = backbone.rotary_emb(positions)
    for layer in backbone.layers:
        x = checkpoint(layer, x, cos, sin, use_reentrant=False)
    return backbone.norm(x).float()


def score(metrics):
    return float(np.mean([m['relative_steering_mse'] for m in metrics]))


def refine(args, pipe, model, spec, all_rows):
    concept = spec['id']
    out = args.output / concept
    target = out / f'{concept}_distilled_refined_rank8.safetensors'
    if (out/'refined-evaluation.json').exists():
        return target
    source = args.release/spec['weights']
    assert d.file_digest(source) == spec['sha256']
    teacher, metadata = d.YuE2Slider.load(model, source)
    d.detach(model, teacher)
    teacher.requires_grad_(False)
    student, record = d.YuE2Slider.load(model, out/f'{concept}_distilled_rank8.safetensors')
    d.detach(model, student)
    train = d.rows(args.release, spec, 'train')
    fitting_rows = [row for row in all_rows if row['lyrics'] != train[-1]['lyrics']]
    sequences = []
    for index, row in enumerate(train):
        saved = json.loads((out/f'train-rollout-{index}.json').read_text())
        assert saved['prefix'] == d.make_plan(pipe, row, 4100+index).prefix
        sequences.append((f'train-{index}', saved['ids'], len(saved['prefix'])))
    with torch.no_grad():
        baseline = d.compare(model, teacher, student, [sequences[-1]])
    best_score, best_step = score(baseline), 0
    best = {k: v.detach().cpu().clone() for k,v in student.state_dict().items()}
    history = [dict(step=0, score=best_score, metrics=baseline)]
    examples = sequences[:-1] + [(f'prefix-{i}', d.make_plan(pipe, row, 7).prefix,
                                 len(d.make_plan(pipe, row, 7).prefix)) for i,row in enumerate(fitting_rows)]
    cached = {}
    with torch.no_grad():
        for label, ids, prefix in examples:
            base = d.hidden(model, ids)
            for scale in (.5, 1.):
                with d.attached(model, teacher, scale):
                    wanted = d.hidden(model, ids)
                cached[(label, scale)] = (wanted.cpu(), base.cpu())
    student.requires_grad_(True)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    rng = random.Random(9918)
    d.log('refine_start', concept=concept, gpu=args.gpu, baseline=best_score)
    for step in range(1, args.steps+1):
        # Half the updates include actual semantic continuations; half broaden
        # the captions. The fourth lyric sheet is reserved for model selection.
        pool = examples[:3] if step % 2 else examples[3:]
        label, ids, prefix = rng.choice(pool)
        scale = rng.choice((.5, 1.))
        wanted, base = (t.to(next(model.parameters()).device) for t in cached[(label, scale)])
        optimizer.zero_grad(set_to_none=True)
        with d.attached(model, student, scale):
            prediction = checkpoint_hidden(model, ids)
            loss = torch.zeros((), device=prediction.device)
            for positions in (slice(prefix-1, prefix), slice(prefix-1, None)):
                energy = (wanted[:, positions]-base[:, positions]).square().mean().clamp_min(1e-7)
                loss = loss + (prediction[:, positions]-wanted[:, positions]).square().mean() / energy / 2
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite distillation loss')
            # Keep the adapters installed until checkpoint recomputation ends.
            loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                metrics = d.compare(model, teacher, student, [sequences[-1]])
            value = score(metrics)
            history.append(dict(step=step, score=value, metrics=metrics, train_loss=float(loss)))
            if value < best_score:
                best_score, best_step = value, step
                best = {k: v.detach().cpu().clone() for k,v in student.state_dict().items()}
            d.log('refine_validation', concept=concept, step=step, score=value, best_step=best_step)
    student.load_state_dict(best)
    student.requires_grad_(False)
    record.update(recipe='particle-to-linear-ridge-and-hidden-distillation-v1',
        refinement_steps=args.steps, selected_step=best_step, learning_rate=1e-4,
        selection='lowest hidden steering MSE on reserved training lyric sheet',
        refinement_script_sha256=d.file_digest(__file__))
    student.save(target, record)
    heldout = []
    for row in range(2):
        heldout.extend(d.existing_eval(args.reference, concept, row, 1709))
    with torch.no_grad():
        metrics = d.compare(model, teacher, student, heldout, logits=True)
    d.write_json(out/'refined-evaluation.json', dict(concept=concept,
        checkpoint=target.name, checkpoint_sha256=d.file_digest(target),
        history=history, selected_step=best_step, heldout=metrics,
        zero_exact=True, ordinary_lora_only=True))
    d.log('refine_complete', concept=concept, selected_step=best_step,
        heldout_music_mse=float(np.mean([m['relative_steering_mse'] for m in metrics if m['region']=='music'])))
    del optimizer, teacher, student, best, cached, loss, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return target


@torch.inference_mode()
def audio(args, pipe, model, spec, target):
    import soundfile as sf
    from yue2.modeling_vae import YuE2VAE
    concept = spec['id']
    out = args.output/concept/'audio'
    out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():
        return
    teacher, _ = d.YuE2Slider.load(model, args.release/spec['weights'])
    d.detach(model, teacher)
    student, _ = d.YuE2Slider.load(model, target)
    d.detach(model, student)
    teacher.requires_grad_(False)
    student.requires_grad_(False)
    if pipe._vae is None:
        pipe._vae = YuE2VAE.from_pretrained(pipe.vae_dir, decoder_only=True,
                                          device=pipe.device, local_files_only=True)
    pipe.generation_config = replace(pipe.generation_config, ode_steps=16)
    records = []
    for index, row in enumerate(d.rows(args.release, spec, 'eval')):
        seed = 1709
        plan = d.make_plan(pipe, row, seed)
        student_semantic = None
        for kind, network, acoustic in [('off', None, False), ('teacher', teacher, False),
                                         ('plain-ar', student, False), ('plain-all', student, True)]:
            start = time.monotonic()
            if kind == 'plain-all':
                semantic = student_semantic
            else:
                with d.attached(model, network, 1.):
                    semantic = pipe.generate_semantic(plan, sampling={'max_tokens': 500, 'min_tokens': 200})
                if kind == 'plain-ar':
                    student_semantic = semantic
            with d.attached(model, network if acoustic else None, 1.):
                latents = pipe.synthesize(semantic)
            z = torch.as_tensor(latents, dtype=torch.float32).T.unsqueeze(0)
            decoded = pipe._vae.decode_tiled(z, core_frames=pipe.vae_core_frames,
                                           halo_frames=16, output_device='cpu')
            if not torch.isfinite(decoded).all():
                raise RuntimeError('Non-finite audio')
            signal = decoded[0].float().T.contiguous().numpy()
            rms = float(np.sqrt(np.mean(signal.astype(np.float64)**2)))
            if rms < 1e-6:
                raise RuntimeError('Silent audio')
            filename = f'row{index}-{kind}.flac'
            sf.write(out/filename, np.clip(signal, -1, 1), 48000, subtype='PCM_16')
            np.save(out/f'row{index}-{kind}-semantic.npy', semantic.tokens)
            entry = dict(row=index, kind=kind, file=filename, seed=seed,
                seconds=len(signal)/48000, rms=rms, peak=float(np.max(np.abs(signal))),
                truncated=semantic.truncated, semantic_tokens=len(semantic.tokens),
                acoustic_prefix_adapted=acoustic, elapsed_seconds=time.monotonic()-start)
            d.write_json(out/f'row{index}-{kind}.json', dict(entry, caption=row['neutral'], lyrics=row['lyrics']))
            records.append(entry)
            d.log('audio_complete', concept=concept, **entry)
    d.write_json(out/'summary.json', dict(concept=concept, recordings=records,
        checkpoint_sha256=d.file_digest(target), teacher_sha256=spec['sha256'],
        protocol='Same caption, lyrics and seed; 20-second diagnostic limit, 16 ODE steps',
        plain_all_note='Native LoRA also applied to acoustic-prefix AR projections, mirroring standard ComfyUI scope; not a ComfyUI audio render'))
    del teacher, student
    gc.collect()
    torch.cuda.empty_cache()

