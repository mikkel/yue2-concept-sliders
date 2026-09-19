#!/usr/bin/env python3
"""Create v2 ordinary LoRAs, comparisons and ComfyUI exports from a Hub snapshot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from yue2 import YuE2Pipeline
import distill_yue2_particles as d
import refine_yue2_distillation as r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--converter', type=Path, required=True, help='conceptmod checkout at a5c3dd8 or later')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA-visible device index')
    parser.add_argument('--ids', help='Comma-separated controls; defaults to all 16')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--allow-hub', action='store_true', help='Allow downloads of the pinned base model and VAE')
    args = parser.parse_args()
    args.release = args.snapshot.resolve()
    args.reference = args.release / 'evidence/particle-gmix-1600-v2/reference'
    args.output = args.output.resolve()
    args.converter = args.converter.resolve()
    args.steps, args.train_tokens = 100, 384
    catalog = json.loads((args.release / 'particle-gmix-1600-v2/catalog.json').read_text())
    assert catalog['release'] == 'particle-gmix-1600-v2'
    specs = {s['id']: s for s in catalog['sliders']}
    selected = args.ids.split(',') if args.ids else list(specs)
    if set(selected) - specs.keys(): parser.error('Unknown control')
    all_rows = {}
    for spec in specs.values():
        for row in d.rows(args.release, spec, 'train'):
            all_rows[(row['neutral'], row['lyrics'])] = row
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(9182)
    torch.cuda.set_device(args.gpu)
    with YuE2Pipeline.from_pretrained(catalog['base_model'], revision=catalog['base_revision'],
        vae=catalog['vae'], vae_revision=catalog['vae_revision'],
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        local_files_only=not args.allow_hub, device=f'cuda:{args.gpu}', backend='torch',
        quantization='none', offload_ar=False, memory_budget_gib=26, progress=False) as pipe:
        model = pipe._load_model().requires_grad_(False)
        for concept in selected:
            spec = specs[concept]
            # Never reuse a completed student from another teacher.
            existing = args.output / concept / 'evaluation.json'
            if existing.exists():
                assert json.loads(existing.read_text())['teacher_sha256'] == spec['sha256']
            with torch.inference_mode():
                d.fit_one(args, pipe, model, args.release, spec, list(all_rows.values()))
            target = r.refine(args, pipe, model, spec, list(all_rows.values()))
            r.audio(args, pipe, model, spec, target)
            subprocess.run([sys.executable, 'scripts/convert_lora_comfyui.py', str(target)],
                cwd=args.converter, env=dict(os.environ, PYTHONPATH=str(args.converter)), check=True)
    print('Completed:', ', '.join(selected))


if __name__ == '__main__':
    main()
