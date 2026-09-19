#!/usr/bin/env python3
"""Calibrate ordinary rank-8 LoRAs against frozen YuE2 particle teachers.

This is an approximate, data-dependent distillation, not an algebraic merge.
The teacher's up matrices are retained; its nonlinear rank-8 features are
regressed on real projection inputs with a homogeneous ridge regression.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from training_runtime import YuE2Slider, attention_targets, file_digest
from training_runtime import _artist_name_hit


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def log(event, **values):
    print(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          event=event, **values)), flush=True)


def detach(model, network):
    for name, module in attention_targets(model).items():
        module.forward = network.adapters[name.replace('.', '-')].org_forward


@contextmanager
def attached(model, network, scale):
    targets = attention_targets(model)
    originals = {name: module.forward for name, module in targets.items()}
    try:
        if network is not None:
            for name, module in targets.items():
                module.forward = network.adapters[name.replace('.', '-')].forward
            with network.scaled(scale):
                yield
        else:
            yield
    finally:
        for name, module in targets.items():
            module.forward = originals[name]


def hidden(model, ids):
    tokens = torch.tensor([ids], device=next(model.parameters()).device)
    positions = torch.arange(len(ids), device=tokens.device)[None]
    return model.model(input_ids=tokens, position_ids=positions, use_cache=False)[0].float()


def rows(release, spec, split):
    result = yaml.safe_load((release / spec['prompts'][split]).read_text())['rows']
    for row in result:
        if _artist_name_hit('', row['neutral'] + '\n' + row['positive'], row['lyrics']):
            raise ValueError('Named music reference in source prompts; do not distill these weights')
    return result


def make_plan(pipe, row, seed):
    from yue2.protocol import SongRequest
    return pipe.plan(request=SongRequest(style=row['neutral'], lyrics=row['lyrics'], cot='off', seed=seed))


def rollout(pipe, model, network, row, seed, count, cache):
    from yue2.protocol import CODEC_OFFSET
    plan = make_plan(pipe, row, seed)
    if cache.exists():
        saved = json.loads(cache.read_text())
        assert saved['prefix'] == plan.prefix and saved['seed'] == seed
        return saved['ids']
    with attached(model, network, 1):
        semantic = pipe.generate_semantic(plan, sampling={'max_tokens': count, 'min_tokens': min(200, count)})
    ids = plan.prefix + [int(t) + CODEC_OFFSET for t in semantic.tokens]
    write_json(cache, dict(prefix=plan.prefix, ids=ids, seed=seed, max_tokens=count,
                          truncated=semantic.truncated, timing=semantic.timing))
    return ids


class Calibration:
    """One covariance per QKV input and per O input; no activation dump."""
    def __init__(self, model, teacher):
        self.groups = {}
        self.handles = []
        targets = attention_targets(model)
        for name, module in targets.items():
            projection = name.rsplit('.', 1)[1]
            if projection not in ('q_proj', 'o_proj'):
                continue
            names = ([name.rsplit('.', 1)[0] + '.' + p for p in ('q_proj', 'k_proj', 'v_proj')]
                     if projection == 'q_proj' else [name])
            adapters = [teacher.adapters[n.replace('.', '-')] for n in names]
            dim = module.in_features
            device = module.weight.device
            group = dict(names=names, xx=torch.zeros(dim, dim, device=device),
                         xy=torch.zeros(dim, 8 * len(names), device=device), count=0)
            self.groups[name] = group

            def collect(_module, inputs, group=group, adapters=adapters):
                x = inputs[0].reshape(-1, dim).float()
                y = torch.cat([a.bridge(a.lora_down(x), teacher.particles) for a in adapters], dim=-1)
                group['xx'].addmm_(x.T, x)
                group['xy'].addmm_(x.T, y)
                group['count'] += x.shape[0]

            self.handles.append(module.register_forward_pre_hook(collect))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def solve(self, student, teacher, ridge):
        for group in self.groups.values():
            xx, xy = group['xx'], group['xy']
            norm = xx.diag().clamp_min(1e-12).sqrt()
            correlation = xx / norm[:, None] / norm[None, :]
            correlation.diagonal().add_(ridge)
            chol = torch.linalg.cholesky(correlation)
            beta = torch.cholesky_solve(xy / norm[:, None], chol) / norm[:, None]
            for index, name in enumerate(group['names']):
                key = name.replace('.', '-')
                student.adapters[key].lora_down.weight.copy_(beta[:, index*8:(index+1)*8].T)
                student.adapters[key].lora_up.weight.copy_(teacher.adapters[key].lora_up.weight)


def compare(model, teacher, student, sequences, *, logits=False):
    records = []
    for label, ids, prefix_len in sequences:
        base = hidden(model, ids)
        with attached(model, student, 0):
            if not torch.equal(base, hidden(model, ids)):
                raise AssertionError('Scale zero changed the base model')
        for scale in (.5, 1.):
            with attached(model, teacher, scale):
                target = hidden(model, ids)
            with attached(model, student, scale):
                prediction = hidden(model, ids)
            regions = {'boundary': slice(prefix_len-1, prefix_len),
                       'music': slice(prefix_len-1, None)}
            for region, positions in regions.items():
                shift = target[:, positions] - base[:, positions]
                estimated = prediction[:, positions] - base[:, positions]
                error = (estimated - shift).square().sum()
                energy = shift.square().sum().clamp_min(1e-12)
                metric = dict(sequence=label, scale=scale, region=region,
                    relative_steering_mse=float(error / energy),
                    steering_cosine=float(F.cosine_similarity(estimated.flatten(), shift.flatten(), dim=0)),
                    steering_norm_ratio=float(estimated.norm() / shift.norm().clamp_min(1e-12)))
                if logits and region == 'music':
                    # Native semantic distribution, before sampling penalties/CFG.
                    from yue2.protocol import CODEC_OFFSET
                    positions = torch.arange(prefix_len-1, len(ids), 32, device=base.device)
                    weight = model.lm_head.weight[CODEC_OFFSET:CODEC_OFFSET+32768].float()
                    tl = F.log_softmax(target[0, positions] @ weight.T, dim=-1)
                    sl = F.log_softmax(prediction[0, positions] @ weight.T, dim=-1)
                    bl = F.log_softmax(base[0, positions] @ weight.T, dim=-1)
                    metric['teacher_student_kl'] = float((tl.exp() * (tl-sl)).sum(-1).mean())
                    metric['teacher_base_kl'] = float((tl.exp() * (tl-bl)).sum(-1).mean())
                records.append(metric)
    return records


def existing_eval(root, concept, row, seed):
    from yue2.protocol import CODEC_OFFSET
    result = []
    for kind in ('off', 'metal'):
        path = root / concept / f'row-{row}-seed-{seed}' / kind
        prefix = np.load(path / 'prefix.npy').reshape(-1).tolist()
        tokens = np.load(path / 'semantic.npy').reshape(-1).tolist()
        result.append((f'row{row}-{kind}-seed{seed}', prefix + [int(t)+CODEC_OFFSET for t in tokens], len(prefix)))
    return result


def fit_one(args, pipe, model, release, spec, calibration_rows):
    concept = spec['id']
    out = args.output / concept
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f'{concept}_distilled_rank8.safetensors'
    if (out / 'evaluation.json').exists():
        log('skip_completed', concept=concept)
        return
    source = release / spec['weights']
    assert file_digest(source) == spec['sha256']
    teacher, metadata = YuE2Slider.load(model, source)
    detach(model, teacher)
    teacher.requires_grad_(False)
    assert metadata['model_identity'] == pipe.weights['mot']
    train, test = rows(release, spec, 'train'), rows(release, spec, 'eval')
    assert not {r['lyrics'] for r in train} & {r['lyrics'] for r in test}
    validation_lyrics = train[-1]['lyrics']
    fitting_rows = [r for r in calibration_rows if r['lyrics'] != validation_lyrics]
    log('rollouts', concept=concept, gpu=args.gpu, prefix_rows=len(fitting_rows))
    sequences = []
    for index, row in enumerate(train):
        ids = rollout(pipe, model, teacher, row, 4100+index, args.train_tokens, out/f'train-rollout-{index}.json')
        sequences.append((f'train-{index}', ids, len(make_plan(pipe, row, 4100+index).prefix)))
        log('rollout_complete', concept=concept, row=index, tokens=len(ids))
    student = YuE2Slider(model, rank=8, alpha=8.)
    detach(model, student)
    student.requires_grad_(False)
    stats = Calibration(model, teacher)
    try:
        for scale in (0., 1.):
            with attached(model, teacher, scale):
                for index, row in enumerate(fitting_rows):
                    hidden(model, make_plan(pipe, row, 7).prefix)
                for _, ids, _ in sequences[:-1]:
                    hidden(model, ids)
            log('calibration_pass', concept=concept, scale=scale,
                tokens=next(iter(stats.groups.values()))['count'])
    finally:
        stats.close()
    scores = []
    for ridge in (1e-4, 1e-3, 1e-2):
        stats.solve(student, teacher, ridge)
        metrics = compare(model, teacher, student, [sequences[-1]])
        score = sum(m['relative_steering_mse'] for m in metrics) / len(metrics)
        scores.append(dict(ridge=ridge, score=score, metrics=metrics))
        log('validation', concept=concept, ridge=ridge, relative_steering_mse=score)
    selected = min(scores, key=lambda s: s['score'])
    stats.solve(student, teacher, selected['ridge'])
    token_count = next(iter(stats.groups.values()))['count']
    del stats
    gc.collect()
    torch.cuda.empty_cache()
    record = dict(model_id=metadata['model_id'], model_identity=metadata['model_identity'],
        concept=concept, recipe='particle-to-linear-ridge-distillation-v1',
        teacher_sha256=spec['sha256'], teacher_format=metadata['format'],
        ridge=selected['ridge'], calibration_tokens=token_count,
        calibration_prompt_rows=len(fitting_rows), calibration_scales=[0, 1],
        train_rollout_tokens=args.train_tokens, validation_train_row=3,
        evaluation_lyrics_excluded=True, seed=4100, gpu=args.gpu,
        rank=8, alpha=8., ordinary_lora=True, approximate=True,
        inference_stages=['semantic'], acoustic_scale=0,
        method='Fix teacher U; solve min_A ||X A^T - routed_teacher(X)||^2 + normalized ridge',
        source_script_sha256=file_digest(__file__))
    student.save(checkpoint, record)
    # Validate the actual exported loader and all tensors before evaluating.
    exported, exported_record = YuE2Slider.load(model, checkpoint)
    detach(model, exported)
    assert exported_record['format'] == 'conceptmod-yue2-ar-v1'
    assert all('bridge' not in k and 'particles' not in k for k in exported.state_dict())
    assert all(torch.equal(v, exported.state_dict()[k]) for k, v in student.state_dict().items())
    del student
    heldout = []
    for row in range(2):
        found = existing_eval(args.reference, concept, row, 1709)
        expected = make_plan(pipe, test[row], 1709).prefix
        assert all(ids[:n] == expected for _, ids, n in found)
        heldout.extend(found)
    metrics = compare(model, teacher, exported, heldout, logits=True)
    write_json(out / 'evaluation.json', dict(concept=concept, checkpoint=checkpoint.name,
        checkpoint_sha256=file_digest(checkpoint), teacher_sha256=spec['sha256'],
        validation=scores, selected_ridge=selected['ridge'], heldout=metrics,
        zero_exact=True, roundtrip_exact=True, ordinary_lora_only=True))
    log('fit_complete', concept=concept, checkpoint=str(checkpoint),
        heldout_music_mse=float(np.mean([m['relative_steering_mse'] for m in metrics if m['region']=='music'])))
    del teacher, exported
    gc.collect()
    torch.cuda.empty_cache()

