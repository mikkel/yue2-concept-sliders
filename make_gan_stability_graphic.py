#!/usr/bin/env python3
"""Plot existing, unsmoothed YuE2 logs; never rerun or alter training."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ROOT / 'sliders-conceptmod/models'
OUT = HERE / 'assets'
DATA = HERE / 'evidence/gan-stability'
RUNS = [
    ('plain-original', 'metal-yue2-uni-control-600-s7-20260916',
     'Without particles · original rates', '#c43b32'),
    ('plain-slower', 'metal-yue2-uni-native-lr-600-s7-20260916',
     'Without particles · rates ÷ 5', '#087f72'),
    ('particle', 'metal-yue2-particle-bridge-s7-20260917',
     'With particles · new recipe', '#7950b4'),
]
DISPLAY_RUN_IDS = ('plain-original', 'particle')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    sys.path.insert(0, str(ROOT))
    from app.rewriter import _artist_name_hit
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    all_rows, records, common = [], [], None
    for run_id, name, label, color in RUNS:
        directory = MODELS / name
        manifest = json.loads((directory / 'manifest.json').read_text())
        settings = manifest['settings']
        prompt_hash = hashlib.sha256(json.dumps(settings['rows'], sort_keys=True).encode()).hexdigest()
        identity = dict(model=manifest['model'], seed=settings['seed'], prompts_sha256=prompt_hash,
                        rank=settings['rank'], alpha=settings['alpha'], model_id=settings['model_id'])
        if common is None:
            common = identity
        assert identity == common and not settings['dummy']
        for row in settings['rows']:
            assert _artist_name_hit('', row['neutral'] + '\n' + row['positive'], row['lyrics']) is None
        by_step, inputs = {}, []
        for path in sorted(directory.glob('updates-from-*.jsonl')):
            inputs.append(dict(name=path.name, sha256=digest(path), bytes=path.stat().st_size))
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if 'step' not in row:
                    continue
                if row['step'] in by_step:
                    assert row == by_step[row['step']], (name, row['step'])
                by_step[row['step']] = row
        steps = sorted(by_step)
        assert steps == list(range(1, max(steps) + 1))
        rows = [by_step[step] for step in steps]
        assert all(row['g_adv'] > 0 for row in rows)
        peak = max(rows, key=lambda row: row['g_adv'])
        record = dict(id=run_id, run=name, label=label, updates=len(rows), seed=settings['seed'],
                      recipe=settings['recipe'], model=manifest['model'], prompts_sha256=prompt_hash,
                      manifest_sha256=digest(directory / 'manifest.json'),
                      source_files=settings['sources'], log_files=inputs,
                      peak_g_adv=peak['g_adv'], peak_g_step=peak['step'],
                      final_g_adv=rows[-1]['g_adv'], final_cos_pos=rows[-1]['cos_pos'])
        records.append(record)
        for row in rows:
            all_rows.append(dict(run=run_id, step=row['step'], g_adv=row['g_adv'],
                                 cos_pos=row['cos_pos'], grad_norm=row['grad_norm'],
                                 g_lr=row['g_lr'], d_lr=row['d_lr']))
    particle_checkpoint = MODELS / RUNS[2][1] / (RUNS[2][1] + '_step1200.safetensors')
    catalog = json.loads((HERE / 'model/catalog.json').read_text())
    released_metal = next(row for row in catalog['sliders'] if row['id'] == 'metal')
    assert digest(particle_checkpoint) == released_metal['sha256']
    with (DATA / 'training-curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(all_rows)
    evidence = dict(description='Selected historical native YuE2 comparison; not a particle-only ablation',
                    displayed_runs=list(DISPLAY_RUN_IDS),
                    common=common, metrics=dict(g_adv='Logged generator adversarial term; excludes particle VIC',
                    cos_pos='Logged cosine to the positive-caption hidden target, not an audio metric',
                    grad_norm='Logged generator parameter gradient norm; different architectures have different parameters'),
                    smoothing='none', downsampling='none', missing_updates=0,
                    csv_sha256=digest(DATA / 'training-curves.csv'),
                    particle_checkpoint_sha256=digest(particle_checkpoint),
                    particle_checkpoint_is_released_metal=True,
                    generator_script_sha256=digest(Path(__file__)), runs=records)
    (DATA / 'provenance.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return all_rows, records


def render(all_rows, records):
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13.5,
                         'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                         'ytick.labelsize': 13, 'svg.fonttype': 'none', 'svg.hashsalt': 'yue2-gan-stability'})
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 9.6), sharex=True,
                             gridspec_kw={'height_ratios': [1.3, 1]})
    fig.subplots_adjust(left=.15, right=.96, top=.79, bottom=.175, hspace=.24)
    fig.suptitle('YuE2 generator training', x=.15, y=.97, ha='left', fontsize=22, fontweight='bold')
    fig.text(.15, .927, 'Metal · seed 7 · unsmoothed training logs', fontsize=13.5, color='#45566b')
    for run_id, name, label, color in RUNS:
        if run_id not in DISPLAY_RUN_IDS:
            continue
        rows = [row for row in all_rows if row['run'] == run_id]
        for ax, key in zip(axes, ('g_adv', 'cos_pos')):
            ax.plot([r['step'] for r in rows], [r[key] for r in rows], label=label,
                    color=color, linewidth=1.25, alpha=.95)
    axes[0].set_yscale('log')
    axes[0].set_ylim(.5, 1400)
    axes[0].yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    axes[0].set_ylabel('G adversarial loss · log scale')
    axes[1].set_ylabel('Hidden-target cosine')
    axes[1].set_ylim(-.16, 1.07)
    axes[1].set_yticks([0, .5, 1.])
    axes[1].set_xlabel('Training update')
    axes[1].set_xticks([0, 300, 600, 900, 1200])
    axes[1].set_xlim(0, 1200)
    for ax in axes:
        ax.grid(axis='y', which='major', color='#dde4eb', linewidth=.7)
        ax.axvline(600, color='#94a3b8', linestyle='--', linewidth=.9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#94a3b8')
        ax.set_axisbelow(True)
    peak = records[0]
    axes[0].scatter([peak['peak_g_step']], [peak['peak_g_adv']], s=32, color=RUNS[0][3], zorder=5)
    axes[0].annotate(f"G = {peak['peak_g_adv']:.1f}\nat update {peak['peak_g_step']}",
                     xy=(peak['peak_g_step'], peak['peak_g_adv']), xytext=(680, 260),
                     fontsize=13, fontweight='bold', color=RUNS[0][3],
                     arrowprops={'arrowstyle': '->', 'color': RUNS[0][3], 'lw': 1.1})
    axes[1].text(625, .09, 'Particle run continues\nto update 1,200', fontsize=12, color='#6835a2')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.13, .897), frameon=False,
               fontsize=13.5, handlelength=2.5, labelspacing=.65)
    fig.text(.15, .041, 'Selected recipes: architecture, loss and rates differ.\n'
             'This is not a matched particle-only ablation.', fontsize=12, color='#45566b')
    metadata = {'Title': 'YuE2 generator training: historical plain and particle recipes',
                'Description': 'All logged updates for the two displayed runs, with no smoothing. '
                               'The original plain LoRA spikes at step 373; the particle run continues to 1200. '
                               'This is not a matched particle-only ablation.',
                'Date': None}
    fig.savefig(OUT / 'gan-training-stability.svg', metadata=metadata, facecolor='white')
    fig.savefig(OUT / 'gan-training-stability.png', dpi=180, facecolor='white')
    plt.close(fig)
    for record in records:
        print(json.dumps({key: record[key] for key in
                          ('id', 'updates', 'peak_g_adv', 'peak_g_step', 'final_cos_pos')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-csv', action='store_true',
                        help='Replot the published CSV and provenance without the original training workspace')
    args = parser.parse_args()
    if args.from_csv:
        with (DATA / 'training-curves.csv').open() as stream:
            all_rows = [{key: (value if key == 'run' else int(value) if key == 'step' else float(value))
                         for key, value in row.items()} for row in csv.DictReader(stream)]
        records = json.loads((DATA / 'provenance.json').read_text())['runs']
    else:
        all_rows, records = collect()
    render(all_rows, records)
