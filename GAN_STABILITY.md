### Why we added particles: GAN training stability

**We introduced the particle recipe to help stabilize the generator’s adversarial training.** Earlier native YuE2 GAN runs showed large generator-loss spikes and a collapse in alignment with the target hidden state. Here, **G** is the trainable slider attached to frozen YuE2; **D** is the training critic.

The learned cloud gives G additional parameters it can move while learning the correction. The variance/covariance regularizer encourages the cloud to stay spread out. The full recipe also caps overly steep critic gradients and gradually reduces noise in the paired-error game. These are intended to make optimization more manageable; their individual contributions need separate controls.

![Unsmoothed native YuE2 training curves. The original plain-LoRA recipe has a generator-loss spike of 549.8 at update 373 and loses target alignment. The particle recipe continues to 1,200 updates with much smaller spikes. A plain-LoRA run with both learning rates divided by five also avoids the large spike.](assets/gan-training-stability.svg)

[Full-size SVG](assets/gan-training-stability.svg) · [PNG](assets/gan-training-stability.png) · [Every plotted value](evidence/gan-stability/training-curves.csv) · [Recipes, source hashes and checkpoint provenance](evidence/gan-stability/provenance.json)

These are existing **Metal, seed 7** runs with the same frozen base, four prompt/lyric pairs and rank-8 attention targets. Every logged update is shown, with no smoothing. The top panel is the logged **G adversarial term** (`g_adv`), excluding particle regularization; its vertical axis is logarithmic. The lower panel shows hidden-target cosine, an internal diagnostic rather than an audio-quality score.

| Historical run | Recorded updates | Peak G adversarial loss | Update at peak |
|---|---:|---:|---:|
| Plain LoRA, original rates | 600 | **549.76** | 373 |
| Plain LoRA, G and D rates divided by 5 | 600 | 3.90 | 5 |
| Routed-particle recipe | 1,200 | 35.66 | 945 |

The particle trace ends at the **exact released Metal checkpoint**. It still has smaller spikes. The slower plain-LoRA control also avoids the large spike, so the evidence does not establish that particles alone caused the improvement. The particle recipe additionally changes the critic, error normalization, noise, batching, optimizer settings and regularization; the loss magnitudes cannot rank these recipes independently of those changes. The dashed line marks the end of the two 600-update runs, not a continuation of their measurements.

The plain-LoRA curves above are earlier **GAN-training experiments**. The downloadable **distilled LoRAs** were subsequently trained to imitate the particle teachers using regression and hidden-state matching. This graph does not describe their training or demonstrate a listening-quality advantage.

See [ParticleGAN](https://github.com/255BITS/ParticleGAN) for the learned-particle and adversarial-training building blocks. In this release, the particles remain part of the learned forward path at inference; distillation is how we approximate that path with ordinary LoRA matrices.


## Recreate the figure

Download this repository’s `make_gan_stability_graphic.py` and `evidence/gan-stability/` folder. With Matplotlib installed, run:

```bash
python make_gan_stability_graphic.py --from-csv
```

This reads the published measurements and redraws the SVG and PNG; it performs no training.
