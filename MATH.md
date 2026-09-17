## How the sliders learn

Each control learns from four pairs of sound descriptions. The frozen YuE2 model supplies a target hidden state from the positive description; the adapted model receives the neutral description and the same lyrics. Training acts on the final prompt state, at the start of music generation. The base model, acoustic synthesis and VAE remain frozen.

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

### Ordinary LoRA versus our particle adapter

![Ordinary LoRA uses two matrices for a fixed linear correction. The particle adapter adds input-dependent routing through a shared learned cloud and a nonlinear MLP before its up projection. Both add their correction to the same frozen base projection.](assets/lora-vs-particle.svg)

[Open the SVG at full size](assets/lora-vs-particle.svg).

Both adapters add a correction to a frozen projection: $y=W_0x+s\Delta(x)$, where $s$ is the slider strength. The native rank and alpha are both 8, so $\alpha/r=1$ in these equations.

$$
\begin{aligned}
\Delta_{\mathrm{LoRA}}(x) &= BAx, \\
\Delta_{\mathrm{particle}}(x) &= U\phi\!\left([Vx,z(x)]\right).
\end{aligned}
$$

- **Ordinary LoRA:** the down matrix $A$ compresses the input into eight features; the up matrix $B$ turns those features into a correction. After training, their product is one fixed matrix $\Delta W=BA$. For a fixed strength, it can be merged into the projection as $W_0+sBA$.
- **Our particle adapter:** the down features $a=Vx$ also enter a router. Its query assigns softmax weights to the cloud $P$, producing the mixture $z(x)$. A nonlinear network $\phi$ combines the original features with that mixture before the up matrix $U$. The cloud contains 128 learned four-dimensional vectors, shared across one slider’s 112 projection branches. The vectors stay fixed during inference; their mixing weights change with the input.

An ordinary LoRA still gives different corrections for different inputs, and the full YuE2 model remains nonlinear. The extra capability here is **nonlinear computation inside the adapter itself**. In general, that whole computation cannot be represented by one fixed weight update. It does not establish that particles produce better music.

**Distillation** learns new matrices $A$ and $B$ so that $BAx$ approximates the entire particle correction on representative activations. It is a learned approximation, not removal of the particle tensors from an existing checkpoint. The original teacher’s $U,V$ and the distilled student’s $A,B$ are separate parameters.

**ParticleGAN reference:** this experiment draws on [ParticleGAN](https://github.com/255BITS/ParticleGAN), with [reference revision `441fdf42`](https://github.com/255BITS/ParticleGAN/tree/441fdf42dd2c0905af312a303add422f700c0ac2), for learned particles, paired adversarial training, the gradient cap and the particle variance/covariance regularizer. The routed transformer adapter in this diagram is our YuE2 implementation. The critic is used during training; it is not part of either inference path shown here.

### One strength control, with learned particle routing

Each autoregressive attention projection gets a nonlinear residual branch. For an input vector `x`, first compute rank-8 features and route them through one shared particle cloud:

$$
\begin{aligned}
a &= Vx, & q &= \rho(a), \\
w &= \mathrm{softmax}\!\left(\frac{Pq}{\sqrt{d_p}}\right),
& z &= P^\top w, \\
f_s(x) &= f_0(x) + s\,\frac{\alpha}{r}\,U\phi([a,z]).
\end{aligned}
$$

Here `f₀` is the frozen projection, `V` and `U` are learned down/up projections, and `s` is the slider strength. The rank and alpha are both **8**, so their ratio is 1. The shared cloud `P` contains **128 learned particles in 4 dimensions** (`dₚ = 4`). Each projection has its own router `ρ` and routed MLP `φ`: three hidden layers of width 16 and 48, respectively, with LeakyReLU slope 0.2.

At **0**, the implementation bypasses the entire branch and returns the base projection exactly. At **1**, it applies the trained positive endpoint. Intermediate values scale the residual; the resulting music need not change linearly. Negative strength is an untrained canary. The up projection starts at zero, so a fresh adapter initially preserves the base model. Learned routing runs during ordinary inference as well as training; all projections in one slider share the same cloud.

### Normalize the error against the target description

Let `hᵢ⁺` be the frozen model's final prompt state for training row `i` under the positive caption, and `hᵢθ` the adapted state under its neutral caption. The lyric sheet is identical within the pair. Normalize each coordinate using the training positive states only:

$$
\begin{aligned}
T(h) &= \frac{h-\mu_+}{\max(\mathrm{std}_{\mathrm{sample}}(h^+),10^{-4})}, \\
e_i &= T(h_i^\theta)-T(h_i^+).
\end{aligned}
$$

The mean and sample standard deviation are fixed after preparation. The floor and division act coordinate by coordinate. Held-out prompts never enter these statistics.

### Learn with a paired-error critic

At update `t`, draw Gaussian noise and add the adapted model's normalized error to a copy of the **same** noise:

$$
\begin{aligned}
\sigma_t &= 0.03^{\min(t/8000,\,1)}, &
n &\sim \mathcal N(0,\sigma_t^2 I), \\
x_{\mathrm{real}} &= n, &
x_{\mathrm{fake}} &= n+e_i.
\end{aligned}
$$

The critic `D` is a scalar MLP with three width-48 hidden layers. Its relativistic loss encourages a higher score for the noise-only input than for noise plus error. The adapter reverses that ordering:

$$
\begin{aligned}
\mathcal L_D &= \mathbb E\!\left[
\mathrm{softplus}\big(D(x_{\mathrm{fake}})-D(x_{\mathrm{real}})\big)
\right] + \mathcal R_{\mathrm{cap}}, \\
\mathcal L_G &= \mathbb E\!\left[
\mathrm{softplus}\big(D(x_{\mathrm{real}})-D(x_{\mathrm{fake}})\big)
\right] + \mathcal V(P_S).
\end{aligned}
$$

`softplus(u) = log(1 + exp(u))`. Each expectation averages **64 independently sampled row/noise pairs**, with training rows sampled with replacement. The critic and generator draw separate batches. The critic updates first; the generator then recomputes its objective through the updated critic with the critic's parameters frozen. Repeated deterministic prompt rows share a model forward, while each noise draw contributes separately to the loss.

### Cap steep critic gradients

The cap penalizes input-gradient norms only above 1, on both real and fake coordinates:

$$
\mathcal R_{\mathrm{base}} = \frac{1}{2}
\sum_{u\in\{x_{\mathrm{real}},x_{\mathrm{fake}}\}}
\mathbb E\!\left[
\max\!\left(0,\lVert\nabla_u D(u)\rVert_2-1\right)^2
\right].
$$

The implementation uses exact autograd and computes the norm as `sqrt(sum(g²) + 10⁻¹²)`. The cap coefficient is **1**. It is evaluated every fourth update, with **4 × Rbase** on those updates and zero on the others. This lazy schedule retains the penalty's average weighting.

### Keep the particle cloud spread out

For each generator update, sample **64 of the 128 particles without replacement**. Let `Pₛ` be this subset and `C` its sample covariance matrix, using denominator 63. The particle regularizer is:

$$
\mathcal V(P_S) =
\frac{1}{d_p}\sum_{j=1}^{d_p}
\max\!\left(0,1-\sqrt{C_{jj}+10^{-4}}\right)
+\frac{1}{d_p}\sum_{j\ne k} C_{jk}^{\,2},
\qquad d_p=4.
$$

The first term penalizes collapsed particle coordinates; the second penalizes covariance between different coordinates. Its coefficient is **1**. Gradients from the adversarial loss also reach the particles through the router. The generator objective has no additional output MSE, feature-matching, lyric-hold or ending-supervision term.

### Export the moving average

After each generator update, average every learned adapter parameter, including the routers and particle cloud:

$$
\bar\theta_t = 0.995\,\bar\theta_{t-1} + 0.005\,\theta_t.
$$

The published files contain this **EMA at update 1,200**. Adam uses betas `(0, 0.999)`, zero weight decay and constant learning rates: **0.0006** for the adapter projections/MLPs/routers, **0.006** for the particles, and **0.0009** for the critic. The shared cloud is registered and optimized once per slider.

The 8,000-update noise schedule is not compressed to the release budget: at update 1,200, the noise standard deviation is still about **0.591**. These are fixed-budget experimental checkpoints. The equations and hidden-state diagnostics do not establish an audio-quality, lyric-preservation or natural-ending guarantee. Automatic duration in the Space uses the native sampler's ending decision; it does not add an ending loss to these trained weights.

[Exact method and architecture record](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/FORMULATION.md) · [Checkpoint and audio audit](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/evidence/particle-1200-v1/integrity.json) · [Native loader source hashes](source-provenance.json)
