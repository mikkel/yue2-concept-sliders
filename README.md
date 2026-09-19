# YuE2 Concept Sliders v2

Sixteen voice and genre controls for **YuE2-3B**. Version 2 publishes the final
**1,600-update EMA gmix teachers**, newly distilled rank-8 ordinary LoRAs,
ComfyUI exports and matched listening comparisons.

[Download v2](https://github.com/mikkel/yue2-concept-sliders/releases/tag/v2) · [How it works and math](#how-the-sliders-learn) · [Distillation method and measurements](DISTILLATION.md) · [Native usage](USAGE.md) · [Live Space — v1 particles](https://huggingface.co/spaces/ntc-ai/yue2-concept-sliders)

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

## How the sliders learn

Version 2 contains the final EMA checkpoint at **1,600 updates** for all 16
controls. The inference architecture remains the routed-particle adapter. The
training formulation changes its training sources, error normalization, critic
and noise schedule. The release also creates new ordinary rank-8 students from
these exact v2 teachers. The archived v1 math describes the earlier release.
The embedded adapter-format identifier still ends in `ar-v1`: its tensor
layout and inference architecture are unchanged. It is separate from the
release version and the training formulation.

### Why train with particles?

The aim is to learn a useful change in YuE2's behavior while keeping the base
model frozen. **G** is the trainable slider attached to YuE2; **D** is a critic
that learns to recognize the remaining error against a positive-caption
hidden target. G learns the correction through that adversarial signal.

The particle cloud gives the adapter a shared, trainable set of features to
draw on. Each projection routes its current input through those features,
then combines them with its own low-rank features using a nonlinear network.
The routing can change with the musical context. The variance/covariance
penalty encourages the cloud to retain spread, while the critic gradient cap
discourages overly steep critic responses. Together, these are
the design rationale for a more flexible adapter and a manageable training
game. Their separate effects require controlled experiments.

The recipe draws on [ParticleGAN at revision `441fdf42`](https://github.com/255BITS/ParticleGAN/tree/441fdf42dd2c0905af312a303add422f700c0ac2):
learned particles, relativistic adversarial losses, a gradient cap and
variance/covariance regularization. Our YuE2 adapter uses input-dependent
routing through the cloud inside transformer projections. The critic is
needed during training only; the cloud and routing remain in the native
adapter at inference. Distillation learns an ordinary LoRA that approximates
this correction for standard loaders.

### Historical evidence: why we pursued this recipe

The following graph is restored from **v1**. It compares earlier **Metal,
seed 7** experiments using the same frozen base, four prompt/lyric pairs and
rank-8 attention targets. It motivated further work on the particle recipe.
The v2 checkpoints use the revised formulation below and finish at 1,600
updates; they are not the runs plotted here.

![Historical v1 training curves: generator adversarial loss and hidden-target cosine for the earlier ordinary-LoRA GAN and routed-particle recipes.](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/assets/gan-training-stability.svg)

[Full-size graph](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/assets/gan-training-stability.svg) · [Measurements](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/evidence/gan-stability/training-curves.csv) · [Recipes and provenance](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/evidence/gan-stability/provenance.json)

| Historical recipe | Updates | Peak G adversarial loss | Update at peak |
|---|---:|---:|---:|
| Ordinary LoRA, original GAN recipe | 600 | 549.76 | 373 |
| Routed particles, v1 recipe | 1,200 | 35.66 | 945 |

Every logged update is shown without smoothing. The upper panel plots the
generator's adversarial term, excluding particle regularization, on a log
scale. The lower panel measures alignment with the hidden target. The earlier
ordinary-LoRA run has a large loss spike and loses alignment; the particle
run stays close to the target through the released **v1** Metal checkpoint.
The dashed line marks the end of the 600-update run.

This is evidence about the complete historical recipes. Their critics,
normalization, noise, batching, optimizer settings and regularization also
differ, so the graph cannot attribute the change to particles alone or rank
audio quality from loss values. The ordinary-LoRA curve is an earlier GAN
experiment; the current **Distill** adapters instead learn from particle
teachers through regression and hidden-state matching.

### Ordinary LoRA and the routed-particle adapter

![Ordinary LoRA adds a fixed linear weight update. The routed-particle adapter adds input-dependent cloud routing and a nonlinear bridge before its up projection.](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/assets/lora-vs-particle.svg)

[Full-size architecture diagram](https://huggingface.co/ntc-ai/yue2-concept-sliders/resolve/main/assets/lora-vs-particle.svg)

Both paths add a strength-scaled correction to the frozen projection:

$$
\begin{aligned}
y&=W_0x+s\Delta(x),\\
\Delta_{\mathrm{LoRA}}(x)&=BAx,\\
\Delta_{\mathrm{particle}}(x)&=U\phi([Vx;z(x)]).
\end{aligned}
$$

An ordinary LoRA compresses the input with A and expands it with B.
Their product is a fixed matrix, so the update can be merged as W₀ + sBA.
The particle adapter also routes the compressed input through its cloud to
obtain z(x). Its nonlinear bridge combines these features before U.
The extra capability is nonlinear computation inside the adapter itself;
the full base model remains nonlinear in both cases.

**Distill** learns separate matrices A and B to approximate the teacher's
whole correction on representative activations. It trades that nonlinear
inference path for the simple two-matrix update supported by ordinary LoRA
loaders. The architecture diagram applies to both v1 and v2.

### Inference: a nonlinear correction in AR attention

YuE2 has 28 autoregressive layers. Each slider modifies their Q, K, V and O
projections: 112 branches, each with rank/alpha 8/8. NAR attention, both MLP
paths, embeddings, normalization, output head and VAE remain frozen. During
native generation the adapter runs in semantic composition, and its hooks are
removed before acoustic synthesis.

For input column vector x, a branch uses its own down projection V, router
R, nonlinear bridge φ and up projection U. The learned cloud P contains
128 four-dimensional vectors shared across this slider's branches:

$$
\begin{aligned}
a&=Vx,\qquad q=R(a),\\
w&=\operatorname{softmax}(Pq/\sqrt4),\\
z&=P^Tw,\\
y&=W_0x+s\,U\phi([a;z]).
\end{aligned}
$$

The router has three width-16 hidden layers; the bridge has three width-48
hidden layers. Both use LeakyReLU with slope 0.2. The cloud is learned once
per slider and stays fixed at inference; its mixing weights depend on each
input. Strength zero bypasses the branch exactly. Strength one is the trained
positive endpoint. Intermediate strengths scale the correction; negative
strengths are unsupported. This nonlinear adapter cannot be merged into a
fixed base-weight update. The critic is used only in training.

### Seedbank supervision

Each of four sound-only caption/lyric templates supplies **128 distinct
continuation seeds**, giving 512 training sources. For each source the frozen
base generates a **32-token neutral history**. The same history is appended
to the neutral and positive prefixes. Their final hidden states are nᵢ and
tᵢ; the student on the neutral prefix and that history produces gᵢ.
The target is the raw positive state tᵢ. This is supervision at the end of
each sampled history, rather than a loss on every music token.

The generator and critic independently draw batches of eight sources with
replacement. Repeated sources share a model forward within a phase but receive
independent noise draws. The run seed is 7. The kept Female and Male runs use
the expanded vocal cues and the h13 noise hold recorded in their metadata.
The failed one-word Male run and the superseded gender candidates are excluded.

### Normalize the paired edit

Let eᵢ = tᵢ − nᵢ and H = 2048. Compute sample standard deviations over the
512 paired edits, then choose a scalar gain so their median normalized row
RMS is one:

$$
\begin{aligned}
d_j&=\max(\operatorname{std}_i(e_{ij}),10^{-4}),\\
\rho_i&=\sqrt{\frac1H\sum_j(e_{ij}/d_j)^2},\\
m&=\max(\operatorname{median}_i\rho_i,10^{-4}),\\
s_j&=m\,d_j,\\
E&=\sqrt{\frac1{NH}\sum_{i,j}(e_{ij}/s_j)^2}.
\end{aligned}
$$

The implementation subtracts the mean positive target from each hidden state
and divides coordinatewise by s. The mean cancels in the paired
error, so the critic sees (gᵢ − tᵢ) / s. In v1 the coordinate scales came
from absolute target states. Here they come from the desired edit, so small
vocal edits are not measured against the much larger spread of unrelated
hidden states. Median row RMS is one; overall RMS E generally differs from
one. `teacher-audit.json` records both the normalization and initial noise.

### Paired-error game and noise

For each source draw one shared Gaussian vector for its real/fake pair:

$$
\begin{aligned}
\epsilon_i&\sim\mathcal N(0,\sigma_t^2I),\\
r_i&=\epsilon_i,\\
f_i&=\epsilon_i+(g_i-t_i)/s,\\
\sigma_t^{\rm raw}&=\sigma_0
\left(\frac{0.03}{\sigma_0}\right)^{\min(t/T,1)},\\
\sigma_0&=\max(E/0.28,0.03).
\end{aligned}
$$

The v2 set spans three recorded schedules. **Metal** has T = 8000 and its
original exponential anneal (the final recorded noise is about 1.537).
**Pop and Hip-Hop** have T = 1600 with a fixed hold of 1; Pop resumed with
the hold after update 462. The other 13 controls, including both kept gender
runs, use T = 1600 and hold the scheduled noise at a minimum of 1.3E.
For those holds the exponential start exceeds the hold. Consequently the
release does **not** reach noise 0.03 at update 1600. Per-run traces and
normalization audits are included in `evidence/particle-gmix-1600-v2/`.

### Global-mix critic

The critic learns a linear map from all 2048 error coordinates into eight
48-dimensional tokens and adds learned token positions. One four-head
attention block processes those tokens. Each token can mix the entire hidden
state; contiguous hidden coordinates are not assumed to be meaningful patches.
The block uses RMS normalization, residual self-attention and a residual MLP.
Afterward the critic independently RMS-normalizes the mean and elementwise
maximum across tokens, concatenates them, and predicts a bounded score:

$$
\begin{aligned}
X&=\operatorname{reshape}_{8\times48}(Ae+b)+P_D,\\
Z&=\operatorname{AttentionBlock}(X),\\
v&=\begin{bmatrix}
\operatorname{RMSNorm}(\operatorname{mean}Z)\\
\operatorname{RMSNorm}(\operatorname{max}Z)
\end{bmatrix},\\
D(e)&=8\tanh\left(\frac{u^Tv+b_D}{8}\right).
\end{aligned}
$$

The actual checkpoint setting is `gmix_t8_w48_l1`, with four heads and score
bound 8. Later trainer defaults are not a description of these trained files.

### Objectives, gradient cap and moving average

Using paired relativistic logistic losses:

$$
\begin{aligned}
L_D&=\mathbb E\,\operatorname{softplus}(D(f)-D(r))+R_D,\\
L_G&=\mathbb E\,\operatorname{softplus}(D(r)-D(f))+\mathcal V(P_S).
\end{aligned}
$$

Every fourth update applies the lazy gradient cap

$$
\begin{aligned}
c(v)&=\max(0,\|\nabla_vD\|_2-1)^2,\\
R_D&=4\cdot\frac12\left(\mathbb E_r c(r)+\mathbb E_f c(f)\right).
\end{aligned}
$$

It is zero on other updates. The factor four compensates for the lazy
frequency. A fresh subset of 64 particles, sampled without replacement, gives
sample covariance C with denominator 63. Its variance/covariance penalty is

$$
\begin{aligned}
\mathcal V(P_S)={}&\frac14\sum_{j=1}^4
\max(0,1-\sqrt{C_{jj}+10^{-4}})\\
&+\frac14\sum_{j\ne k}C_{jk}^2.
\end{aligned}
$$

There is no extra output MSE, lyric preservation or ending loss in teacher
training. Adam uses betas (0, 0.999), zero weight decay and constant rates:
0.0006 for the adapter branches, 0.006 for the shared particles, and 0.0009 for
the critic. After each update all learned adapter parameters enter an EMA:

$$\bar\theta_t=0.995\bar\theta_{t-1}+0.005\theta_t.$$

The release uses final EMA weights, not a listening-quality selection. Training
cosines and critic losses diagnose hidden-state optimization; they do not
establish perceptual quality, lyric preservation or natural endings.

### Ordinary LoRA distillation and ComfyUI

For each exact v2 teacher, regression fits a rank-8 linear down projection to
its routed features while retaining its up projection. Hidden-state refinement
then optimizes both matrices, selecting on a reserved training lyric sheet.
The two evaluation lyric sheets stay excluded from fitting and selection.
See [DISTILLATION.md](DISTILLATION.md) for the regression, refinement and
held-out error equations and the resulting measurements.

An ordinary student adds sBAx. ComfyUI fuses Q/K/V by concatenating their
down matrices and placing their up matrices on a block diagonal: fused QKV
rank 24, O rank 8, with alpha/rank preserved. Conversion is exact before BF16
rounding; teacher-to-student distillation is an approximation. Standard
**Load LoRA**, MODEL **0**, CLIP **1**, also affects acoustic-prefix processing.
The native particle custom node applies its correction only during AR
generation. The gallery's Distill examples apply the ordinary LoRA only during
autoregressive composition.

## Reproduction and previous versions

[Release tools](release-tools/README.md) create the ordinary distills from the
versioned teacher catalog, then convert and validate the exports. The
[v2 manifest](https://huggingface.co/ntc-ai/yue2-concept-sliders/blob/main/particle-gmix-1600-v2/release-manifest.json) records file hashes;
each student records its exact teacher hash. Training and evaluation prompts
contain sound descriptions. Weights retain CC BY-NC 4.0 terms.

[V1 particle weights](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/weights/particle-1200-v1) · [V1 ordinary LoRAs](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/main/distilled-rank8-v1) · [V1 README and math at their original revision](https://huggingface.co/ntc-ai/yue2-concept-sliders/tree/f7c2e7ab024b6505f009ea9c6f7deb36e3373bc0).
The live Space continues to run its pinned v1 particle deployment.
