# YuE2 v2: routed particles trained with a global-mix critic

Version 2 contains the final EMA checkpoint at **1,600 updates** for all 16
controls. The inference architecture remains the routed-particle adapter. The
training formulation changes its training sources, error normalization, critic
and noise schedule. The release also creates new ordinary rank-8 students from
these exact v2 teachers. The archived v1 math describes the earlier release.
The embedded adapter-format identifier still ends in `ar-v1`: its tensor
layout and inference architecture are unchanged. It is separate from the
release version and the training formulation.

## Inference: a nonlinear correction in AR attention

YuE2 has 28 autoregressive layers. Each slider modifies their Q, K, V and O
projections: 112 branches, each with rank/alpha 8/8. NAR attention, both MLP
paths, embeddings, normalization, output head and VAE remain frozen. During
native generation the adapter runs in semantic composition, and its hooks are
removed before acoustic synthesis.

For input column vector $x$, a branch uses its own down projection $V$, router
$R$, nonlinear bridge $\phi$ and up projection $U$. The learned cloud
$P\in\mathbb R^{128\times4}$ is shared across this slider's branches:

$$
a=Vx,\qquad q=R(a),\qquad
w=\operatorname{softmax}(Pq/\sqrt4),\qquad z=P^Tw,
\qquad y=W_0x+s\,U\phi([a;z]).
$$

The router has three width-16 hidden layers; the bridge has three width-48
hidden layers. Both use LeakyReLU with slope 0.2. The cloud is learned once
per slider and stays fixed at inference; its mixing weights depend on each
input. Strength zero bypasses the branch exactly. Strength one is the trained
positive endpoint. Intermediate strengths scale the correction; negative
strengths are unsupported. This nonlinear adapter cannot be merged into a
fixed base-weight update. The critic is used only in training.

## Seedbank supervision

Each of four sound-only caption/lyric templates supplies **128 distinct
continuation seeds**, giving 512 training sources. For each source the frozen
base generates a **32-token neutral history**. The same history is appended
to the neutral and positive prefixes. Their final hidden states are $n_i$ and
$t_i$; the student on the neutral prefix and that history produces $g_i$.
The target is the raw positive state $t_i$. This is supervision at the end of
each sampled history, rather than a loss on every music token.

The generator and critic independently draw batches of eight sources with
replacement. Repeated sources share a model forward within a phase but receive
independent noise draws. The run seed is 7. The kept Female and Male runs use
the expanded vocal cues and the h13 noise hold recorded in their metadata.
The failed one-word Male run and the superseded gender candidates are excluded.

## Normalize the paired edit

Let $e_i=t_i-n_i$ and $H=2048$. Compute sample standard deviations over the
512 paired edits, then choose a scalar gain so their median normalized row
RMS is one:

$$
d_j=\max(\operatorname{std}_i(e_{ij}),10^{-4}),\qquad
m=\max\left(\operatorname{median}_i
\sqrt{\frac1H\sum_j(e_{ij}/d_j)^2},10^{-4}\right),\qquad
s_j=m\,d_j,\qquad
E=\sqrt{\frac1{NH}\sum_{i,j}(e_{ij}/s_j)^2}.
$$

The implementation normalizes hidden states as
$\tilde h=(h-\operatorname{mean}_i t_i)/s$. The mean cancels in the paired
error, so the critic sees $(g_i-t_i)/s$. In v1 the coordinate scales came
from absolute target states. Here they come from the desired edit, so small
vocal edits are not measured against the much larger spread of unrelated
hidden states. Median row RMS is one; overall RMS $E$ generally differs from
one. `teacher-audit.json` records both the normalization and initial noise.

## Paired-error game and noise

For each source draw one shared Gaussian vector for its real/fake pair:

$$
\epsilon_i\sim\mathcal N(0,\sigma_t^2I),\qquad
r_i=\epsilon_i,\qquad
f_i=\epsilon_i+(g_i-t_i)/s,\qquad
\sigma_t^{\rm raw}=\sigma_0
\left(\frac{0.03}{\sigma_0}\right)^{\min(t/T,1)},\qquad
\sigma_0=\max(E/0.28,0.03).
$$

The v2 set spans three recorded schedules. **Metal** has $T=8000$ and its
original exponential anneal (the final recorded noise is about 1.537).
**Pop and Hip-Hop** have $T=1600$ with a fixed hold of 1; Pop resumed with
the hold after update 462. The other 13 controls, including both kept gender
runs, use $T=1600$ and $\sigma_t=\max(\sigma_t^{\rm raw},1.3E)$.
For those holds the exponential start exceeds the hold. Consequently the
release does **not** reach noise 0.03 at update 1600. Per-run traces and
normalization audits are included in `evidence/particle-gmix-1600-v2/`.

## Global-mix critic

The critic learns a linear map from all 2048 error coordinates into eight
48-dimensional tokens and adds learned token positions. One four-head
attention block processes those tokens. Each token can mix the entire hidden
state; contiguous hidden coordinates are not assumed to be meaningful patches.
The block uses RMS normalization, residual self-attention and a residual MLP.
Afterward the critic independently RMS-normalizes the mean and elementwise
maximum across tokens, concatenates them, and predicts a bounded score:

$$
X=\operatorname{reshape}_{8\times48}(Ae+b)+P_D,\qquad
Z=\operatorname{AttentionBlock}(X),\qquad
D(e)=8\tanh\left(
\frac{u^T[\operatorname{RMSNorm}(\operatorname{mean}Z);
\operatorname{RMSNorm}(\operatorname{max}Z)]+b_D}{8}
\right).
$$

The actual checkpoint setting is `gmix_t8_w48_l1`, with four heads and score
bound 8. Later trainer defaults are not a description of these trained files.

## Objectives, gradient cap and moving average

Using paired relativistic logistic losses:

$$
L_D=\mathbb E\,\operatorname{softplus}(D(f)-D(r))+R_D,
\qquad
L_G=\mathbb E\,\operatorname{softplus}(D(r)-D(f))+\mathcal V(P_S).
$$

Every fourth update applies the lazy gradient cap

$$
R_D=4\cdot\frac12\left(
\mathbb E_r[\max(0,\|\nabla_rD\|_2-1)^2]+
\mathbb E_f[\max(0,\|\nabla_fD\|_2-1)^2]\right).
$$

It is zero on other updates. The factor four compensates for the lazy
frequency. A fresh subset of 64 particles, sampled without replacement, gives
sample covariance $C$ with denominator 63. Its variance/covariance penalty is

$$
\mathcal V(P_S)=\frac14\sum_{j=1}^4
\max(0,1-\sqrt{C_{jj}+10^{-4}})
+\frac14\sum_{j\ne k}C_{jk}^2.
$$

There is no extra output MSE, lyric preservation or ending loss in teacher
training. Adam uses betas $(0,0.999)$, zero weight decay and constant rates:
0.0006 for the adapter branches, 0.006 for the shared particles, and 0.0009 for
the critic. After each update all learned adapter parameters enter an EMA:

$$\bar\theta_t=0.995\bar\theta_{t-1}+0.005\theta_t.$$

The release uses final EMA weights, not a listening-quality selection. Training
cosines and critic losses diagnose hidden-state optimization; they do not
establish perceptual quality, lyric preservation or natural endings.

## Ordinary LoRA distillation and ComfyUI

For each exact v2 teacher, regression fits a rank-8 linear down projection to
its routed features while retaining its up projection. Hidden-state refinement
then optimizes both matrices, selecting on a reserved training lyric sheet.
The two evaluation lyric sheets stay excluded from fitting and selection.
See [DISTILLATION.md](DISTILLATION.md) for the regression, refinement and
held-out error equations and the resulting measurements.

An ordinary student adds $sBAx$. ComfyUI fuses Q/K/V by concatenating their
down matrices and placing their up matrices on a block diagonal: fused QKV
rank 24, O rank 8, with alpha/rank preserved. Conversion is exact before BF16
rounding; teacher-to-student distillation is an approximation. Standard
**Load LoRA**, MODEL **0**, CLIP **1**, also affects acoustic-prefix processing.
The native particle custom node applies its correction only during AR
generation. The gallery's Distill examples apply the ordinary LoRA only during
autoregressive composition.
