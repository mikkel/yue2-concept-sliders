# YuE2 v2 particle-to-LoRA distillation

Each v2 ordinary LoRA is newly fitted to its corresponding **1,600-update EMA
gmix teacher**. These are approximate rank-8 students. They contain only down
and up matrices and alpha scalars; no particle cloud, router or inference MLP.
The teacher checkpoints and [training math](MATH.md) remain available.

## Projection regression

The teacher adds $U f(Vx,P)$ and the initial student adds $UAx$. Retain $U$
and fit $A$ through the origin on real AR inputs. With input rows in $X$,
routed-feature rows in $Y$, and
$D_{jj}=\sqrt{\max((X^TX)_{jj},10^{-12})}$, solve

$$
Z=XD^{-1},\qquad B=(Z^TZ+\lambda I)^{-1}Z^TY,\qquad A=B^TD^{-1}.
$$

The calibration set uses the release's neutral training captions, excluding
the reserved training lyric sheet, plus teacher continuations for the first
three training rows. Each continuation has a 384-token guard and seeds
4100–4102. Collect activations at teacher strengths zero and one. Q/K/V share
one input covariance per layer; O has its own.

Choose $\lambda$ from $10^{-4},10^{-3},10^{-2}$ on the fourth training row,
seed 4103. The selection score averages relative hidden steering MSE at the
first music-token boundary and over the continuation at strengths 0.5 and 1.
Both evaluation lyric sheets are excluded from fitting and selection.

## Hidden-state refinement

Starting from that regression, optimize both matrices for 100 Adam updates
at learning rate $10^{-4}$ and gradient norm cap 1. The base and teacher stay
frozen. Alternate continuation examples and broader training captions,
sampling strength 0.5 or 1. The objective is

$$
L=\frac12\sum_{r\in\{\mathrm{boundary},\mathrm{music}\}}
\frac{\operatorname{MSE}(h_S^{(r)},h_T^{(r)})}
{\max(\operatorname{MSE}(h_T^{(r)},h_0^{(r)}),10^{-7})}.
$$

Evaluate the reserved training row every ten updates. Save the lowest
validation score, including step zero as a candidate. Refinement is discarded
if it does not improve the reserved-row score. The evaluation set never
chooses the winner.

## Measurements and recordings

Use both base and teacher trajectories from the two held-out prompts, seed
1709, at strengths 0.5 and 1. Report

$$
E=\frac{\|h_S-h_T\|^2}{\max(\|h_T-h_0\|^2,10^{-12})},\qquad
C=\frac{(h_S-h_0)\cdot(h_T-h_0)}{\|h_S-h_0\|\|h_T-h_0\|}.
$$

Zero error means exact teacher agreement; one is the error from leaving the
slider off. These measure hidden-state fidelity, not perceptual quality.
The validation files also include norm ratios and sampled semantic-logit KL.

The listening gallery compares **Off**, **Particles** and **Distill** for each
control. Distill applies the ordinary LoRA during autoregressive composition.
All recordings use the native
runtime with a 500-token diagnostic guard, 16 acoustic steps and seed 1709.
These are approximately 20-second excerpts. They are not full-song tests.
The Female and Male evaluation captions specify the opposite voice at Off;
On applies the selected control to that same caption.

## ComfyUI export

Use files ending in `_comfyui.safetensors` with standard **Load LoRA**,
MODEL strength **0**, CLIP strength **1**. Place them in `ComfyUI/models/loras/`
and load the included workflow. Planning is off; automatic stopping has the
native 360-second upper guard, independent of the diagnostic clip limit.

The existing [conceptmod converter](https://github.com/mikkel/conceptmod/blob/a5c3dd8a9cdc34a86d633b9e1aed5e7efe4bc489/scripts/convert_lora_comfyui.py)
concatenates Q/K/V down matrices and block-diagonalizes their up matrices.
Fused QKV rank is 24; O rank is 8. Alpha/rank is preserved. This is exact
rearrangement before BF16 rounding. Every exported file is checked with the
real ComfyUI LoRA loader and projection probes. The original nonlinear
teachers require the particle custom node.

Standard loading applies the ordinary matrices during acoustic-prefix
processing too. Full ComfyUI GPU audio generation remains unvalidated; the
release includes real CPU loader checks and native GPU comparison audio.

## Reproduction and provenance

The release builder creates the distillations, converts them, validates the
exports and packages the results before publishing. `release-tools/README.md`
documents the standalone reproduction commands. Every student records its
teacher hash, base-model identity, calibration settings, selected refinement
step and source hashes. The release manifest hashes all published artifacts.
Published v1 weights and examples remain available under their versioned paths.

Weights retain the base model's CC BY-NC 4.0 terms. Runtime code and the
workflow include their existing license notices.


## V2 held-out measurements

| Control | Relative steering MSE | Steering cosine | Selected refinement step |
|---|---:|---:|---:|
| female | 0.0608 | 0.9693 | 0 |
| male | 0.0560 | 0.9715 | 0 |
| pop | 0.0648 | 0.9672 | 50 |
| hiphop | 0.0141 | 0.9930 | 30 |
| rnb | 0.0320 | 0.9842 | 70 |
| indie-rock | 0.0317 | 0.9841 | 100 |
| pop-punk | 0.0232 | 0.9885 | 0 |
| metal | 0.0125 | 0.9937 | 0 |
| country | 0.0213 | 0.9894 | 90 |
| acoustic-folk | 0.0370 | 0.9816 | 100 |
| house | 0.0201 | 0.9900 | 0 |
| disco-funk | 0.0283 | 0.9859 | 50 |
| kpop | 0.0479 | 0.9760 | 50 |
| reggaeton | 0.0204 | 0.9899 | 90 |
| afrobeats | 0.0144 | 0.9928 | 0 |
| lofi | 0.0269 | 0.9866 | 0 |
