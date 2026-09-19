# Create the v2 distills

Use an isolated environment with the official YuE2 runtime at
`ef1936f2ee39fe8de486a0f47a481c95f8d4da87`, PyTorch 2.11.0+cu128,
Transformers 4.57.6, NumPy, PyYAML, safetensors and soundfile. Do not install
the old conceptmod training requirements. This tool uses the packaged runtime;
it needs no local training repository. A CUDA GPU with sufficient room for
the unquantized 3B model, covariance matrices and refinement is required.
The release ran on a 48 GB GPU.

Download the versioned teachers, prompts and frozen evaluation trajectories:

```python
from huggingface_hub import snapshot_download
snapshot = snapshot_download('ntc-ai/yue2-concept-sliders', revision='v2',
    allow_patterns=['particle-gmix-1600-v2/catalog.json',
                    'weights/particle-gmix-1600-v2/*',
                    'prompts/particle-gmix-1600-v2/*',
                    'evidence/particle-gmix-1600-v2/reference/**',
                    'release-tools/*'])
print(snapshot)
```

Clone the existing converter, then run the release tool (replace SNAPSHOT):

```bash
git clone https://github.com/mikkel/conceptmod.git conceptmod-converter
git -C conceptmod-converter checkout a5c3dd8a9cdc34a86d633b9e1aed5e7efe4bc489
CUDA_VISIBLE_DEVICES=1 python SNAPSHOT/release-tools/create_distills.py \
  --snapshot SNAPSHOT --output ./v2-distills --gpu 0 \
  --converter ./conceptmod-converter --allow-hub
```

`--gpu` addresses the CUDA-visible device. The example selects physical GPU 1.
Omit `--allow-hub` to require cached base weights; `--cache-dir` selects a cache.
Use `--ids metal` for one control. Completed stages resume; use a fresh output
directory when changing the teacher or recipe.

The tool performs projection regression, 100-step hidden-state refinement,
held-out diagnostics, eight audio comparisons per control and conversion of
each final ordinary LoRA. It verifies teacher hashes, exact native roundtrips,
zero-strength bypass and finite non-silent audio. The release additionally
checks every export with real ComfyUI and records the results in
`distilled-rank8-v2/verification.json`. Full ComfyUI GPU audio generation is
not covered by those checks. Source-provenance.json describes the standalone
import adaptations; the fitting and refinement algorithms are the release's
original implementations. Numerical results can vary across GPU/runtime versions.
