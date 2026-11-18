# Reviewed GitHub releases

Stage the approved files in a GitHub draft release. Set its target to the full
source commit used to prepare the release and use a tag that does not already
point to a different revision. The source commit must be in `main` history.
Include the particle, native LoRA and ComfyUI LoRA ZIPs using the corresponding
`-particles.zip`, `-native-loras.zip` and `-comfyui-loras.zip` suffixes. Include
`SHA256SUMS` covering the three ZIPs and a `-verification.json` record with the
completed distillation reports and number of particle checkpoints. Each archive
must contain the same controls and their configuration files.

Open a PR adding or updating `release.json` in the repository root.

| Field | Value |
| --- | --- |
| `tag` | New bundle version, such as `v3` or `v3.1.0` |
| `release_id` | Numeric ID of the draft GitHub release |
| `source_commit` | Full Git commit used to produce the reviewed release |
| `prerelease` | Boolean matching the draft's prerelease setting |
| `files` | Map of exact attachment filenames to `bytes` and `sha256` records |

Run `python release-tools/publish_github_release.py audit` with GitHub CLI access to the draft.
Keep the reviewed draft fixed while the request is open. Merging the request
rechecks the GitHub metadata, downloads and inspects each attachment, publishes
the draft, and verifies the public tag and attachment records. An exact retry
accepts the existing release after verifying it again.
