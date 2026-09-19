"""Standalone training helpers; save method extracted from the native backend."""
import hashlib
import json
from pathlib import Path
from safetensors.torch import save_file
from slider_runtime import YuE2Slider as BaseSlider, attention_targets, sound_only, UPSTREAM_REVISION
from sound_guard import validate_sound
FORMAT = 'conceptmod-yue2-ar-v1'

def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def _artist_name_hit(*values):
    validate_sound('\n'.join(values))
    return None

class YuE2Slider(BaseSlider):
    def save(self, path, metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(metadata, format=FORMAT, rank=self.rank, alpha=self.alpha, architecture=self.architecture, targets=self.target_names, upstream_revision=UPSTREAM_REVISION)
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False)
        sound_only(payload)
        save_file({k: v.detach().float().cpu().contiguous() for k, v in self.state_dict().items()}, str(path), metadata={'conceptmod': payload})
        path.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
