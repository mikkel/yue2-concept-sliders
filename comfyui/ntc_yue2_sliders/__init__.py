"""Native ntc-ai YuE2 particle sliders for ComfyUI's YuE2 CLIP model."""
import math

import folder_paths
from comfy.text_encoders.yue2 import YuE2TEModel

from .adapter import ParticleGeneration, load_particles


class YuE2ConceptSlider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "slider_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/yue2"
    DESCRIPTION = "Apply a native ntc-ai routed-particle slider during YuE2 composition. Connect to YuE2 Generate Music or Generate ABC."

    def apply(self, clip, slider_name, strength):
        if not isinstance(clip.cond_stage_model, YuE2TEModel):
            raise ValueError("Connect the CLIP output of ComfyUI's native YuE2 checkpoint loader.")
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("YuE2 particle-slider strength must be between 0 and 1.")
        if strength == 0:
            return (clip,)
        host = clip.cond_stage_model
        path = folder_paths.get_full_path_or_raise("loras", slider_name)
        particles = load_particles(host.config, path)
        result = clip.clone()
        original = result.patcher.get_model_object("_generate")
        result.patcher.add_object_patch("_generate", ParticleGeneration(host, original, particles, strength))
        return (result,)


# Keep the registered node ID stable so existing workflows still load.
NODE_CLASS_MAPPINGS = {"NTCYuE2ConceptSlider": YuE2ConceptSlider}
NODE_DISPLAY_NAME_MAPPINGS = {"NTCYuE2ConceptSlider": "YuE2 Particle Slider (ntc-ai)"}
