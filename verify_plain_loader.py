"""CPU check of the ordinary LoRA loader, independent of training packages."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from torch import nn
from safetensors.torch import save_file
from slider_runtime import YuE2Slider, ParticleSlider, architecture, attention_targets

torch.set_num_threads(1)
torch.manual_seed(129)
torch.set_grad_enabled(False)
model=nn.Module()
model.config=SimpleNamespace(model_type='yue2',hidden_size=32,num_hidden_layers=1,
    num_attention_heads=2,num_key_value_heads=1,head_dim=16,vocab_size=184704,intermediate_size=64)
model.model=nn.Module()
layer=nn.Module();layer.self_attn=nn.Module()
for name,width in [('q_proj',32),('k_proj',16),('v_proj',16),('o_proj',32)]:
    setattr(layer.self_attn,name,nn.Linear(32,width,bias=False))
model.model.layers=nn.ModuleList([layer])
targets=attention_targets(model)
originals={k:v.forward for k,v in targets.items()}
state={}
for name,module in targets.items():
    prefix='adapters.'+name.replace('.','-')
    state[prefix+'.lora_down.weight']=torch.randn(8,32)*.1
    state[prefix+'.lora_up.weight']=torch.randn(module.out_features,8)*.1
    state[prefix+'.alpha']=torch.tensor(8.)
record=dict(format='conceptmod-yue2-ar-v1',rank=8,alpha=8.,architecture=architecture(model),targets=list(targets))
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'ordinary.safetensors'
    save_file(state,path,metadata={'conceptmod':json.dumps(record)})
    adapter,_=YuE2Slider.load(model,path)
    x=torch.randn(2,5,32)
    for scale in (0.,.5,1.):
        with adapter.scaled(scale):
            for name,module in targets.items():
                prefix='adapters.'+name.replace('.','-')
                base=originals[name](x)
                wanted=base+((x@state[prefix+'.lora_down.weight'].T)@state[prefix+'.lora_up.weight'].T)*scale
                torch.testing.assert_close(module(x),wanted,rtol=1e-6,atol=1e-7)
                if scale==0:assert torch.equal(module(x),base)
    for name,module in targets.items():module.forward=originals[name]
    state.pop('adapters.model-layers-0-self_attn-k_proj.lora_up.weight')
    save_file(state,path,metadata={'conceptmod':json.dumps(record)})
    try:YuE2Slider.load(model,path)
    except ValueError:pass
    else:raise AssertionError('Incomplete checkpoint accepted')
    assert all(module.forward==originals[name] for name,module in targets.items())
    particles=ParticleSlider(model)
    for branch in particles.adapters.values():nn.init.normal_(branch.lora_up.weight,std=.1)
    with particles.scaled(1.):expected={name:module(x) for name,module in targets.items()}
    particle_path=Path(folder)/'particles.safetensors'
    particles.save(particle_path,{'purpose':'native loader dispatch check'})
    for name,module in targets.items():module.forward=originals[name]
    restored,_=YuE2Slider.load(model,particle_path)
    assert isinstance(restored,ParticleSlider)
    with restored.scaled(1.):
        for name,module in targets.items():torch.testing.assert_close(module(x),expected[name],rtol=0,atol=0)
print('Ordinary LoRA parity, exact zero, pre-attachment rejection and particle dispatch passed.')
