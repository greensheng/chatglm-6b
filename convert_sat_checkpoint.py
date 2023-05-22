import sys
import os
import torch
from transformers import AutoModel

state_dict = torch.load(sys.argv[1])['module']

rename_keys = []
for key in state_dict:
    if key.startswith("mixins.eva.model"):
        rename_keys.append((key, key.replace("mixins.eva.model", "image_encoder")))

for old, new in rename_keys:
    val = state_dict.pop(old)
    state_dict[new] = val
del state_dict['transformer.position_embeddings.weight']
state_dict['lm_head.weight'] = state_dict.pop('mixins.chatglm-final.lm_head.weight')

torch.save(state_dict, os.path.join(sys.argv[2], "pytorch_model.bin"))

