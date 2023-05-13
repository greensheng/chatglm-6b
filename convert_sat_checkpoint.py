import sys
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


model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, revision="visual")
output = model.load_state_dict(state_dict)

model.save_pretrained(sys.argv[2], max_shard_size="4GB")

