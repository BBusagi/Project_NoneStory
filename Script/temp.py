import torch

state_dict = torch.load("")
print(state_dict.keys())  # 看看有没有 'lm_head.weight'
