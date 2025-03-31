import torch
print(torch.__version__)
print(torch.version.cuda) 
print(torch.cuda.is_available())      # 应该为 True
print(torch.cuda.device_count())      # 应该 >= 1
print(torch.cuda.get_device_name(0))  # 应该返回你的显卡名
a = torch.rand(10000, 10000).cuda()       # 实际调度
