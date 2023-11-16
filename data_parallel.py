#  dataparallel: mainly focus on the first GPU. 
#  
#  distributed
#  
#
#
#


# dataparallel
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

# 定义模型
class YourModel(nn.Module):
    # ...

# 创建模型实例
model = YourModel()

# 使用DataParallel包装模型
model = DataParallel(model)

if torch.cuda.is_available():
    model.cuda()

# datadistributedparallel
import parser
import args 
import os

parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)



if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
    device=torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')


model.to(device)
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    print('use {} gpus!'.format(num_gpus))
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                output_device=args.local_rank)
    













