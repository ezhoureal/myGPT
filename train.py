
import math
import os
import time
import torch
import torch.nn as nn
import tqdm
from data_loader import DataLoader
from model import GPT, Config

from torch.distributed import init_process_group, destroy_process_group

MAX_STEP = 50
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEP = 10

def get_lr(step):
    if step < WARMUP_STEP:
        return MIN_LR + step / WARMUP_STEP * (MAX_LR - MIN_LR)
    if step > MAX_STEP:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEP) / (MAX_STEP - WARMUP_STEP)
    return MIN_LR + (math.cos(decay_ratio * math.pi) / 2.0 + 0.5) * (MAX_LR - MIN_LR)

torch.set_float32_matmul_precision('high')
def train(device: torch.device):
    desired_batch = 2
    B = 1
    grad_accum_step = desired_batch // B # to simulate larger batch size
    T = 1024
    model = GPT(Config(vocab_size=50304)).to(device)
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, [ddp_local_rank])
    raw_model = model.module if ddp else model
    # model = torch.compile(model)
    optimizer = raw_model.configure_optimizer(weight_decay=0.1,learning_rate=3e-4, device=device)
    data = DataLoader(device)
    for i in tqdm.trange(10):
        x, y = data.get_batch(B, T)

        t1 = time.time()
        optimizer.zero_grad()
        accum_loss = 0
        for inner_step in range(grad_accum_step):
            if device == 'cuda':
                with torch.autocast(device_type=device.__str__(), dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            loss /= grad_accum_step
            accum_loss += loss.detach()
            if ddp: # only sync backward when accum grad finishes
                assert isinstance(model, nn.parallel.DistributedDataParallel)
                model.require_backward_grad_sync = inner_step == grad_accum_step - 1
            loss.backward()
        if ddp:
            accum_loss = torch.distributed.all_reduce(accum_loss, op=torch.distributed.ReduceOp.AVG)

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(i)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()
        print(f'loss = {accum_loss}, token processed speed = {B * T / (t2 - t1)}, norm = {norm:.3e}, learning rate = {lr:.3e}')


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get('RANK'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    is_master = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    world_size = 1
    is_master = True
    # Check if CUDA is available and set the device
    device = (
        # torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
print(f'using device {device}')

train(device=device)
if ddp:
    destroy_process_group()
