import torch


def module_size(module: torch.nn.Module):
    total_bytes = 0
    for p in module.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes
