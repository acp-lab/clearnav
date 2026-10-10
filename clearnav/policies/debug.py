import torch as th
import torch.nn as nn


def check_none_parameters(model):
    for name, param in model.named_parameters():
        if param is None:
            print(f"Uninitialized parameter found in layer: {name}")


def get_network_statistics(model, logger, is_record):
    stats = {}

    if is_record:
        for name, param in model.named_parameters():
            key = "debug/" + name
            if "weight" in name:
                stats = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "max": param.data.abs().max().item(),
                }
                logger.record(key + ".mean", stats["mean"])
                logger.record(key + ".std", stats["std"])
                logger.record(key + ".max", stats["max"])

            logger.record(key, param)

    # return stats


def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:  # Some parameters may not have gradients
            param_norm = p.grad.norm(2)  # L2 norm
            total_norm += param_norm.item() ** 2
    return total_norm**0.5  # sqrt of sum of squares
