import logging
import numpy as np
from PIL import Image
import torch.nn as nn
import torch


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

def standarlization_multi_steps(tensor):
    mean = torch.mean(tensor, dim=2, keepdim=True)
    std = torch.std(tensor, dim=2, keepdim=True)
    return (tensor - mean) / std

def standarlization_multi_class_multi_steps(tensor):
    mean = torch.mean(tensor, dim=3, keepdim=True)
    std = torch.std(tensor, dim=3, keepdim=True)
    return (tensor - mean) / std

# xuan: only save the lazy learning weights
def save_lazy_learning_weights(model, opt, args, file_path, epoch=None, step=None):
    weights_to_save = {}
    if hasattr(model, 'blocks') or (hasattr(model, 'module') and hasattr(model.module, 'blocks')):
        all_blocks = model.blocks if hasattr(model, 'blocks') else model.module.blocks
    elif hasattr(model, 'layers') or (hasattr(model, 'module') and hasattr(model.module, 'layers')):
        all_blocks = model.layers if hasattr(model, 'layers') else model.module.layers
    else:
        raise ValueError
    for idx, block in enumerate(all_blocks):
        if hasattr(block.attn_lazy_learning, 'weight'):
            weights_to_save[f'block_{idx}_attn_lazy_learning_weight'] = block.attn_lazy_learning.weight.data.clone()
        elif len(block.attn_lazy_learning) > 0:
            for i, attn in enumerate(block.attn_lazy_learning):
                weights_to_save[f'block_{idx}_attn_lazy_learning_weight_{i}'] = attn.weight.data.clone()
        else:
            print(f'block {idx} has no attn_lazy_learning')
        if hasattr(block.mlp_lazy_learning, 'weight'):
            weights_to_save[f'block_{idx}_mlp_lazy_learning_weight'] = block.mlp_lazy_learning.weight.data.clone()
        elif len(block.mlp_lazy_learning) > 0:
            for i, mlp in enumerate(block.mlp_lazy_learning):
                weights_to_save[f'block_{idx}_mlp_lazy_learning_weight_{i}'] = mlp.weight.data.clone()
        else:
            print(f'block {idx} has no mlp_lazy_learning')

    checkpoint = {
        "model": weights_to_save,
        "opt": opt.state_dict(),
        "args": args,
        "epoch": epoch,
        "step": step,
    }
    torch.save(checkpoint, file_path)
    # print(f"Weights have been saved to {file_path}")

# xuan: only load the lazy learning weights
def load_lazy_learning_weights(model, file_path):
    checkpoint = torch.load(file_path)
    weights_to_load = checkpoint["model"]
    if hasattr(model, 'blocks') or (hasattr(model, 'module') and hasattr(model.module, 'blocks')):
        all_blocks = model.blocks if hasattr(model, 'blocks') else model.module.blocks
    elif hasattr(model, 'layers') or (hasattr(model, 'module') and hasattr(model.module, 'layers')):
        all_blocks = model.layers if hasattr(model, 'layers') else model.module.layers
    else:
        raise ValueError
    for idx, block in enumerate(all_blocks):
        if hasattr(block.attn_lazy_learning, 'weight'):
            weight_key = f'block_{idx}_attn_lazy_learning_weight'
            if weight_key in weights_to_load:
                block.attn_lazy_learning.weight.data.copy_(weights_to_load[weight_key])
            else:
                print(f"weight {weight_key} not found in checkpoint")
        elif len(block.attn_lazy_learning) > 0:
            for i, attn in enumerate(block.attn_lazy_learning):
                weight_key = f'block_{idx}_attn_lazy_learning_weight_{i}'
                if weight_key in weights_to_load:
                    attn.weight.data.copy_(weights_to_load[weight_key])
                else:
                    print(f"weight {weight_key} not found in checkpoint")
        else:
            print(f'block {idx} has no attn_lazy_learning')
        if hasattr(block.mlp_lazy_learning, 'weight'):
            weight_key = f'block_{idx}_mlp_lazy_learning_weight'
            if weight_key in weights_to_load:
                block.mlp_lazy_learning.weight.data.copy_(weights_to_load[weight_key])
            else:
                print(f"weight {weight_key} not found in checkpoint")
        elif len(block.mlp_lazy_learning) > 0:
            for i, mlp in enumerate(block.mlp_lazy_learning):
                weight_key = f'block_{idx}_mlp_lazy_learning_weight_{i}'
                if weight_key in weights_to_load:
                    mlp.weight.data.copy_(weights_to_load[weight_key])
                else:
                    print(f"weight {weight_key} not found in checkpoint")
        else:
            print(f'block {idx} has no attn_lazy_learning')

    # print(f"Weights have been loaded from {file_path}")


