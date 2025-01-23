# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import random
from diffusers.models import AutoencoderKL

import sys; sys.path.append("../../")
from lazydit.diffusion import create_diffusion
from lazydit.models.utils import save_lazy_learning_weights, load_lazy_learning_weights


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
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


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if 'Llama' in args.model:  # for Large-DiT
        from lazydit.models.models_lazy_learning_imagenet_largedit import DiT_models
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        )
    else:
        from lazydit.models.models_lazy_learning_imagenet import DiT_models
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        )

    # xuan: load pretrained model weights
    model.load_state_dict(torch.load(args.ckpt), strict=True)

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    if hasattr(model, 'blocks'):
        all_blocks = model.blocks
    elif hasattr(model, 'layers'):  # for Large-DiT
        all_blocks = model.layers
    else:
        raise ValueError()
    for block in all_blocks:
        block.create_lazy_learning(args.num_sampling_steps, diffusion.timestep_map)
        if args.resume is None:
            for lazy_learning_layer in block.attn_lazy_learning:
                lazy_learning_layer.weight.data.fill_(0)
            for lazy_learning_layer in block.mlp_lazy_learning:
                lazy_learning_layer.weight.data.fill_(0)

    if args.resume is not None:
        load_lazy_learning_weights(model. args.resume)
        resume_ckpt = torch.load(args.resume)

    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)   # xuan: important !!!!

    # xuan: only optimize the lazy learning weights
    parameters_to_optimize = [param for name, param in model.named_parameters() if "lazy_learning" in name]
    opt = torch.optim.AdamW(parameters_to_optimize, lr=args.lr, weight_decay=0)
    if args.resume is not None:
        opt.load_state_dict(resume_ckpt["opt"])

    for name, param in model.named_parameters():
        if "lazy_learning" not in name:
            param.requires_grad = False

    logger.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance

    # Variables for monitoring/logging purposes:
    start_epoch = 0
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_main_loss = 0
    running_lazy_loss = 0

    if args.resume is not None:
        start_epoch = resume_ckpt["epoch"]
        train_steps = resume_ckpt["step"]

    start_time = time()

    logger.info("======================= Training Details ===================================")
    logger.info(
        "Training {} with image size: {} and sampling steps: {}; "
        "epoch: {}; learning rate: {}; local batch size: {}; "
        "attn lazy loss ratio: {}; MLP lazy loss ratio: {}".format(
            args.model, args.image_size, args.num_sampling_steps,
            args.epochs, args.lr, args.local_batch_size,
            args.attn_lazy_loss_ratio, args.mlp_lazy_loss_ratio
        )
    )
    if args.resume is not None:
        logger.info(
            "Resume training from checkpoint: {}; epoch: {}; steps: {}".format(
                args.resume, start_epoch, train_steps
            )
        )
    logger.info("============================================================================")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch+1}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            t = torch.randint(0, diffusion.num_timesteps // 2, (1,), device=device)
            t = t.repeat(x.shape[0]) * 2 + 1

            loss_dict = diffusion.lazy_learning_loss(model, x, t, model_kwargs)

            main_loss = loss_dict['main_mse_loss'].mean()
            attn_lazy_loss = loss_dict['attn_lazy_loss'].mean()
            mlp_lazy_loss = loss_dict['mlp_lazy_loss'].mean()
            lazy_loss = args.attn_lazy_loss_ratio * attn_lazy_loss \
                          + args.mlp_lazy_loss_ratio * mlp_lazy_loss
            loss = main_loss + lazy_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.empty_cache()

            # xuan: reset cache !!!!!!!!!!!!!!!!!!
            if hasattr(model.module, 'blocks'):
                all_blocks = model.module.blocks
            elif hasattr(model.module, 'layers'):  # for Large-DiT
                all_blocks = model.module.layers
            else:
                raise ValueError()
            for block in all_blocks:
                block.cache_attn = None
                block.cache_mlp = None
            model.module.mask_attn_total_time = []
            model.module.mask_mlp_total_time = []

            # 记录损失值
            running_loss += loss.item()
            running_main_loss += main_loss.item()
            running_lazy_loss += lazy_loss
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0 or train_steps == 1:
                # 测量训练速度
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # 计算平均损失
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                avg_main_loss = torch.tensor(running_main_loss / log_steps, device=device)
                avg_lazy_loss = torch.tensor(running_lazy_loss / log_steps, device=device)

                logger.info(
                    f"(Step={train_steps:07d}) "
                    f"Distributed Train Loss: {avg_loss:.4f}, "
                    f"Main Loss: {avg_main_loss:.4f}, "
                    f"lazy Loss: {avg_lazy_loss:.4f}, "
                    # f"Train Steps/Sec: {steps_per_sec:.2f}"
                    f"Mean Attn lazy: {loss_dict['attn_lazy']*100:.2f}%, "
                    f"Mean MLP lazy: {loss_dict['mlp_lazy']*100:.2f}%, "
                )
                # 重置监控变量
                running_loss = 0
                running_main_loss = 0
                running_lazy_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    save_lazy_learning_weights(model, opt, args, checkpoint_path, epoch, train_steps)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps >= args.max_step:
                break
        if train_steps >= args.max_step:
            break

    # Save final DiT checkpoint:
    if rank == 0:
        checkpoint_path = f"{checkpoint_dir}/final_ckpt.pt"
        save_lazy_learning_weights(model, opt, args, checkpoint_path, epoch, train_steps)
        logger.info(f"Saved final checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--local-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=200)

    # xuan
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--attn_lazy_loss_ratio", type=float, default=1e-2)
    parser.add_argument("--mlp_lazy_loss_ratio", type=float, default=1e-2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_step", type=int, default=500)

    args = parser.parse_args()
    main(args)
