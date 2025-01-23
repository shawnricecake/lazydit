# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
import argparse
from diffusers.models import AutoencoderKL

import sys; sys.path.append("../../")
from lazydit.diffusion import create_diffusion
from lazydit.models.utils import load_lazy_learning_weights


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Load model:
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

    # xuan: load weights
    model.load_state_dict(torch.load(args.ckpt), strict=False)

    # if imagenet learning:
    if hasattr(model, 'blocks'):
        all_blocks = model.blocks
    elif hasattr(model, 'layers'):  # for Large-DiT
        all_blocks = model.layers
    else:
        raise ValueError()

    for block in all_blocks:
        block.create_lazy_learning(args.num_sampling_steps, diffusion.timestep_map)

    load_lazy_learning_weights(model, args.lazy_learning_weights)
    model = model.to(device)

    # xuan: set threshold
    for block in all_blocks:
        block.attn_threshold = args.attn_threshold
        block.mlp_threshold = args.mlp_threshold

    model.accelerate = True
    model.jump_attn = 0
    model.jump_mlp = 0

    model.eval()  # important!

    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # class_labels = [207, 992, 387, 974, 142, 979, 417, 279]     # learn2cache
    # class labels that are good for sampling:
    # 388 panda
    # 248 husky
    # 291 lion
    # 105 考拉
    # 323 butterfly
    # 289 雪豹
    # 336 土拨鼠
    # 285 埃及猫
    # 207 金毛狗
    # 145 企鹅
    # 292 老虎
    # 415 面包店
    # 88  鹦鹉
    # 979 山水湖泊
    # 387 小浣熊
    # 360 水獭
    # 22  白头鹰
    # 31  树蛙
    # 99  加拿大鹅
    # 934 热狗
    # 20  水边 小肥鸟

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    if args.sample_method == "p-sample":
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    elif args.sample_method == "ddim":
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    else:
        raise ValueError("not support for {} smaple method".format(args.sample_method))

    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))

    jump_attn = model.jump_attn
    jump_mlp = model.jump_mlp
    total = len(all_blocks) * args.num_sampling_steps * z.shape[0]
    print("number of jumps in attn: {}/{}, {:.2f}%".format(
        jump_attn, total, jump_attn / total * 100
    ))
    print("number of jumps in mlp: {}/{}, {:.2f}%".format(
        jump_mlp, total, jump_mlp / total * 100
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)

    # xuan:
    parser.add_argument("--lazy_learning_weights", type=str, default=None)
    parser.add_argument("--sample-method", type=str, default="ddim")
    parser.add_argument("--attn_threshold", type=float, default=0.5)
    parser.add_argument("--mlp_threshold", type=float, default=0.5)

    args = parser.parse_args()
    main(args)
