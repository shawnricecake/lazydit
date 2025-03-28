# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from typing import Optional, Tuple, List

from flash_attn import flash_attn_func
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


from apex.normalization import FusedRMSNorm as RMSNorm
# if no apex:
# class RMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-8):
#         super(RMSNorm, self).__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#
#     def forward(self, x):
#         norm = x.norm(2, dim=-1, keepdim=True)
#         rms = norm * (x.size(-1) ** -0.5)
#         x = x / (rms + self.eps)
#         return self.weight * x


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, thres=0.5):
        return (i >= thres).float()     # xuan:  initialize for lazy learning is all zeros, after sigmoid is 0.5

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#############################################################################
#             Embedding Layers for Timesteps and Class Labels               #
#############################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size, hidden_size, bias=True,
            ),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32
            ) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([
                embedding, torch.zeros_like(embedding[:, :1])
            ], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb


class LabelEmbedder(nn.Module):
    r"""Embeds class labels into vector representations. Also handles label
    dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size,
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(
                labels.shape[0], device=labels.device
            ) < self.dropout_prob
            drop_ids = drop_ids.cuda()
            dist.broadcast(
                drop_ids,
                0,  # Change to correct rank if needed
                group=None  # Change to correct group if needed
            )
            drop_ids = drop_ids.to(labels.device)
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#############################################################################
#                               Core DiT Model                              #
#############################################################################


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int], qk_norm: bool):
        """
        Initialize the Attention module.

        Args:
            dim (int): Number of input dimensions.
            n_heads (int): Number of heads.
            n_kv_heads (Optional[int]): Number of kv heads, if using GQA.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(
            dim, n_heads * self.head_dim, bias=False,
        )
        self.wk = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=False,
        )
        self.wv = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=False,
        )
        self.wo = nn.Linear(
            n_heads * self.head_dim, dim, bias=False,
        )

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.n_local_heads * self.head_dim)
            self.k_norm = nn.LayerNorm(self.n_local_kv_heads * self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    @staticmethod
    def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
        """
        Reshape frequency tensor for broadcasting it with another tensor.

        This function reshapes the frequency tensor to have the same shape as
        the target tensor 'x' for the purpose of broadcasting the frequency
        tensor during element-wise operations.

        Args:
            freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
            x (torch.Tensor): Target tensor for broadcasting compatibility.

        Returns:
            torch.Tensor: Reshaped frequency tensor.

        Raises:
            AssertionError: If the frequency tensor doesn't match the expected
                shape.
            AssertionError: If the target tensor 'x' doesn't have the expected
                number of dimensions.
        """
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1
                 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    @staticmethod
    def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency
        tensor.

        This function applies rotary embeddings to the given query 'xq' and
        key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
        input tensors are reshaped as complex numbers, and the frequency tensor
        is reshaped for broadcasting compatibility. The resulting tensors
        contain rotary embeddings and are returned as real tensors.

        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
                exponentials.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
                and key tensor with rotary embeddings.
        """
        with torch.cuda.amp.autocast(enabled=False):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = Attention.reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        dtype = xq.dtype

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = Attention.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        xq, xk = xq.to(dtype), xk.to(dtype)

        if dtype in [torch.float16, torch.bfloat16]:
            output = flash_attn_func(xq, xk, xv, dropout_p=0., causal=False)
        else:
            n_rep = self.n_local_heads // self.n_local_kv_heads
            if n_rep >= 1:
                xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
                xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            output = F.scaled_dot_product_attention(
                xq.permute(0, 2, 1, 3),
                xk.permute(0, 2, 1, 3),
                xv.permute(0, 2, 1, 3),
                dropout_p=0., is_causal=False,
            ).permute(0, 2, 1, 3)
        output = output.flatten(-2)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple
                of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden
                dimension. Defaults to None.

        Attributes:
            w1 (nn.Linear): Linear transformation for the first
                layer.
            w2 (nn.Linear): Linear transformation for the second layer.
            w3 (nn.Linear): Linear transformation for the third
                layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * (
            (hidden_dim + multiple_of - 1) // multiple_of
        )

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False,
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False,
        )

    @torch.compile
    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
                 multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
                 qk_norm: bool) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(dim, 1024), 6 * dim, bias=True,
            ),
        )

        # xuan: add lazy learning
        self.lazy_learning_activation_func = nn.Sigmoid()
        self.timestep_map = None
        self.hidden_size = dim
        self.attn_lazy_learning = None
        self.cache_attn = None
        self.mlp_lazy_learning = None
        self.cache_mlp = None

        # we can revise it for better performance
        self.attn_threshold = 0.5
        self.mlp_threshold = 0.5

    def create_lazy_learning(self, num_sampling_steps, timestep_map):
        self.timestep_map = {timestep: i for i, timestep in enumerate(timestep_map)}
        self.attn_lazy_learning = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, 1, bias=False) for _ in range(num_sampling_steps)]
        )
        self.mlp_lazy_learning = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, 1, bias=False) for _ in range(num_sampling_steps)]
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor],
        sample_cache=False,
        lazy=False, accelerate=False,
        timestep=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(adaln_input).chunk(6, dim=1)

        if not lazy and not accelerate:
            temp = self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                freqs_cis,
            )
            if sample_cache:
                self.cache_attn = temp
            x = x + gate_msa.unsqueeze(1) * temp

            temp = self.feed_forward(
                modulate(self.ffn_norm(x), shift_mlp, scale_mlp),
            )
            if sample_cache:
                self.cache_mlp = temp
            x = x + gate_mlp.unsqueeze(1) * temp
            return x

        # accelerate
        if accelerate:
            count_attn = 0
            count_mlp = 0

            temp = modulate(self.attention_norm(x), shift_msa, scale_msa)
            if self.cache_attn is None:
                temp = self.attention(temp, freqs_cis)
                x = x + gate_msa.unsqueeze(1) * temp
                self.cache_attn = temp
            else:
                score = self.lazy_learning_activation_func(torch.sum(
                    self.attn_lazy_learning[self.timestep_map[timestep]](temp), dim=(-1, -2)
                ))
                mask_attn = STE.apply(score, self.attn_threshold).to(torch.bool)
                count_attn += torch.sum(mask_attn == 0).item()
                if torch.all(mask_attn):  # xuan: all need to stay, can not jump
                    temp = self.attention(temp, freqs_cis)
                    x = x + gate_msa.unsqueeze(1) * temp
                    self.cache_attn = temp
                else:
                    if torch.all(mask_attn == 0):       # jump all attn
                        x = x + gate_msa.unsqueeze(1) * self.cache_attn
                    else:
                        final = torch.zeros_like(temp)
                        final[mask_attn] = self.attention(temp[mask_attn], freqs_cis)
                        final[~mask_attn] = self.cache_attn[~mask_attn]
                        x = x + gate_msa.unsqueeze(1) * final
                        self.cache_attn = final

            temp = modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
            if self.cache_mlp is None:
                temp = self.feed_forward(temp)
                x = x + gate_mlp.unsqueeze(1) * temp
                self.cache_mlp = temp
            else:
                score = self.lazy_learning_activation_func(torch.sum(
                    self.mlp_lazy_learning[self.timestep_map[timestep]](temp), dim=(-1, -2)
                ))
                mask_mlp = STE.apply(score, self.mlp_threshold).to(torch.bool)
                count_mlp += torch.sum(mask_mlp == 0).item()
                if torch.all(mask_mlp):  # xuan: all need to stay, can not jump
                    temp = self.feed_forward(temp)
                    x = x + gate_mlp.unsqueeze(1) * temp
                    self.cache_mlp = temp
                else:
                    if torch.all(mask_mlp == 0):  # jump all mlp
                        x = x + gate_mlp.unsqueeze(1) * self.cache_mlp
                    else:
                        final = torch.zeros_like(temp)
                        final[mask_mlp] = self.feed_forward(temp[mask_mlp])
                        final[~mask_mlp] = self.cache_mlp[~mask_mlp]
                        x = x + gate_mlp.unsqueeze(1) * final
                        self.cache_mlp = final

            return x, count_attn, count_mlp

        # training
        temp = modulate(self.attention_norm(x), shift_msa, scale_msa)
        if self.cache_attn is not None:
            score = self.lazy_learning_activation_func(torch.sum(
                self.attn_lazy_learning[self.timestep_map[timestep]](temp), dim=(-1, -2)
            ))
            mask_attn = score
            mask = mask_attn.unsqueeze(-1).unsqueeze(-1)

            temp = self.attention(temp, freqs_cis) * mask + self.cache_attn * (torch.ones_like(mask) - mask)
            x = x + gate_msa.unsqueeze(1) * temp
            self.cache_attn = temp
        else:
            temp = self.attention(temp, freqs_cis)
            x = x + gate_msa.unsqueeze(1) * temp
            self.cache_attn = temp
            mask_attn = torch.ones(x.shape[0], device=x.device)

        temp = modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
        if self.cache_mlp is not None:
            score = self.lazy_learning_activation_func(torch.sum(
                self.mlp_lazy_learning[self.timestep_map[timestep]](temp), dim=(-1, -2)
            ))
            mask_mlp = score
            mask = mask_mlp.unsqueeze(-1).unsqueeze(-1)

            temp = self.feed_forward(temp) * mask + self.cache_mlp * (torch.ones_like(mask) - mask)
            x = x + gate_mlp.unsqueeze(1) * temp
            self.cache_mlp = temp
        else:
            temp = self.feed_forward(temp)
            x = x + gate_mlp.unsqueeze(1) * temp
            self.cache_mlp = temp
            mask_mlp = torch.ones(x.shape[0], device=x.device)

        return x, mask_attn, mask_mlp

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                min(hidden_size, 1024), 2 * hidden_size, bias=True,
            ),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_Llama(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = patch_size

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
        )
        nn.init.constant_(self.x_embedder.bias, 0.)

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.y_embedder = LabelEmbedder(num_classes, min(dim, 1024),
                                        class_dropout_prob)

        self.layers = nn.ModuleList([
            TransformerBlock(layer_id, dim, n_heads, n_kv_heads, multiple_of,
                             ffn_dim_multiplier, norm_eps, qk_norm)
            for layer_id in range(n_layers)
        ])
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        self.freqs_cis = DiT_Llama.precompute_freqs_cis(dim // n_heads, 4096)

        # xuan:
        self.sample_cache = False
        self.lazy = False
        self.mask_attn_total_time = []
        self.mask_mlp_total_time = []
        self.accelerate = False
        self.jump_attn = 0
        self.jump_mlp = 0

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        assert (H, W) == (self.input_size, self.input_size)
        pH = pW = self.patch_size
        x = x.view(B, C, H // pH, pH, W // pW, pW)
        x = x.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)
        return x

    def forward(self, x, t, y, label_for_dropout=None):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent
           representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        self.freqs_cis = self.freqs_cis.to(x.device)

        mask_attn_total = []; mask_mlp_total = []
        timestep = t[0].item()

        x = self.patchify(x)
        x = self.x_embedder(x)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training, force_drop_ids=label_for_dropout)    # (N, D)
        adaln_input = t + y

        for layer in self.layers:
            if self.accelerate:
                x, count_attn, count_mlp = layer(
                    x, self.freqs_cis[:x.size(1)], adaln_input=adaln_input,
                    accelerate=True, timestep=timestep
                )
                self.jump_attn += count_attn
                self.jump_mlp += count_mlp
            elif not self.lazy:
                x = layer(
                    x, self.freqs_cis[:x.size(1)], adaln_input=adaln_input,
                    sample_cache=self.sample_cache
                )
            elif self.lazy:
                x, mask_attn, mask_mlp = layer(
                    x, self.freqs_cis[:x.size(1)], adaln_input=adaln_input,
                    lazy=True, timestep=timestep
                )
                mask_attn_total.append(mask_attn)
                mask_mlp_total.append(mask_mlp)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x)         # (N, out_channels, H, W)

        if self.lazy:
            self.mask_attn_total_time.append(torch.stack(mask_attn_total))
            self.mask_mlp_total_time.append(torch.stack(mask_mlp_total))

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """
        freqs = 1.0 / (theta ** (
            torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
        ))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def parameter_count(self) -> int:
        total_params = 0

        def _recursive_count_params(module):
            nonlocal total_params
            for param in module.parameters(recurse=False):
                total_params += param.numel()
            for submodule in module.children():
                _recursive_count_params(submodule)

        _recursive_count_params(self)
        return total_params

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


DiT_models = {}


def export(func):
    assert func.__name__ not in DiT_models, (
        f"Model with name 'func.__name__' is exported twice."
    )
    DiT_models[func.__name__] = func
    return func


#############################################################################
#                                 DiT Configs                               #
#############################################################################


@export
def DiT_Llama_600M_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=1536, n_layers=16, n_heads=32, **kwargs
    )


@export
def DiT_Llama_3B_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=3072, n_layers=32, n_heads=32, **kwargs
    )


@export
def DiT_Llama_7B_patch2(**kwargs):
    return DiT_Llama(
        patch_size=2, dim=4096, n_layers=32, n_heads=32, **kwargs
    )
