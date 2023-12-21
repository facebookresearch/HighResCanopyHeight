# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torchvision
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple
import math
import warnings
from collections import OrderedDict
from torch import Tensor

import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Union
from functools import partial
import pdb

class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=np.bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
                  
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU(),
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
    
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU(),
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor) -> Tensor:
        #pdb.set_trace()
        def attn_residual_func(x: Tensor) -> Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path1(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(tuple) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class DinoVisionTransformer(nn.Module):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
               
    def __init__(
        self,
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=0,
        global_pool="token",
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        mlp_ratio=4.0, 
        qkv_bias=True, 
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        init_values=1.,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        block_fn=Block,
        ffn_layer="mlp",
        drop_path_uniform=False,
        patch_drop=0.0,
        sin_cos_embeddings=False,
        local_crops_size=96,
        multiple_pos_embeddings=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ("", "avg", "token")
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False
        self.sin_cos_embeddings = sin_cos_embeddings
        self.multiple_pos_embeddings = multiple_pos_embeddings

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.sin_cos_embeddings:
            self.pos_embed = torch.Tensor(())
            logger.info("using sin-cos fixed embeddings")
            pass
        elif self.multiple_pos_embeddings:
            logger.info("using multiple position embeddings (one for global one for local)")
            self.pos_embeds = nn.ParameterDict()
            self.pos_embeds[str(img_size)] = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            n_local_patches = (local_crops_size // patch_size) ** 2
            self.pos_embeds[str(local_crops_size)] = nn.Parameter(torch.zeros(1, n_local_patches, embed_dim))
            self.pos_embed = None
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            #print("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglu":
            #print("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFN
        elif ffn_layer == "identity":
            #print("using Identity layer as FFN")
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError

        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=ffn_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        use_fc_norm = self.global_pool == "avg"
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

        self.mask_generator = MaskingGenerator(
            input_size=(img_size // patch_size, img_size // patch_size),
            max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # if weight_init != "skip":
        # self.init_weights(weight_init)

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(
                OrderedDict([("fc", nn.Linear(self.embed_dim, self.representation_size)), ("act", nn.Tanh())])
            )
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "moco", "")
        head_bias = -math.log(self.num_classes) if "nlhb" in mode else 0.0
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        elif self.pos_embeds:
            for v in self.pos_embeds.values():
                trunc_normal_(v, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token")
            self.global_pool = global_pool
        if representation_size is not None:
            self._reset_representation(representation_size)
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == "avg" else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def interpolate_pos_encoding(self, x, w, h):
        if self.sin_cos_embeddings:
            
            w0 = w // self.patch_embed.patch_size[0]
            step_coef = (w0-1) / 3.14
            omega_coef = 10000
            sin_cos_embed = get_2d_sincos_pos_embed_cached_device(
                embed_dim=x.shape[-1], grid_size=w0, step_coef=step_coef, omega_coef=omega_coef, device=x.device, cls_token=True
            )
            
            return sin_cos_embed
        elif self.multiple_pos_embeddings:
            
            _m = sum((v.mean() * 0 for v in self.pos_embeds.values()))
            pos_embed = self.pos_embeds[str(w)] + _m
            class_pos_embed = torch.zeros_like(pos_embed[:1,:1])
            return torch.cat((class_pos_embed, pos_embed), dim=1)
        else:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1

            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",  align_corners=True,  recompute_scale_factor=True
            )

            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def mask_patches_with_probability_p(self, x, mask_ratio_tuple, p):
        B, N, _ = x.shape
        n_samples_masked = int(B * p)
        mask_ratio_min, mask_ratio_max = mask_ratio_tuple
        masks = torch.stack(
            [
                torch.BoolTensor(self.mask_generator(int(N * random.uniform(mask_ratio_min, mask_ratio_max))))
                for _ in range(0, n_samples_masked)
            ]
            + [torch.BoolTensor(self.mask_generator(0)) for _ in range(n_samples_masked, B)]
        ).to(
            x.device
        )  
        masks = masks[torch.randperm(B, device=x.device)].flatten(1)
        x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        
        return x, masks

    def mask_patches_with_probability_p_upperbound(self, x, mask_ratio_tuple, p):
        B, N, _ = x.shape
        n_samples_masked = int(B * p)
        probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
        upperbound = 0
        masks_list = []
        for i in range(0, n_samples_masked):
            prob_min = probs[i]
            prob_max = probs[i+1]
            masks_list.append(torch.BoolTensor(self.mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)
        for i in range(n_samples_masked, B):
            masks_list.append(torch.BoolTensor(self.mask_generator(0)))
        masks = torch.stack(masks_list).to(x.device)
        masks = masks[torch.randperm(B, device=x.device)].flatten(1)
        x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        
        return x, masks, upperbound

    def prepare_tokens(self, x, mask_ratio_tuple=(0.0, 0.0), mask_sample_probability=0.0, ibot_balanced_masking=False):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        masks = None
        n_masked_patches_upperbound = None
        cls_token = self.cls_token
        do_ibot = max(mask_ratio_tuple) > 0.0 and mask_sample_probability > 0.0
        if do_ibot:
            if ibot_balanced_masking:
                logger.debug("using balanced masking")
                x, masks, n_masked_patches_upperbound = self.mask_patches_with_probability_p_upperbound(
                x, mask_ratio_tuple=mask_ratio_tuple, p=mask_sample_probability
            )
            else:
                logger.debug("not using balanced masking")
                x, masks = self.mask_patches_with_probability_p(
                    x, mask_ratio_tuple=mask_ratio_tuple, p=mask_sample_probability
                )
        else:
            cls_token = cls_token + 0 * self.mask_token  # hack to use the mask_token param to not crash ddp...

        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.interpolate_pos_encoding(x, w, h))

        return x, masks, n_masked_patches_upperbound

    def forward_features(self, x, mask_ratio_tuple=(0.0, 0.0), mask_sample_probability=0.0, ibot_balanced_masking=False):
        x, masks, n_masked_patches_upperbound = self.prepare_tokens(x, mask_ratio_tuple, mask_sample_probability, ibot_balanced_masking)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
            "n_masked_patches_upperbound": n_masked_patches_upperbound,
        }

    def get_intermediate_layers(self, x, n=1):
        x, _, _ = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return ret["x_norm_clstoken"]



class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.
    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x
    
    

class SSLVisionTransformer(DinoVisionTransformer):
    """Vision Transformer.
    """

    def __init__(self,
                interpolate_mode='bicubic',
                init_cfg=None,
                pretrained=None,
                img_size=224, 
                patch_size=16,
                #embed_dim=1024, 
                #depth=24, 
                #num_heads=16, 
                mlp_ratio=4,
                qkv_bias=True,
                init_values=1.,
                out_indices=(4, 11, 17, 23),
                final_norm=False,
                with_cls_token=True,
                output_cls_token=True,
                frozen_stages=100,
                 *args, **kwargs):
        super(SSLVisionTransformer, self).__init__(*args, **kwargs) 
       
        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

            
        if len(self.blocks)==1:    
            self.blocks = self.blocks[0] 
        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = len(self.blocks) - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        self.interpolate_mode = interpolate_mode
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.detach = False
        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.final_norm = final_norm
        self.patch_size = self.patch_embed.patch_size
        self.adapad = AdaptivePadding(kernel_size=self.patch_size, stride=self.patch_size, padding='same')
        if pretrained:
            self.init_weights(pretrained)
        
        self._freeze_stages()

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed
    
    def init_weights(self, pretrained):
        print("init_weights", pretrained)
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            
            checkpoint = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in checkpoint:
                # timm checkpoint
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                # deit checkpoint
                state_dict = checkpoint['model']
            elif 'teacher' in checkpoint:
                # dino eval checkpoint
                state_dict = checkpoint['teacher']
            else:
                state_dict = checkpoint
            
            if len([k for k in state_dict.keys() if 'teacher.backbone.' in k]) > 0:
                state_dict = {k.replace('teacher.backbone.', ''):v for k,v in state_dict.items() if 'teacher.backbone' in k}
            if len([k for k in state_dict.keys() if 'backbone.' in k]) > 0:
                state_dict = {k.replace('backbone.', ''):v for k,v in state_dict.items()}

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print(f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')
                    h, w = (224, 224) # self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size[0], w // self.patch_size[1]),
                        (pos_size, pos_size), self.interpolate_mode)
            self.load_state_dict(state_dict)
        else:
            super(SSLVisionTransformer, self).init_weights()
            

    def forward(self, x):
        
        with torch.set_grad_enabled(not self.detach):
            _, _, old_w, old_h = x.shape
            xx = self.adapad(x)
            
            x = F.pad(x, (0, xx.shape[-1] - x.shape[-1], 0, xx.shape[-2] - x.shape[-2]))
            B, nc, w, h = x.shape

            x, _, _ = self.prepare_tokens(x)
            # we return the output tokens from the `n` last blocks
            outs = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.out_indices:
                    if self.with_cls_token:
                        out = x[:, 1:]
                    else:
                        out = x
                    B, _, C = out.shape
                    out = out.reshape(B, w // self.patch_size[0], h // self.patch_size[1],
                                    C).permute(0, 3, 1, 2).contiguous()
                    if self.output_cls_token:
                        out = [out, x[:, 0]]
                    else:
                        out = [out]
                    if self.final_norm:
                        out = [self.norm(o) for o in out]
                    if self.detach:
                        out = [o.detach() for o in out]
                    outs.append(out)
            return tuple(outs)

    def train(self, mode=True):
        super(SSLVisionTransformer, self).train(mode)
        self.detach = False
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for m in [self.patch_embed]:
                for param in m.parameters():
                    param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.mask_token.requires_grad = False

        if self.frozen_stages >= len(self.blocks) - 1:
            self.norm.eval()
            for param in self.norm.parameters():
                param.requires_grad = False
            self.detach = True

        for i, layer in enumerate(self.blocks):
            if i <= self.frozen_stages:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

                    
