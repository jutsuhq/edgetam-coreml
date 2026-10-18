"""Converts EdgeTAM's video tracker to four Core ML packages for apps on Apple silicon.

    python -m edgetam_coreml.convert --edgetam <EdgeTAM checkout> --out <folder> [--only encoder,initializer,memory,propagator]

Writes EdgeTAMVideoImageEncoder, EdgeTAMVideoInitializer, EdgeTAMVideoMemoryEncoder and EdgeTAMVideoPropagator
(.mlpackage, ML program for macOS 15; fp16, except the initializer in fp32: in fp16 its two-way transformer
flips near-tied candidate choices on the Neural Engine, and it runs once per seed or click). Inputs and outputs keep
the names and shapes of the community conversion (AhmadYarAI/EdgeTAM-CoreML-Video) where the meaning is the same, so
code written for it runs these; what is new is added:

- Image encoder: `image` (1024², RGB) → `raw_vision_features` (the stride-16 features), `initial_vision_features`
  (the same plus the no-memory embedding, for a frame without memory), `high_res_feature_0/1` (the decoder's
  projected stride-4 and stride-8 features).
- Initializer (a conditioning frame, no memory): `point_coords` [1, N, 2] in 1024² pixels, `point_labels` [1, N]
  (1 positive, 0 negative, -1 none), N from 1 to 16, and `mask_input` [1, 1, 1024, 1024] (0/1; optional, zeros by
  default). With a mask and no labelled point it is SAM 2's mask prompt (`_use_mask_as_output`): the mask itself is
  the output (±10 logits) and the object pointer comes from the decoder prompted with the downsampled mask. With
  points, SAM 2's click path (`_forward_sam_heads`): one point picks the best of three masks by predicted IoU, more
  points take the single-mask output with the stability fallback; a mask given with points is the dense prompt, as
  the mask already on a frame is when points are added to it. Labels 2 and 3 are a box's corners.
  Outputs `low_res_mask` [1, 1, 256, 256] and `high_res_mask` [1, 1, 1024, 1024] logits, `best_iou` [1],
  `object_pointer` [1, 256], `object_score` [1, 1] logit, and the three candidates: `candidate_masks`
  [1, 3, 256, 256], `candidate_ious` [1, 3], `candidate_pointers` [1, 3, 256].
- Memory encoder: `raw_vision_features`, `high_res_mask` logits, `object_score` (at or below 0 the frame is encoded
  as empty, as the propagator does) → `memory_features` and `memory_positions` [1, 512, 64] (spatial perceiver
  latents: 256 global, then 256 on a 16² grid), `temporal_positions` [7, 64] (row k: k + 1 frames back for k < 6;
  row 6: a conditioning frame).
- Propagator (a tracked frame): `raw_vision_features`, `high_res_feature_0/1`, the bank — `spatial_bank` and
  `spatial_positions` [1, 7, 512, 64] (memory positions plus the slot's temporal row), `pointer_bank` [1, 16, 256],
  `attention_bias` [1, 1, 1, 3648] (not read; kept for the community interface) → the initializer's outputs for the
  best of the three masks plus that mask's `memory_features` and `memory_positions`. The community package's
  `rotary_weight` input is gone; rotary positions are fixed.

Shared by the initializer and the propagator: the object score masks every mask (-1024 when the object is not
there) and swaps the pointer for the learned no-object pointer, as SAM2Base does.
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import upstream

NO_OBJ_SCORE = -1024.0
VERSION = "1.1"
# Packages computed in fp32 by default (--fp32).
FP32 = "initializer"
NAMES = {
    "encoder": "EdgeTAMVideoImageEncoder",
    "initializer": "EdgeTAMVideoInitializer",
    "memory": "EdgeTAMVideoMemoryEncoder",
    "propagator": "EdgeTAMVideoPropagator",
}


def axial_rotation(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    """cos and sin of SAM 2's axial rotary encoding (`compute_axial_cis`), [end_x · end_y, dim / 2] each."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    tx, ty = t % end_x, torch.div(t, end_x, rounding_mode="floor")
    angles = torch.cat([torch.outer(tx, freqs), torch.outer(ty, freqs)], dim=-1)
    return torch.cos(angles), torch.sin(angles)


def rotate(x, cos, sin):
    """The complex rotation of `apply_rotary_enc` in real arithmetic: pairs (x[2i], x[2i+1]) of [1, N, C]."""
    b, n, c = x.shape
    pairs = x.reshape(b, n, c // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).reshape(b, n, c)


def attend(attn, q, k, v, q_rot=None, k_rot=None, bias=None):
    """One of SAM 2's single-head attentions (`Attention`, `RoPEAttention`, `RoPEAttentionv2`) with fixed rotary
    tables and an optional additive bias [1, keys], written out because the Neural Engine drops SDPA's `attn_mask`."""
    assert attn.num_heads == 1
    q, k, v = attn.q_proj(q), attn.k_proj(k), attn.v_proj(v)
    if q_rot is not None:
        q = rotate(q, *q_rot)
    if k_rot is not None:
        k = rotate(k, *k_rot)
    if bias is not None:
        logits = (q * q.shape[-1] ** -0.5) @ k.transpose(1, 2) + bias[:, None, :]
        return attn.out_proj(torch.softmax(logits, dim=-1) @ v)
    out = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])
    return attn.out_proj(out[:, 0])


class Heads(torch.nn.Module):
    """SAM 2's prompt encoder and mask decoder (`_forward_sam_heads`) without Python branches."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        pe = model.sam_prompt_encoder
        self.register_buffer("dense_pe", pe.get_dense_pe().detach().clone())
        self.register_buffer("no_mask", pe.no_mask_embed.weight.reshape(1, -1, 1, 1).detach().clone())

    def embed_points(self, coords, labels):
        """`PromptEncoder._embed_points` with the padding point, labels as floats."""
        pe = self.model.sam_prompt_encoder
        coords = torch.cat([coords + 0.5, torch.zeros_like(coords[:, :1])], dim=1) / 1024.0
        labels = torch.cat([labels, -torch.ones_like(labels[:, :1])], dim=1)[..., None]
        embedding = pe.pe_layer._pe_encoding(coords)
        embedding = torch.where(labels == -1, torch.zeros_like(embedding), embedding)
        embedding = embedding + (labels == -1).float() * pe.not_a_point_embed.weight
        for i in range(4):
            embedding = embedding + (labels == i).float() * pe.point_embeddings[i].weight
        return embedding

    def decode(self, features, sparse, dense, high_res):
        """All four masks (single, then the three of multimask), IoUs, pointer tokens and the object score logit."""
        return self.model.sam_mask_decoder.predict_masks(
            image_embeddings=features, image_pe=self.dense_pe, sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense, repeat_image=False, high_res_features=high_res)

    def pointer(self, token, present):
        """Object pointer of a decoder token, the no-object pointer where the object is not there."""
        ptr = self.model.obj_ptr_proj(token)
        return present * ptr + (1 - present) * self.model.no_obj_ptr

    def outputs(self, masks, ious, tokens, score, multimask):
        """SAM2Base's choice among the masks: the best of three by IoU when `multimask`, else the single mask unless
        unstable (`_dynamic_multimask_via_stability`), then the no-object masking. Returns low-res logits, IoU,
        pointer, and the three candidates (masks, IoUs, pointers)."""
        decoder = self.model.sam_mask_decoder
        present = (score > 0).float()  # [1, 1]
        multi_masks, multi_ious = masks[:, 1:], ious[:, 1:]
        # argmax as a one-hot of the first maximum, so the choice stays on the Neural Engine (argmax and gather
        # run on the CPU and split the graph).
        top = (multi_ious >= multi_ious.amax(dim=-1, keepdim=True)).float()  # [1, 3]
        first = torch.stack([top[:, 0], top[:, 1] * (1 - top[:, 0]), top[:, 2] * (1 - top[:, 0]) * (1 - top[:, 1])],
                            dim=1)
        best_mask = (multi_masks * first[:, :, None, None]).sum(dim=1, keepdim=True)
        best_iou = (multi_ious * first).sum(dim=1, keepdim=True)
        best_token = (tokens[:, 1:] * first[:, :, None]).sum(dim=1, keepdim=True)
        single = masks[:, 0:1]
        delta = decoder.dynamic_multimask_stability_delta
        inner = (single > delta).float().sum(dim=(2, 3))
        outer = (single > -delta).float().sum(dim=(2, 3))
        stability = inner / outer.clamp(min=1) + (outer == 0).float()
        stable = stability >= decoder.dynamic_multimask_stability_thresh
        if not decoder.dynamic_multimask_via_stability:
            stable = torch.ones_like(stable)
        single_mask = pick(stable, single, best_mask)
        single_iou = pick(stable, ious[:, 0:1], best_iou)
        low = pick(multimask, best_mask, single_mask)
        iou = pick(multimask, best_iou, single_iou)
        token = pick(multimask, best_token, tokens[:, 0:1])
        visible = score[:, :, None, None] > 0
        low = torch.where(visible, low, torch.full_like(low, NO_OBJ_SCORE))
        candidates = torch.where(visible, multi_masks, torch.full_like(multi_masks, NO_OBJ_SCORE))
        return (low, iou.reshape(1), self.pointer(token[:, 0], present), candidates, multi_ious,
                self.pointer(tokens[:, 1:], present[:, :, None]))


def pick(flag, a, b):
    """torch.where on a one-element flag, shaped to `a` (Core ML cannot broadcast a 0-d condition)."""
    return torch.where(flag.reshape([1] * a.dim()), a, b)


def upsample(low):
    return F.interpolate(low, size=(1024, 1024), mode="bilinear", align_corners=False)


class ImageEncoder(torch.nn.Module):
    """The image arrives as 0–255 (no ImageType scale or bias: Core ML runs those in fp32 on the CPU before the cast
    to fp16). The mean is subtracted in the graph, on the Neural Engine, and the scaling by 1 / (255 · std) is folded
    into the trunk's first convolution. The fold is exact: the convolution is linear and scaling leaves its zero
    padding at zero."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
        first = next(m for m in self.model.image_encoder.trunk.modules() if isinstance(m, torch.nn.Conv2d))
        assert first.in_channels == 3 and first.groups == 1
        with torch.no_grad():
            first.weight /= 255 * torch.tensor(self.STD).view(1, 3, 1, 1)
        self.register_buffer("offset", 255 * torch.tensor(self.MEAN).view(1, 3, 1, 1))

    def forward(self, image):
        fpn = self.model.forward_image(image - self.offset)["backbone_fpn"]
        raw = fpn[2]
        initial = raw + self.model.no_mem_embed.reshape(1, -1, 1, 1)
        return raw, initial, fpn[0], fpn[1]


class Initializer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.heads = Heads(model)
        # SAM 2's antialiased bilinear 4× downsampling (`_use_mask_as_output`) as a convolution: an 8-tap triangle,
        # renormalised where it overhangs the edge.
        tri = torch.tensor([1, 3, 5, 7, 7, 5, 3, 1], dtype=torch.float32) / 16
        self.register_buffer("triangle", torch.outer(tri, tri).view(1, 1, 8, 8))
        ones = torch.ones(1, 1, 1024, 1024)
        self.register_buffer("coverage", F.conv2d(ones, self.triangle, stride=4, padding=2))

    def downsample(self, high):
        return F.conv2d(high, self.triangle, stride=4, padding=2) / self.coverage

    def forward(self, initial_vision_features, high_res_feature_0, high_res_feature_1, point_coords, point_labels,
                mask_input):
        model = self.model
        labels = point_labels.float()
        has_mask = mask_input.amax(dim=(1, 2, 3)) > 0.5  # [1]
        clicks = (labels >= 0).float().sum(dim=1)
        as_output = has_mask & (clicks == 0)
        # The mask prompt decodes on the raw features, a click on the features with the no-memory embedding.
        features = initial_vision_features - as_output.float() * model.no_mem_embed.reshape(1, -1, 1, 1)
        # The mask as ±10 logits, as `_use_mask_as_output` makes them.
        given = mask_input * 20 - 10
        # The dense prompt. A mask alone: the learned `mask_downsample` (the pointer of `_use_mask_as_output`).
        # Points with a mask: the mask's 256² logits, antialiased, as the video predictor passes a frame's previous
        # mask with new points (`prev_sam_mask_logits` into `_forward_sam_heads`).
        downscale = model.sam_prompt_encoder.mask_downscaling
        dense = pick(
            has_mask, pick(as_output, downscale(model.mask_downsample(mask_input)), downscale(self.downsample(given))),
            self.heads.no_mask.expand(1, -1, 64, 64))
        sparse = self.heads.embed_points(point_coords, labels)
        masks, ious, tokens, score = self.heads.decode(
            features, sparse, dense, [high_res_feature_0, high_res_feature_1])
        # `_use_multimask`: one point or none picks the best of three, with or without a mask; a mask alone takes
        # the single mask (`_use_mask_as_output` decodes its pointer with multimask_output=False).
        multimask = (clicks <= 1) & ~as_output
        low, iou, pointer, candidates, candidate_ious, candidate_pointers = self.heads.outputs(
            masks, ious, tokens, score, multimask)
        high = upsample(low)
        # The mask prompt: the mask as logits, IoU 1, object score 10.
        low = pick(as_output, self.downsample(given), low)
        high = pick(as_output, given, high)
        iou = pick(as_output, torch.ones_like(iou), iou)
        score = pick(as_output, torch.full_like(score, 10.0), score)
        return low, high, iou, pointer, score, candidates, candidate_ious, candidate_pointers


class MemoryEncoder(torch.nn.Module):
    """`_encode_new_memory`: sigmoid of the logits, scaled to ±10, through the memory encoder and the perceiver.

    With `smooth` the package's own input first goes through SAM 2's 4× antialiased downsampling and bilinear
    upsampling, as a mask prompt's logits do in the video predictor (`_use_mask_as_output` keeps the 256² mask, which
    is upsampled for the memory); the propagator's own memory is untouched (its mask is already upsampled)."""

    def __init__(self, model, smooth=False):
        super().__init__()
        self.model = model
        self.smooth = smooth
        if smooth:
            tri = torch.tensor([1, 3, 5, 7, 7, 5, 3, 1], dtype=torch.float32) / 16
            self.register_buffer("triangle", torch.outer(tri, tri).view(1, 1, 8, 8))
            self.register_buffer("coverage", F.conv2d(torch.ones(1, 1, 1024, 1024), self.triangle, stride=4, padding=2))

    def encode(self, raw, high):
        model = self.model
        mask = torch.sigmoid(high) * model.sigmoid_scale_for_mem_enc + model.sigmoid_bias_for_mem_enc
        out = model.memory_encoder(raw, mask, skip_mask_sigmoid=True)
        return model.spatial_perceiver(out["vision_features"], out["vision_pos_enc"][0])

    def forward(self, raw_vision_features, high_res_mask, object_score):
        if self.smooth:
            high_res_mask = upsample(F.conv2d(high_res_mask, self.triangle, stride=4, padding=2) / self.coverage)
        high = torch.where(object_score[:, :, None, None] > 0, high_res_mask,
                           torch.full_like(high_res_mask, NO_OBJ_SCORE))
        features, positions = self.encode(raw_vision_features, high)
        return features, positions, self.model.maskmem_tpos_enc.reshape(7, -1)


class Propagator(torch.nn.Module):
    """`_prepare_memory_conditioned_features` on a given bank, then the tracking heads and the new memory."""

    def __init__(self, model, mask_empty_slots=False):
        super().__init__()
        self.model = model
        self.mask_empty_slots = mask_empty_slots
        self.heads = Heads(model)
        self.memory = MemoryEncoder(model)
        attention = model.memory_attention
        layer = attention.layers[0]
        assert attention.pos_enc_at_input and attention.batch_first
        assert not layer.pos_enc_at_attn and layer.pos_enc_at_cross_attn_keys
        assert not layer.pos_enc_at_cross_attn_queries
        # The stride-16 features' sine positions, for the current frame.
        dummy = model.image_encoder.neck.position_encoding(torch.zeros(1, 256, 64, 64))
        self.register_buffer("curr_pos", dummy.flatten(2).permute(0, 2, 1).detach().clone())
        cos, sin = axial_rotation(256, 64, 64)
        self.register_buffer("self_cos", cos)
        self.register_buffer("self_sin", sin)
        # Cross attention: queries on the 64² grid, keys rotated on each memory's 16² latents (the last 256 of its
        # 512); the global latents and the pointer tokens are not rotated.
        k_sizes = layer.cross_attn_image.freqs_cis_k.shape[0]
        assert k_sizes == 256
        kc, ks = axial_rotation(256, 16, 16)
        slot_cos = torch.cat([torch.ones(256, 128), kc])
        slot_sin = torch.cat([torch.zeros(256, 128), ks])
        self.register_buffer("key_cos", torch.cat([slot_cos.repeat(7, 1), torch.ones(64, 128)]))
        self.register_buffer("key_sin", torch.cat([slot_sin.repeat(7, 1), torch.zeros(64, 128)]))

    def forward(self, raw_vision_features, high_res_feature_0, high_res_feature_1, spatial_bank, spatial_positions,
                pointer_bank, attention_bias):
        model = self.model
        attention = model.memory_attention
        # Empty slots (all-zero memory or pointer) are attended by default; `mask_empty_slots` gives their keys -100,
        # enough as the logits stay within ±31.
        bias = None
        if self.mask_empty_slots:
            spatial = (spatial_bank.abs().amax(dim=(2, 3)) > 0).float()  # [1, 7]
            pointers = (pointer_bank.abs().amax(dim=2) > 0).float()  # [1, 16]
            filled = torch.cat([spatial[:, :, None].expand(1, 7, 512).reshape(1, 7 * 512),
                                pointers[:, :, None].expand(1, 16, 4).reshape(1, 64)], dim=1)
            bias = (filled - 1) * 100.0
        memory = torch.cat([spatial_bank.reshape(1, 7 * 512, 64), pointer_bank.reshape(1, 64, 64)], dim=1)
        memory_pos = torch.cat(
            [spatial_positions.reshape(1, 7 * 512, 64), torch.zeros_like(pointer_bank.reshape(1, 64, 64))], dim=1)
        keys = memory + memory_pos
        x = raw_vision_features.flatten(2).permute(0, 2, 1) + 0.1 * self.curr_pos
        self_rot = (self.self_cos, self.self_sin)
        for layer in attention.layers:
            y = layer.norm1(x)
            x = x + attend(layer.self_attn, y, y, y, self_rot, self_rot)
            y = layer.norm2(x)
            x = x + attend(layer.cross_attn_image, y, keys, memory, self_rot, (self.key_cos, self.key_sin), bias)
            y = layer.norm3(x)
            x = x + layer.linear2(layer.activation(layer.linear1(y)))
        features = attention.norm(x).permute(0, 2, 1).reshape(1, 256, 64, 64)
        sparse = self.heads.embed_points(torch.zeros(1, 1, 2), -torch.ones(1, 1))
        dense = self.heads.no_mask.expand(1, -1, 64, 64)
        masks, ious, tokens, score = self.heads.decode(
            features, sparse, dense, [high_res_feature_0, high_res_feature_1])
        multimask = torch.ones(1, dtype=torch.bool)
        low, iou, pointer, candidates, candidate_ious, candidate_pointers = self.heads.outputs(
            masks, ious, tokens, score, multimask)
        high = upsample(low)
        memory_features, memory_positions = self.memory.encode(raw_vision_features, high)
        return (low, high, iou, pointer, score, memory_features, memory_positions, candidates, candidate_ious,
                candidate_pointers)


def tensor(name, shape, dtype=np.float16, **kwargs):
    import coremltools as ct

    return ct.TensorType(name=name, shape=shape, dtype=dtype, **kwargs)


def outputs(*names):
    import coremltools as ct

    return [ct.TensorType(name=n, dtype=np.float16) for n in names]


MASK_OUTPUTS = ["low_res_mask", "high_res_mask", "best_iou", "object_pointer", "object_score"]
CANDIDATES = ["candidate_masks", "candidate_ious", "candidate_pointers"]


def example(model):
    torch.manual_seed(0)
    image = torch.rand(1, 3, 1024, 1024) * 255
    with torch.no_grad():
        raw, initial, h0, h1 = ImageEncoder(model)(image)
    return image, raw, initial, h0, h1


def specs(model, options=None):
    """Each package: module, example inputs, Core ML inputs, output names."""
    options = options or {}
    import coremltools as ct

    image, raw, initial, h0, h1 = example(model)
    features = [tensor("high_res_feature_0", (1, 32, 256, 256)), tensor("high_res_feature_1", (1, 64, 128, 128))]
    points = ct.RangeDim(1, 16, default=1)
    mask = torch.zeros(1, 1, 1024, 1024)
    mask[..., 300:700, 400:600] = 1
    return {
        "encoder": (
            ImageEncoder(model), (image,),
            [ct.ImageType(name="image", shape=(1, 3, 1024, 1024), color_layout=ct.colorlayout.RGB)],
            ["raw_vision_features", "initial_vision_features", "high_res_feature_0", "high_res_feature_1"]),
        "initializer": (
            Initializer(model),
            (initial, h0, h1, torch.tensor([[[512.0, 512.0], [100.0, 100.0]]]),
             torch.tensor([[1, 0]], dtype=torch.int32), mask),
            [tensor("initial_vision_features", (1, 256, 64, 64)), *features,
             tensor("point_coords", (1, points, 2)), tensor("point_labels", (1, points), dtype=np.int32),
             tensor("mask_input", (1, 1, 1024, 1024), default_value=np.zeros((1, 1, 1024, 1024), np.float16))],
            MASK_OUTPUTS + CANDIDATES),
        "memory": (
            MemoryEncoder(model, smooth=options.get("smooth_seed_memory", False)),
            (raw, torch.randn(1, 1, 1024, 1024) * 10, torch.ones(1, 1) * 10),
            [tensor("raw_vision_features", (1, 256, 64, 64)), tensor("high_res_mask", (1, 1, 1024, 1024)),
             tensor("object_score", (1, 1))],
            ["memory_features", "memory_positions", "temporal_positions"]),
        "propagator": (
            Propagator(model, mask_empty_slots=options.get("mask_empty_slots", False)),
            (raw, h0, h1, torch.randn(1, 7, 512, 64), torch.randn(1, 7, 512, 64), torch.randn(1, 16, 256),
             torch.zeros(1, 1, 1, 3648)),
            [tensor("raw_vision_features", (1, 256, 64, 64)), *features, tensor("spatial_bank", (1, 7, 512, 64)),
             tensor("spatial_positions", (1, 7, 512, 64)), tensor("pointer_bank", (1, 16, 256)),
             tensor("attention_bias", (1, 1, 1, 3648))],
            MASK_OUTPUTS + ["memory_features", "memory_positions"] + CANDIDATES),
    }


def check_downsample(model):
    """The convolution matches SAM 2's antialiased interpolation."""
    mask = (torch.rand(1, 1, 1024, 1024) > 0.5).float() * 20 - 10
    ours = Initializer(model).downsample(mask)
    theirs = F.interpolate(mask, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
    error = (ours - theirs).abs().max().item()
    assert error < 1e-4, f"antialiased downsampling differs by {error}"


def compress(mlmodel, mode):
    """Weight compression: `int8` (linear, per channel) or `pal<bits>` (k-means palettization, per tensor)."""
    import coremltools.optimize.coreml as cto

    if mode == "int8":
        config = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric"))
        return cto.linear_quantize_weights(mlmodel, config)
    if mode.startswith("pal"):
        config = cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=int(mode[3:])))
        return cto.palettize_weights(mlmodel, config)
    raise ValueError(f"unknown weight compression {mode}")


def convert(key, model, out, fp32=False, weights=None, options=None):
    """One package: fp16 unless `fp32`, its weights compressed by `weights` (see `compress`)."""
    import coremltools as ct

    module, args, inputs, names = specs(model, options)[key]
    module.eval()
    start = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(module, args, check_trace=False)
    mlmodel = ct.convert(
        traced, inputs=inputs, outputs=outputs(*names), convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32 if fp32 else ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU)
    if weights:
        mlmodel = compress(mlmodel, weights)
    mlmodel.author = "Meta EdgeTAM (Apache-2.0), converted to Core ML by edgetam-coreml"
    mlmodel.license = "Apache-2.0"
    precision = "fp32" if fp32 else "fp16"
    mlmodel.short_description = (f"EdgeTAM video tracker: {key} (upstream {upstream.COMMIT[:12]}, {precision}"
                                 + (f", weights {weights}" if weights else "") + ")")
    mlmodel.version = VERSION
    path = os.path.join(out, NAMES[key] + ".mlpackage")
    mlmodel.save(path)
    print(f"{NAMES[key]}: {time.time() - start:.0f} s → {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edgetam", required=True, help="checkout of facebookresearch/EdgeTAM at the pinned commit")
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="encoder,initializer,memory,propagator")
    parser.add_argument("--fp32", default=FP32, help="packages computed in fp32 (comma-separated, or none)")
    parser.add_argument("--weights", default="", help="package:mode,… weight compression (int8, pal4, pal6, pal8)")
    parser.add_argument("--smooth-seed-memory", action="store_true",
                        help="memory encoder input down- and upsampled as a mask prompt's (MemoryEncoder)")
    parser.add_argument("--mask-empty-slots", action="store_true",
                        help="mask the bank's empty (all-zero) slots, as PyTorch does (Propagator)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp32 = set(args.fp32.split(",")) - {"none", ""}
    weights = dict(w.split(":") for w in args.weights.split(",") if w)
    options = {"smooth_seed_memory": args.smooth_seed_memory, "mask_empty_slots": args.mask_empty_slots}
    model = upstream.build_video_predictor(args.edgetam).eval()
    check_downsample(model)
    for key in args.only.split(","):
        convert(key, model, args.out, fp32=key in fp32, weights=weights.get(key), options=options)


if __name__ == "__main__":
    main()
