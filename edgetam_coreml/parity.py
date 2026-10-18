"""Parity of the converted packages with EdgeTAM's PyTorch video predictor.

    python -m edgetam_coreml.parity --edgetam <EdgeTAM checkout> --models <packages> --work <scratch folder> \
        [--attend-empty-slots] [--frames 40] [--units gpu|ane|all|cpu] [--json <out.json>]

The sequences are the video in EdgeTAM's own notebooks/videos/bedroom, each seeded by a click (PyTorch's mask for
it): three objects, one of them from frame 120, and a gap sequence. Both sides see the same 1024² JPEGs and the same
seed mask:

- PyTorch: SAM2VideoPredictor, add_new_mask on the first frame, propagate_in_video.
- Core ML: the four packages, with the memory bank assembled here the way SAM2Base assembles it (seed in slot 6 at
  temporal row 6, the six previous frames in slots 0–5 at rows 0–5, pointers: the seed's, then the 15 previous
  frames'), carrying their own state from frame to frame (closed loop, no PyTorch tensors fed back).

Reported per sequence: mask IoU per frame (logits > 0 at 1024²), the cosine of the image features, memory features
and object pointers with PyTorch's, and the object scores. The gap sequence replaces frames with black ones in the
middle, so the object leaves and comes back (no-object masking, no-object pointer, recovery). On the seed frame the
click paths are checked too: one positive click (best of three), every point of the sequence (positive and negative),
and every point with the first click's mask (points added to a frame that has a mask).

The packages attend the memory bank's empty slots unless built with `convert --mask-empty-slots`; PyTorch has no
empty slots (it concatenates only the memories it has). `--attend-empty-slots` pads PyTorch's bank the packages' way
(zero memories and pointers up to 7 spatial slots and 16 pointers), so it is the reference for the default packages;
without it the reference is stock EdgeTAM, for packages built with `--mask-empty-slots`.

Done when every frame's IoU is at least 0.95, every cosine at least 0.99 and every click picks PyTorch's candidate.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import upstream

# (name, first frame, black frames: (from, count) or None, points (x, y, label) in the video's pixels)
BEDROOM = [
    ("bedroom-girl", 0, None, [(395, 210, 1), (230, 250, 0)]),
    ("bedroom-boy", 0, None, [(230, 250, 1), (395, 210, 0)]),
    ("bedroom-pillow", 0, None, [(600, 350, 1)]),
    ("bedroom-boy-late", 120, None, [(230, 250, 1)]),
    ("bedroom-girl-gap", 0, (16, 8), [(395, 210, 1), (230, 250, 0)]),
]
NAMES = ["EdgeTAMVideoImageEncoder", "EdgeTAMVideoInitializer", "EdgeTAMVideoMemoryEncoder", "EdgeTAMVideoPropagator"]


def cosine(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    na, nb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if na == 0 and nb == 0:
        return 1.0
    return float((a * b).sum() / max(na * nb, 1e-12))


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def video_frames(source, start, count, gap, folder):
    """`count` frames of a JPEG folder from `start`, resized to 1024² JPEGs 00000.jpg…, black where `gap` says.
    Returns the names and the source size (width, height)."""
    os.makedirs(folder, exist_ok=True)
    for old in os.listdir(folder):
        if old.endswith(".jpg"):
            os.remove(os.path.join(folder, old))
    names = sorted(f for f in os.listdir(source) if f.lower().endswith((".jpg", ".jpeg")))[start:start + count]
    size = None
    for i, name in enumerate(names):
        image = Image.open(os.path.join(source, name)).convert("RGB")
        size = image.size
        if gap and gap[0] <= i < gap[0] + gap[1]:
            image = Image.new("RGB", size)
        image.resize((1024, 1024), Image.BILINEAR).save(os.path.join(folder, f"{i:05d}.jpg"), quality=95)
    return sorted(f for f in os.listdir(folder) if f.endswith(".jpg")), size


def click_seed(predictor, folder, points):
    """The seed mask from a click: PyTorch's mask for `points` (1024² pixels) on the first frame, logits > 0."""
    state = predictor.init_state(video_path=folder)
    _, _, logits = predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=1, points=np.array([p[:2] for p in points], np.float32),
        labels=np.array([p[2] for p in points], np.int32))
    return (logits[0:1] > 0).float()


def attend_empty_slots(predictor):
    """Pads the memory attention's bank the packages' way: zero memories (and zero positions) up to 7 spatial slots
    of 512 tokens, zero pointer tokens up to 16 pointers of 4 tokens."""
    attention = predictor.memory_attention
    forward = attention.forward

    def padded(curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0, num_spatial_mem=-1):
        spatial = memory.shape[0] - num_obj_ptr_tokens
        assert spatial % 512 == 0 and spatial // 512 == num_spatial_mem <= 7 and num_obj_ptr_tokens <= 64

        def pad(x):
            zeros = lambda n: x.new_zeros(n, *x.shape[1:])
            return torch.cat([x[:spatial], zeros(7 * 512 - spatial), x[spatial:], zeros(64 - num_obj_ptr_tokens)])

        return forward(curr, pad(memory), curr_pos, pad(memory_pos), num_obj_ptr_tokens=64, num_spatial_mem=7)

    attention.forward = padded


class CoreML:
    def __init__(self, folder, units):
        import coremltools as ct

        unit = {"gpu": ct.ComputeUnit.CPU_AND_GPU, "ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL,
                "cpu": ct.ComputeUnit.CPU_ONLY}[units]
        self.models, self.load_s = {}, {}
        for name in NAMES:
            start = time.time()
            self.models[name] = ct.models.MLModel(os.path.join(folder, name + ".mlpackage"), compute_units=unit)
            self.load_s[name] = time.time() - start
        self.time_s = {n: 0.0 for n in NAMES}
        self.calls = {n: 0 for n in NAMES}

    def run(self, name, inputs):
        start = time.time()
        out = self.models[name].predict(
            {k: (v if isinstance(v, Image.Image) else np.asarray(v, np.int32 if k == "point_labels" else np.float16))
             for k, v in inputs.items()})
        self.time_s[name] += time.time() - start
        self.calls[name] += 1
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


def track_coreml(cm, jpegs, folder, seed):
    """The closed-loop Core ML run: per frame its mask logits (1024²), object score, pointer and memory."""
    temporal = None
    seed_entry, recent, results = None, [], []
    for i, name in enumerate(jpegs):
        image = Image.open(os.path.join(folder, name)).convert("RGB")
        enc = cm.run(NAMES[0], {"image": image})
        if i == 0:
            init = cm.run(NAMES[1], {
                "initial_vision_features": enc["initial_vision_features"],
                "high_res_feature_0": enc["high_res_feature_0"], "high_res_feature_1": enc["high_res_feature_1"],
                "point_coords": np.zeros((1, 1, 2)), "point_labels": -np.ones((1, 1)),
                "mask_input": seed.numpy()})
            # As propagate_in_video_preflight: memory from the low-res mask upsampled, binarized (a prompted frame).
            up = F.interpolate(torch.tensor(init["low_res_mask"]), size=(1024, 1024), mode="bilinear",
                               align_corners=False)
            mem = cm.run(NAMES[2], {"raw_vision_features": enc["raw_vision_features"],
                                    "high_res_mask": torch.where(up > 0, 30.0, -30.0).numpy(),
                                    "object_score": init["object_score"]})
            temporal = mem["temporal_positions"]
            seed_entry = (mem["memory_features"], mem["memory_positions"], init["object_pointer"])
            results.append(dict(enc=enc, logits=init["high_res_mask"], score=float(init["object_score"].ravel()[0]),
                                pointer=init["object_pointer"], memory=mem["memory_features"]))
            continue
        spatial = np.zeros((1, 7, 512, 64), np.float32)
        positions = np.zeros((1, 7, 512, 64), np.float32)
        pointers = np.zeros((1, 16, 256), np.float32)

        def place(entry, slot, row):
            spatial[0, slot] = entry[0][0]
            positions[0, slot] = entry[1][0] + temporal[row]

        def point(pointer, slot):
            pointers[0, slot] = pointer[0]

        place(seed_entry, 6, 6)
        point(seed_entry[2], 0)
        for back, entry in enumerate(reversed(recent[-6:])):  # back 0 = the previous frame
            place(entry, back, back)
        for k, entry in enumerate(reversed(recent[-15:])):
            point(entry[2], k + 1)
        out = cm.run(NAMES[3], {
            "raw_vision_features": enc["raw_vision_features"], "high_res_feature_0": enc["high_res_feature_0"],
            "high_res_feature_1": enc["high_res_feature_1"], "spatial_bank": spatial,
            "spatial_positions": positions, "pointer_bank": pointers, "attention_bias": np.zeros((1, 1, 1, 3648))})
        recent.append((out["memory_features"], out["memory_positions"], out["object_pointer"]))
        results.append(dict(enc=enc, logits=out["high_res_mask"], score=float(out["object_score"].ravel()[0]),
                            pointer=out["object_pointer"], memory=out["memory_features"],
                            candidates=out["candidate_masks"], candidate_ious=out["candidate_ious"]))
    return results


def track_torch(predictor, folder, seed):
    state = predictor.init_state(video_path=folder)
    predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=seed[0, 0].bool())
    logits = {}
    start = time.time()
    for f, _, masks in predictor.propagate_in_video(state):
        logits[f] = masks[0, 0].float().numpy()
    elapsed = time.time() - start
    out = state["output_dict_per_obj"][0]
    results = []
    for f in range(len(logits)):
        o = out["cond_frame_outputs"].get(f) or out["non_cond_frame_outputs"][f]
        image = state["images"][f].float()[None]
        with torch.no_grad():
            fpn = predictor.forward_image(image)["backbone_fpn"]
        results.append(dict(
            enc={"raw_vision_features": fpn[2].numpy(),
                 "initial_vision_features": (fpn[2] + predictor.no_mem_embed.reshape(1, -1, 1, 1)).numpy(),
                 "high_res_feature_0": fpn[0].numpy(), "high_res_feature_1": fpn[1].numpy()},
            logits=logits[f], score=float(o["object_score_logits"].ravel()[0]), pointer=o["obj_ptr"].numpy(),
            memory=o["maskmem_features"].float().numpy()))
    return results, elapsed


def clicks(predictor, cm, enc, torch_enc, points, allow_ambiguous=False):
    """Initializer click paths against SAM2Base._forward_sam_heads on the seed frame (add_new_points_or_box's path),
    each side on its own encoder's features. The chosen candidate (`chosen_index`, ours and PyTorch's), its mask
    and its pointer must match. Only with `allow_ambiguous` (an fp16 initializer) is a click whose two best
    PyTorch IoUs are within 0.02 reported instead of required: the fp16 two-way transformer flips such choices. The
    three candidates are compared one by one with the initializer on PyTorch's features (the package itself) and,
    reported only, on its own (the losing candidates of an ambiguous click are the most sensitive to the encoder's
    rounding)."""
    cases = []
    positive = [p for p in points if p[2] == 1][:1]
    features = torch.tensor(torch_enc["initial_vision_features"])
    high_res = [torch.tensor(torch_enc["high_res_feature_0"]), torch.tensor(torch_enc["high_res_feature_1"])]
    mask = None
    for label, chosen, with_mask in (("one click", positive, False), ("all points", points, False),
                                     ("points and mask", points, True)):
        if not chosen:
            continue
        coords = np.array([[[p[0], p[1]] for p in chosen]], np.float32)
        labels = np.array([[p[2] for p in chosen]], np.int32)
        prompt = {"point_coords": torch.tensor(coords), "point_labels": torch.tensor(labels)}
        given = {}
        mask_inputs = None
        if with_mask:
            # Points added to a frame that already has a mask: the video predictor passes that mask's 256² logits.
            given = {"mask_input": mask[None, None]}
            mask_inputs = F.interpolate(torch.tensor(mask[None, None] * 20 - 10), size=(256, 256), mode="bilinear",
                                        align_corners=False, antialias=True)
        multi = predictor._forward_sam_heads(features, point_inputs=prompt, mask_inputs=mask_inputs,
                                             high_res_features=high_res, multimask_output=True)
        ref = predictor._forward_sam_heads(features, point_inputs=prompt, mask_inputs=mask_inputs,
                                           high_res_features=high_res,
                                           multimask_output=predictor._use_multimask(True, prompt))
        if mask is None:
            # The mask for "points and mask": PyTorch's mask for the first case.
            mask = (ref[4][0, 0].numpy() > 0).astype(np.float32)
        out = cm.run(NAMES[1], {
            "initial_vision_features": enc["initial_vision_features"],
            "high_res_feature_0": enc["high_res_feature_0"], "high_res_feature_1": enc["high_res_feature_1"],
            "point_coords": coords, "point_labels": labels, **given})
        same = cm.run(NAMES[1], {
            "initial_vision_features": torch_enc["initial_vision_features"],
            "high_res_feature_0": torch_enc["high_res_feature_0"],
            "high_res_feature_1": torch_enc["high_res_feature_1"],
            "point_coords": coords, "point_labels": labels, **given})
        same_candidates = [iou(same["candidate_masks"][0, k] > 0, multi[0][0, k].numpy() > 0) for k in range(3)]
        ref_ious = np.sort(ref[2].numpy().ravel())[::-1]
        tie = len(ref_ious) > 1 and ref_ious[0] - ref_ious[1] < 0.02
        candidates = [iou(out["candidate_masks"][0, k] > 0, multi[0][0, k].numpy() > 0) for k in range(3)]
        case = dict(case=label, points=len(chosen), candidate_iou_min=round(min(same_candidates), 4),
                    candidate_iou_min_own_features=round(min(candidates), 4),
                    candidate_iou_diff_max=round(float(np.abs(out["candidate_ious"] - multi[2].numpy()).max()), 4),
                    chosen_iou=round(iou(out["high_res_mask"][0, 0] > 0, ref[4][0, 0].numpy() > 0), 4),
                    chosen_pointer_cosine=round(cosine(out["object_pointer"], ref[5].numpy()), 4),
                    ambiguous=bool(tie),
                    chosen_index=[int(np.argmax(out["candidate_ious"])), int(np.argmax(multi[2].numpy()))]
                    if len(chosen) == 1 else None,
                    score=[round(float(out["object_score"].ravel()[0]), 2),
                           round(float(ref[6].ravel()[0]), 2)])
        case["pass"] = bool(case["candidate_iou_min"] >= 0.95) and bool(
            (tie and allow_ambiguous) or (case["chosen_iou"] >= 0.95 and case["chosen_pointer_cosine"] >= 0.99))
        cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edgetam", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--work", required=True, help="scratch folder for the frames")
    parser.add_argument("--attend-empty-slots", action="store_true",
                        help="pad PyTorch's bank with empty slots as the default packages attend them")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--units", default="gpu")
    parser.add_argument("--only", default="")
    parser.add_argument("--json")
    parser.add_argument("--allow-ambiguous", action="store_true",
                        help="report, not require, the choice of a click with PyTorch's two best IoUs within 0.02")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    predictor = upstream.build_video_predictor(args.edgetam).eval()
    if args.attend_empty_slots:
        attend_empty_slots(predictor)
    cm = CoreML(args.models, args.units)
    report, passed = [], True
    for name, seed_frame, gap, points in BEDROOM:
        if args.only and name not in args.only.split(","):
            continue
        folder = os.path.join(args.work, name)
        jpegs, size = video_frames(os.path.join(args.edgetam, "notebooks", "videos", "bedroom"), seed_frame,
                                   args.frames, gap, folder)
        scale = (1024 / size[0], 1024 / size[1])
        seed = click_seed(predictor, folder, [[p[0] * scale[0], p[1] * scale[1], p[2]] for p in points])
        ref, torch_s = track_torch(predictor, folder, seed)
        start = time.time()
        ours = track_coreml(cm, jpegs, folder, seed)
        coreml_s = time.time() - start
        ious = [iou(a["logits"].reshape(1024, 1024) > 0, b["logits"] > 0) for a, b in zip(ours, ref)]
        enc_cos = [min(cosine(a["enc"][k], b["enc"][k]) for k in b["enc"]) for a, b in zip(ours, ref)]
        mem_cos = [cosine(a["memory"], b["memory"]) for a, b in zip(ours[1:], ref[1:])]
        ptr_cos = [cosine(a["pointer"], b["pointer"]) for a, b in zip(ours, ref)]
        absent = [i for i, b in enumerate(ref) if b["score"] <= 0]
        # Points in the video's pixels → 1024².
        scale = (1024 / size[0], 1024 / size[1])
        scaled = [[p[0] * scale[0], p[1] * scale[1], p[2]] for p in points]
        click = clicks(predictor, cm, ours[0]["enc"], ref[0]["enc"], scaled, args.allow_ambiguous)
        entry = dict(
            sequence=name, frames=len(ious), seed_frame=seed_frame, gap=gap,
            mask_iou_min=round(min(ious), 4), mask_iou_mean=round(float(np.mean(ious)), 4),
            encoder_cosine_min=round(min(enc_cos), 5), memory_cosine_min=round(min(mem_cos), 5),
            pointer_cosine_min=round(min(ptr_cos), 5),
            score_max_abs_diff=round(max(abs(a["score"] - b["score"]) for a, b in zip(ours, ref)), 3),
            absent_frames_torch=absent, absent_frames_coreml=[i for i, a in enumerate(ours) if a["score"] <= 0],
            clicks=click, iou_by_frame=[round(x, 4) for x in ious], torch_s=round(torch_s, 1),
            coreml_s=round(coreml_s, 1))
        ok = (entry["mask_iou_min"] >= 0.95 and min(enc_cos + mem_cos + ptr_cos) >= 0.99
              and all(c["pass"] for c in click))
        entry["pass"] = bool(ok)
        passed &= ok
        report.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "iou_by_frame"}), flush=True)
    timing = {n: dict(load_s=round(cm.load_s[n], 2), mean_ms=round(1000 * cm.time_s[n] / max(1, cm.calls[n]), 1),
                      calls=cm.calls[n]) for n in NAMES}
    print(json.dumps({"units": args.units, "timing": timing, "pass": passed}), flush=True)
    if args.json:
        json.dump({"sequences": report, "timing": timing, "units": args.units, "pass": passed},
                  open(args.json, "w"), indent=1)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
