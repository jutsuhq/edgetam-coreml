"""Tracks one object through a video from a click with the four packages, on macOS (coremltools, no PyTorch).

    python examples/track.py --models <packages folder> --video <folder of JPEGs | video file> --point X,Y \
        [--point X,Y ...] [--negative X,Y ...] [--out masks] [--units ane|gpu|all|cpu] [--start N] \
        [--direction forward|backward|both] [--no-overlay]

X, Y are pixels of frame N (--start, 0 by default); --point and --negative repeat. One point picks the best of three
nested masks, several points give one mask. Writes masks/00000.png… (the object in white, at the video's size) and an
overlay JPEG per frame, named by the video's own frame index; --no-overlay writes the masks alone. A video file needs
ffmpeg. This is the whole tracking loop, in the order the packages expect:

1. Every frame: the image encoder on the frame resized to 1024².
2. The start frame: the initializer with the clicks → the mask, its object pointer; the memory encoder on that mask
   (binarized, as SAM 2 does for a prompted frame) → the seed memory.
3. Every later frame in that direction: the propagator with the memory bank → the mask, and the frame's own memory
   and pointer, which join the bank. The bank holds the seed in spatial slot 6 (temporal row 6) and the six previous
   frames in slots 0–5 (row k = k + 1 frames back), the seed's pointer first and the 15 previous frames' pointers
   after it.

--direction forward walks from the start frame to the last one, backward walks from it to frame 0, and both walks
forward and then backward. Each pass starts from the same prompt frame and its own empty memory bank: the same seed
memory, propagated in each direction, as SAM 2 does for bidirectional tracking.
"""

import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image

NAMES = ["EdgeTAMVideoImageEncoder", "EdgeTAMVideoInitializer", "EdgeTAMVideoMemoryEncoder", "EdgeTAMVideoPropagator"]


def frame_paths(video, folder=None):
    """The video's frames as file paths, in order. A video file is decoded into `folder` first."""
    if os.path.isdir(video):
        names = sorted(f for f in os.listdir(video) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return [os.path.join(video, name) for name in names]
    if folder is None:
        raise ValueError("a folder is needed to decode a video file")
    subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-q:v", "2", os.path.join(folder, "%05d.jpg")],
                   check=True)
    return frame_paths(folder)


def point(text, size, label):
    x, y = (float(v) for v in text.split(","))
    return [x * 1024 / size[0], y * 1024 / size[1]], label


def passes(count, start, direction):
    """The frame indices to visit, in order: one pass per direction, each seeded by `start`."""
    if direction == "forward":
        return [range(start, count)]
    if direction == "backward":
        return [range(start, -1, -1)]
    return [range(start, count), range(start - 1, -1, -1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--point", action="append", required=True, help="X,Y of the object on the start frame")
    parser.add_argument("--negative", action="append", default=[], help="X,Y of something that is not the object")
    parser.add_argument("--out", default="masks")
    parser.add_argument("--units", default="ane", choices=["ane", "gpu", "all", "cpu"])
    parser.add_argument("--start", type=int, default=0, help="frame the prompt is placed on (0 by default)")
    parser.add_argument("--direction", default="forward", choices=["forward", "backward", "both"])
    parser.add_argument("--no-overlay", action="store_true",
                        help="write only the masks, not the tinted overlay JPEGs")
    args = parser.parse_args()

    import coremltools as ct

    unit = {"ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU, "all": ct.ComputeUnit.ALL,
            "cpu": ct.ComputeUnit.CPU_ONLY}[args.units]
    # The initializer is fp32, which the Neural Engine does not run: with --units ane it goes to the GPU.
    units = [unit, ct.ComputeUnit.ALL if args.units == "ane" else unit, unit, unit]
    encoder, initializer, memory_encoder, propagator = (
        ct.models.MLModel(os.path.join(args.models, n + ".mlpackage"), compute_units=u) for n, u in zip(NAMES, units))
    os.makedirs(args.out, exist_ok=True)

    temp = None
    try:
        if not os.path.isdir(args.video):
            temp = tempfile.mkdtemp(prefix="edgetam-track-")
        paths = frame_paths(args.video, temp)
        if not paths:
            raise SystemExit(f"no frames in {args.video}")
        start = min(max(args.start, 0), len(paths) - 1)

        seed_state = None
        for order in passes(len(paths), start, args.direction):
            recent = []
            for index in order:
                if index == start:
                    if seed_state is not None:
                        # A later pass reuses the prompt frame's seed and its own empty bank; nothing to write.
                        continue
                    frame = Image.open(paths[index]).convert("RGB")
                    features = encoder.predict({"image": frame.resize((1024, 1024), Image.BILINEAR)})
                    prompts = ([point(p, frame.size, 1) for p in args.point]
                               + [point(p, frame.size, 0) for p in args.negative])
                    out = initializer.predict({
                        "initial_vision_features": features["initial_vision_features"],
                        "high_res_feature_0": features["high_res_feature_0"],
                        "high_res_feature_1": features["high_res_feature_1"],
                        "point_coords": np.array([[c for c, _ in prompts]], np.float16),
                        "point_labels": np.array([[label for _, label in prompts]], np.int32),
                        "mask_input": np.zeros((1, 1, 1024, 1024), np.float16)})
                    binary = np.where(out["high_res_mask"] > 0, 30.0, -30.0).astype(np.float16)
                    memory = memory_encoder.predict({"raw_vision_features": features["raw_vision_features"],
                                                     "high_res_mask": binary, "object_score": out["object_score"]})
                    seed = (memory["memory_features"], memory["memory_positions"], out["object_pointer"])
                    seed_state = (out, seed, memory["temporal_positions"].astype(np.float32))
                else:
                    frame = Image.open(paths[index]).convert("RGB")
                    features = encoder.predict({"image": frame.resize((1024, 1024), Image.BILINEAR)})
                    out, seed, temporal = seed_state
                    spatial = np.zeros((1, 7, 512, 64), np.float32)
                    positions = np.zeros((1, 7, 512, 64), np.float32)
                    pointers = np.zeros((1, 16, 256), np.float32)

                    def place(entry, slot, row):
                        spatial[0, slot] = entry[0][0]
                        positions[0, slot] = entry[1][0] + temporal[row]

                    def pointer(entry, slot):
                        pointers[0, slot] = entry[2][0]

                    place(seed, 6, 6)
                    pointer(seed, 0)
                    for back, entry in enumerate(reversed(recent[-6:])):
                        place(entry, back, back)
                    for k, entry in enumerate(reversed(recent[-15:])):
                        pointer(entry, k + 1)
                    out = propagator.predict({
                        "raw_vision_features": features["raw_vision_features"],
                        "high_res_feature_0": features["high_res_feature_0"],
                        "high_res_feature_1": features["high_res_feature_1"],
                        "spatial_bank": spatial.astype(np.float16), "spatial_positions": positions.astype(np.float16),
                        "pointer_bank": pointers.astype(np.float16),
                        "attention_bias": np.zeros((1, 1, 1, 3648), np.float16)})
                    recent = (recent + [(out["memory_features"], out["memory_positions"],
                                         out["object_pointer"])])[-15:]
                logits = Image.fromarray(out["high_res_mask"][0, 0].astype(np.float32), "F").resize(
                    frame.size, Image.BILINEAR)
                mask = Image.fromarray(((np.asarray(logits) > 0) * 255).astype(np.uint8))
                mask.save(os.path.join(args.out, f"{index:05d}.png"))
                if not args.no_overlay:
                    tint = Image.new("RGB", frame.size, (255, 64, 64))
                    Image.composite(Image.blend(frame, tint, 0.5), frame, mask).save(
                        os.path.join(args.out, f"overlay-{index:05d}.jpg"), quality=90)
                print(f"frame {index}: object score {float(out['object_score'].ravel()[0]):.1f}, "
                      f"{int((np.asarray(mask) > 0).sum())} px", flush=True)
    finally:
        if temp:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    main()
