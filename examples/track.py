"""Tracks one object through a video from a click with the four packages, on macOS (coremltools, no PyTorch).

    python examples/track.py --models <packages folder> --video <folder of JPEGs | video file> --point X,Y \
        [--point X,Y ...] [--negative X,Y ...] [--out masks] [--units ane|gpu|all|cpu]

X, Y are pixels of the first frame; --point and --negative repeat. One point picks the best of three nested masks,
several points give one mask. Writes masks/00000.png… (the object in white, at the video's size) and an overlay
JPEG per frame. A video file needs ffmpeg. This is the whole tracking loop, in the order the packages expect:

1. Every frame: the image encoder on the frame resized to 1024².
2. The first frame: the initializer with the click → the mask, its object pointer; the memory encoder on that mask
   (binarized, as SAM 2 does for a prompted frame) → the seed memory.
3. Every later frame: the propagator with the memory bank → the mask, and the frame's own memory and pointer, which
   join the bank. The bank holds the seed in spatial slot 6 (temporal row 6) and the six previous frames in slots
   0–5 (row k = k + 1 frames back), the seed's pointer first and the 15 previous frames' pointers after it.
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

NAMES = ["EdgeTAMVideoImageEncoder", "EdgeTAMVideoInitializer", "EdgeTAMVideoMemoryEncoder", "EdgeTAMVideoPropagator"]


def frames(video):
    """The video's frames as PIL images, in order."""
    if os.path.isdir(video):
        names = sorted(f for f in os.listdir(video) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for name in names:
            yield Image.open(os.path.join(video, name)).convert("RGB")
        return
    with tempfile.TemporaryDirectory() as folder:
        subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-q:v", "2", os.path.join(folder, "%05d.jpg")],
                       check=True)
        yield from frames(folder)


def point(text, size, label):
    x, y = (float(v) for v in text.split(","))
    return [x * 1024 / size[0], y * 1024 / size[1]], label


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--point", action="append", required=True, help="X,Y of the object on the first frame")
    parser.add_argument("--negative", action="append", default=[], help="X,Y of something that is not the object")
    parser.add_argument("--out", default="masks")
    parser.add_argument("--units", default="ane", choices=["ane", "gpu", "all", "cpu"])
    args = parser.parse_args()

    import coremltools as ct

    unit = {"ane": ct.ComputeUnit.CPU_AND_NE, "gpu": ct.ComputeUnit.CPU_AND_GPU, "all": ct.ComputeUnit.ALL,
            "cpu": ct.ComputeUnit.CPU_ONLY}[args.units]
    # The initializer is fp32, which the Neural Engine does not run: with --units ane it goes to the GPU.
    units = [unit, ct.ComputeUnit.ALL if args.units == "ane" else unit, unit, unit]
    encoder, initializer, memory_encoder, propagator = (
        ct.models.MLModel(os.path.join(args.models, n + ".mlpackage"), compute_units=u) for n, u in zip(NAMES, units))
    os.makedirs(args.out, exist_ok=True)

    seed, recent, temporal = None, [], None
    for i, frame in enumerate(frames(args.video)):
        features = encoder.predict({"image": frame.resize((1024, 1024), Image.BILINEAR)})
        if i == 0:
            prompts = [point(p, frame.size, 1) for p in args.point] + [point(p, frame.size, 0) for p in args.negative]
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
            temporal = memory["temporal_positions"].astype(np.float32)
            seed = (memory["memory_features"], memory["memory_positions"], out["object_pointer"])
        else:
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
                "pointer_bank": pointers.astype(np.float16), "attention_bias": np.zeros((1, 1, 1, 3648), np.float16)})
            recent = (recent + [(out["memory_features"], out["memory_positions"], out["object_pointer"])])[-15:]
        logits = Image.fromarray(out["high_res_mask"][0, 0].astype(np.float32), "F").resize(frame.size, Image.BILINEAR)
        mask = Image.fromarray(((np.asarray(logits) > 0) * 255).astype(np.uint8))
        mask.save(os.path.join(args.out, f"{i:05d}.png"))
        tint = Image.new("RGB", frame.size, (255, 64, 64))
        Image.composite(Image.blend(frame, tint, 0.5), frame, mask).save(
            os.path.join(args.out, f"overlay-{i:05d}.jpg"), quality=90)
        print(f"frame {i}: object score {float(out['object_score'].ravel()[0]):.1f}, "
              f"{int((np.asarray(mask) > 0).sum())} px", flush=True)


if __name__ == "__main__":
    main()
