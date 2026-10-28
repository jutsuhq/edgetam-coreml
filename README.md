# EdgeTAM for Core ML

Meta's [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) video object tracker, converted to four Core ML
packages that run on the Apple Neural Engine. Click an object once, and it is tracked through the video
at about 49 ms a frame on an M4. On Windows and Linux with an NVIDIA GPU, the same tracker runs in Rust on ONNX Runtime's
CUDA provider in [edgetam-onnx](https://github.com/jutsuhq/edgetam-onnx).

![A river surfer tracked through spray and camera motion from one click on the first frame](docs/demo.webp)

<sub>22 seconds of [Die Ewige Welle und Surfers in Augsburg](https://commons.wikimedia.org/wiki/File:Die_Ewige_Welle_und_Surfers_in_Augsburg.ogv)
(Sillerkiil, CC0), tracked on the Neural Engine with `examples/track.py` from a single click on frame 0.</sub>

- **Whole per-frame graph on the Neural Engine.** Core ML's compute plan puts every operation of the image encoder,
  the propagator and the memory encoder on the Neural Engine (`ops` in [`results/speed.jsonl`](results/speed.jsonl)),
  and a tracked frame takes about 49 ms of model time.
- **Exact conversion.** Built in fp32, the graph matches EdgeTAM's PyTorch video predictor with cosine 1.0000 on
  features, memories and object pointers, given the same memory bank. The shipped fp16 packages and the bank's empty
  slots each move the result away from stock EdgeTAM, and [Accuracy](#accuracy) measures both (mean mask IoU
  0.98–0.998).
- **Every SAM 2 prompt.** It takes 1–16 positive and negative points, a box, a mask prompt, or points with a mask. It
  returns all three candidate masks with their predicted IoUs and object pointers, so you can disambiguate a click or
  build your own memory policy.
- **Familiar interface.** Input and output names and shapes follow the community conversion
  [AhmadYarAI/EdgeTAM-CoreML-Video](https://huggingface.co/AhmadYarAI/EdgeTAM-CoreML-Video) wherever the meaning is
  the same (see [the differences](#compared-with-the-community-conversion)).

## Quick start

Prebuilt packages are on the [Releases](../../releases/latest) page: unzip them and skip to tracking below
(`examples/track.py` needs only `coremltools`, `numpy` and `Pillow`). To build them yourself, you need macOS on Apple
silicon, [uv](https://github.com/astral-sh/uv) and git. ffmpeg is needed only to read video files.

Releases from v1.2 also carry `edgetam-coreml-<version>-masked-propagator.zip`: an `EdgeTAMVideoPropagator` built
with `--mask-empty-slots`, a drop-in replacement that tracks much closer to stock EdgeTAM for about 1.6 ms more a frame (see
[Neural Engine and attention masks](#neural-engine-and-attention-masks)). It treats an all-zero slot as empty, so
leave unused slots as zeros.

```bash
./convert.sh ~/work/edgetam
```

This clones EdgeTAM at the pinned commit, checks the checkpoint's SHA-256 and sets up a Python 3.12 environment. It
then writes the packages to `~/work/edgetam/out/` and runs the parity check described below. Add `--no-parity` to
skip the check.

To track an object:

```bash
~/work/edgetam/venv/bin/python examples/track.py --models ~/work/edgetam/out \
    --video clip.mp4 --point 400,300
```

This writes `masks/00000.png…` and an overlay for each frame. `--video` is a video file or a folder of JPEGs, and
`--point` is a pixel on the first frame. Add `--negative X,Y` (repeatable) to exclude something. A single click picks
the best of three nested masks (a shirt rather than the whole person), so use more points to get the whole object.

The demo above is the surfer clip at 15 fps, cropped to 960×540, with one click:

```bash
ffmpeg -t 22 -i surf.ogv -vf fps=15,crop=960:540:640:270 -q:v 2 frames/%05d.jpg
~/work/edgetam/venv/bin/python examples/track.py --models ~/work/edgetam/out --video frames --point 632,345
```

## The packages

| Package | Runs | Size | Runs on | Median |
|---|---|---|---|---|
| `EdgeTAMVideoImageEncoder` | every frame | 10 MB | Neural Engine | 14.2 ms |
| `EdgeTAMVideoPropagator` | every tracked frame | 29 MB | Neural Engine | 35.0 ms |
| `EdgeTAMVideoInitializer` | a prompted frame (fp32) | 26 MB | GPU (`.all`) | 12.3 ms |
| `EdgeTAMVideoMemoryEncoder` | after a prompted frame | 4 MB | Neural Engine | 5.9 ms |

The initializer is fp32, which the Neural Engine does not run: load it with `.all` (GPU) rather than `.cpuAndNeuralEngine`
(CPU, 44 ms), and the other three with `.cpuAndNeuralEngine`.

- **Image encoder**: a 1024² RGB image → `raw_vision_features`, `initial_vision_features` and
  `high_res_feature_0/1`.
- **Initializer**: the features, `point_coords` [1, N, 2] in 1024² pixels, `point_labels` [1, N] (1 positive,
  0 negative, 2 and 3 a box's top-left and bottom-right corners, −1 none) and an optional `mask_input`
  [1, 1, 1024, 1024] (0/1) → `low_res_mask`, `high_res_mask` (logits),
  `best_iou`, `object_pointer`, `object_score` and the three `candidate_masks`, `candidate_ious` and
  `candidate_pointers`. A box takes the same path as in SAM 2, with its corners embedded as the labels say, but the
  parity check does not cover it.
- **Memory encoder**: `raw_vision_features`, a `high_res_mask` and `object_score` → `memory_features`,
  `memory_positions` and `temporal_positions`.
- **Propagator**: the features and the memory bank (`spatial_bank`, `spatial_positions` [1, 7, 512, 64],
  `pointer_bank` [1, 16, 256], `attention_bias`) → the initializer's outputs, plus this frame's `memory_features`
  and `memory_positions`.

The docstring of [`edgetam_coreml/convert.py`](edgetam_coreml/convert.py) has every shape and the exact semantics.

### The tracking loop

[`examples/track.py`](examples/track.py) is the whole loop, with coremltools alone:

1. Run the image encoder on every frame.
2. On the prompted frame, run the initializer with the click, then the memory encoder on its mask (binarized). That
   gives the seed memory and pointer.
3. On every later frame, run the propagator with the bank. Its mask is the answer, and its memory and pointer join
   the bank.

The bank is assembled the way SAM 2 does it:

| Slot | Spatial memory (`spatial_bank`) | Temporal row added to its positions |
|---|---|---|
| 0–5 | the 1st…6th previous frame | 0–5 |
| 6 | the prompted (seed) frame | 6 |

| Slot | Pointer (`pointer_bank`) |
|---|---|
| 0 | the seed's |
| 1–15 | the 15 previous frames', most recent first |

Leave slots that have no frame yet as zeros (see below).

## Accuracy

`edgetam_coreml.parity` runs EdgeTAM's PyTorch `SAM2VideoPredictor` and the packages side by side on EdgeTAM's bedroom video.
There are five sequences of 40 frames: three objects from frame 0, one from frame 120, and one with 8 black frames in
the middle, where the object leaves and comes back. Each is seeded by a click, and the Core ML side carries its own
memory from frame to frame. Each also checks three prompts on the seed frame: one click must pick PyTorch's
candidate, and several points, and several points with a mask, must give PyTorch's mask.

The packages attend the memory bank's empty slots (see [below](#neural-engine-and-attention-masks)), so there are
two references: PyTorch with its bank padded the same way (`--attend-empty-slots`), which tests the conversion, and
stock EdgeTAM, which shows what that design choice costs.

| Build, backend | Reference | mask IoU min / mean | memory cosine min | pointer cosine min | sequences within bounds | prompts |
|---|---|---|---|---|---|---|
| fp32, CPU | padded | 0.981 / 0.9997 | 0.99997 | 1.0000 | 5 of 5 | 15 of 15 |
| fp16 (shipped), GPU | padded | 0.963 / 0.998 | 0.905 | 0.944 | 3 of 5 | 15 of 15 |
| fp16 (shipped), Neural Engine | padded | 0.839 / 0.984 | 0.815 | 0.671 | 0 of 5 | 15 of 15 |
| fp32, CPU | stock | 0.836 / 0.983 | 0.792 | 0.487 | 0 of 5 | 15 of 15 |
| fp16 (shipped), GPU | stock | 0.836 / 0.984 | 0.791 | 0.488 | 0 of 5 | 15 of 15 |
| fp16 (shipped), Neural Engine | stock | 0.894 / 0.986 | 0.798 | 0.479 | 0 of 5 | 15 of 15 |
| `--mask-empty-slots` fp32, CPU | stock | 0.979 / 0.9996 | 0.972 | 0.957 | 4 of 5 | 15 of 15 |
| `--mask-empty-slots` fp16, GPU | stock | 0.941 / 0.997 | 0.944 | 0.933 | 0 of 5 | 15 of 15 |
| `--mask-empty-slots` fp16, Neural Engine | stock | 0.900 / 0.991 | 0.907 | 0.894 | 0 of 5 | 15 of 15 |

The bounds are a mask IoU of at least 0.95 and cosines of at least 0.99 on every frame. The reports are in
[`results/`](results).

What the rows show:
- **The conversion is exact.** The fp32 build matches the padded reference. `convert.sh` fails if it falls short.
- **The shipped fp16 packages add rounding.** It accumulates over long runs, and the Neural Engine rounds most.
  Every prompt still gives PyTorch's choice on every backend.
- **Attending the empty slots changes tracking.** It accounts for most of the difference from stock EdgeTAM: the
  fp32 and fp16 builds are equally far from it. The slots hold zero memories with zero positions. They still draw
  attention, because the memory attention's key and value projections have biases. That matters most in the first
  frames after a prompt, while the bank is filling (frames 1–6 for memories, 1–15 for pointers). Masks stay close
  (mean IoU 0.98), but they are not stock EdgeTAM's.
- **Masking the empty slots brings tracking close to stock EdgeTAM.** In fp32 four of five sequences are within the
  bounds. The fifth misses on frame 38 of the gap sequence (memory cosine 0.972, pointer 0.957, mask IoU 0.979),
  where two candidates have the same predicted IoU (0.96338) and Core ML picks the other one. In the shipped fp16 no
  sequence is within the bounds, because fp16 rounding remains. The masked build is still much closer to stock than the
  default: pointer cosine 0.894 against 0.479 on the Neural Engine.

## Speed

Apple M4, warm, back to back, load average 2.5 (`edgetam_coreml.speed`, [`results/speed.jsonl`](results/speed.jsonl)).
Devices are where Core ML's compute plan puts the operations, constants left out:

| | compute units | operations | median | p95 |
|---|---|---|---|---|
| image encoder | `.cpuAndNeuralEngine` | 275 on the Neural Engine | 14.2 ms | 15.8 ms |
| propagator | `.cpuAndNeuralEngine` | 623 on the Neural Engine | 35.0 ms | 35.5 ms |
| **a tracked frame** | | | **49.2 ms** | |
| initializer (once per prompt) | `.all` | 277 on the GPU, 90 on the CPU | 12.3 ms | 16.1 ms |
| initializer (once per prompt) | `.cpuAndNeuralEngine` | 367 on the CPU | 44.2 ms | 44.5 ms |
| memory encoder (once per prompt) | `.cpuAndNeuralEngine` | 256 on the Neural Engine | 5.9 ms | 6.0 ms |

With `.all` Core ML splits the propagator between the Neural Engine and the GPU, and it takes 40.9 ms.

The first load compiles for the Neural Engine and takes about 3 s for the propagator. The image encoder does not read
the memory bank, so an app can encode frame n + 1 while frame n propagates.

## Neural Engine and attention masks

Core ML on the Neural Engine **drops the `attn_mask` of `scaled_dot_product_attention`**. With
`scaled_dot_product_attention(…, attn_mask=bias)`, the compiled graph gives bit-identical outputs whatever the mask,
while the GPU and CPU apply it. Any SAM 2 port that masks the empty slots of a fixed-size memory bank this way
therefore attends them on the Neural Engine, and differs between backends.

The default packages make that behavior explicit:
- The empty slots are attended on every backend, as zero memories with zero positions.
- `attention_bias` stays an input, so the interface does not change, but it is not read.
- Tracking is the same on every backend.

`parity --attend-empty-slots` pads PyTorch's bank the same way. The "padded" rows of [Accuracy](#accuracy) are
against it, and the "stock" rows show what attending the slots costs.

`convert --mask-empty-slots` masks the empty (all-zero) slots as stock EdgeTAM does. The mask is computed from the
bank, and the masked attention is written out as softmax(QKᵀ/√d + bias)·V instead of SDPA's `attn_mask`. The
Neural Engine keeps it, so the propagator stays there (641 operations): 37.7 ms against the default's 36.1 ms,
measured interleaved on the same M4. Releases from v1.2 ship it as a separate asset.

## Compared with the community conversion

[AhmadYarAI/EdgeTAM-CoreML-Video](https://huggingface.co/AhmadYarAI/EdgeTAM-CoreML-Video) uses the same names and
shapes where the meaning is the same. Code that builds its inputs by name (Python, or Swift with
`MLDictionaryFeatureProvider`) carries over once it stops passing `rotary_weight`. The classes Xcode generates
differ, because the propagator has no `rotary_weight` input, so Swift code written against the community package's
generated classes needs that argument removed. Differences:

| | community | this repo |
|---|---|---|
| Prompts | one positive click | 1–16 points (positive and negative), a box, a mask prompt, or points with a mask |
| Outputs | the chosen mask | the chosen mask plus all three candidates with IoUs and pointers |
| Rotary positions | `rotary_weight` input | fixed inside the propagator |
| Propagator, Neural Engine | ~45 ms | 35 ms |

## Building by hand

From the repository root, in the environment `convert.sh` made (or any Python 3.12 with `requirements.txt`):

```bash
python -m edgetam_coreml.convert --edgetam <EdgeTAM checkout> --out <folder> [options]
python -m edgetam_coreml.parity --edgetam <EdgeTAM checkout> --models <folder> --work <scratch> [--units ane|gpu|cpu] [--attend-empty-slots]
python -m edgetam_coreml.speed <folder> [--units ane|gpu|cpu]
```

| Option | Effect |
|---|---|
| `--only encoder,initializer,memory,propagator` | build some of the packages |
| `--fp32 <packages or none>` | packages computed in fp32 (default: `initializer`) |
| `--mask-empty-slots` | the propagator masks the bank's empty slots, as PyTorch does |
| `--weights pkg:int8,…` | weight compression: int8, pal4, pal6, pal8. EdgeTAM's small weights lose accuracy, so it is not recommended |
| `--smooth-seed-memory` | the memory encoder downsamples and upsamples its input the way SAM 2 treats a mask prompt |

Python 3.12 is required, because coremltools 9.0 has no wheels for 3.13 and later. The upstream source is
facebookresearch/EdgeTAM at commit `7711e01`, and `edgetam_coreml/upstream.py` pins the checkpoint's SHA-256.

## Repository layout

```
convert.sh               the clean build: clone EdgeTAM, venv, convert, parity
edgetam_coreml/
  convert.py             the conversion: PyTorch modules restated for Core ML, then ct.convert
  parity.py              the comparison with EdgeTAM's PyTorch video predictor
  speed.py               latency per package and compute unit
  upstream.py            the pinned EdgeTAM commit and checkpoint, and the predictor builder
examples/track.py        tracking from a click with coremltools alone
results/                 parity reports and speed
docs/                    the demo animation
```

## Related

- [edgetam-onnx](https://github.com/jutsuhq/edgetam-onnx): the same tracker on Windows and Linux, in Rust on ONNX Runtime's CUDA provider
- [Jutsu Studio](https://github.com/jutsuhq/jutsu-studio): the open-source Higgsfield desktop app for Mac
- [Swapjutsu](https://github.com/jutsuhq/swapjutsu-seedance-character-swap): free character swap for Mac, no watermark
- [Content Factory](https://github.com/jutsuhq/content-factory): open-source faceless video generator for Shorts, TikTok and Reels
- [Higgsfield Gateway](https://github.com/jutsuhq/higgsfield-gateway): a budget-capped Higgsfield API gateway on Cloudflare Workers
- [AI video API prices](https://github.com/jutsuhq/video-api-prices): what one second of AI video costs at 30+ providers, updated daily
- [awesome-higgsfield](https://github.com/jutsuhq/awesome-higgsfield): tools, SDKs, agent skills and apps for Higgsfield

## License

EdgeTAM is Apache-2.0 (Meta Platforms, Inc.), and so are the converted packages and this repository's code.
[`LICENSE`](LICENSE) is EdgeTAM's license. [`NOTICE`](NOTICE) names the upstream work and must travel with every copy
of the packages. The demo video is [CC0](https://creativecommons.org/publicdomain/zero/1.0/) from Wikimedia Commons.
