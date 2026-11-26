# DNA Origami Structural Decoding and Random-Number Pipeline

A reproducible, modular computational framework for decoding structural
information from atomic force microscopy (AFM) images of DNA origami. The
repository integrates instance segmentation, keypoint detection, geometric
classification, image-derived feature extraction, balanced bit encoding, and
statistical validation of random bitstreams.

The code supports three connected workflows:

1. **Dimer analysis** using YOLOv5-Seg followed by HRNet-based keypoint
   detection and grouping.
2. **Monomer classification** for two-point and three-point structural
   variants, with explicit validity checks.
3. **Multicode analysis** for AFM chain segmentation, visual model-assisted
   code recognition, feature extraction, bitstream construction, and NIST
   SP 800-22 testing.

> This repository contains source code, model configurations, method
> documentation, and example reference assets. AFM images, annotations,
> trained weights, API credentials, and experiment outputs are intentionally
> excluded.

## Scientific Overview

AFM images of DNA origami encode information through the presence and position
of nanoscale features, including streptavidin-labeled white dots on designed
cross-shaped structures. Manual inspection is accurate but does not scale to
large image libraries. This framework converts those images into structured
measurements in a staged and auditable manner.

```text
AFM image
   │
   ├─ YOLOv5-Seg instance detection and segmentation
   │      │
   │      ├─ dimer workflow ── HRNet keypoints ── grouping/classification
   │      └─ monomer workflow ── HRNet keypoints ── validity/code
   │
   └─ multicode workflow
          ├─ chain splitting: template and periodic segmentation
          ├─ single-cross recognition: Qwen-assisted or manual labels
          ├─ 62-dimensional geometric and image-feature extraction
          ├─ balanced feature-to-bit encoding
          └─ NIST SP 800-22 assessment and publication figures
```

The modular separation is deliberate. Image segmentation, geometric
measurement, bit generation, and statistical testing can be run and audited
independently, and a change in one stage does not require retraining an
unrelated model.

## Principal Capabilities

- Low-contrast AFM image preprocessing with grayscale conversion, mask
  dilation, box expansion, and Unicode-path-safe image I/O.
- YOLOv5-Seg training and inference for DNA origami structures.
- HRNet heatmap regression for two-, three-, and four-point structures.
- PCK and OKS keypoint evaluation.
- Rule-based grouping with distance constraints for visually similar dot
  positions.
- Template-matching and periodic-center methods for splitting connected
  origami chains.
- Qwen-compatible visual recognition with an explicit manual fallback.
- Reproducible geometric descriptors, including dot residuals, contrast,
  area, arm length, inter-arm angle, orientation, roughness, and tilt.
- Balanced binary encoding designed to reduce constant-bit and class-imbalance
  effects in small image-derived datasets.
- NIST SP 800-22 wrappers, pass-rate summaries, p-value uniformity analysis,
  and publication-quality SVG/PNG charts.

## Repository Layout

```text
.
├── dimer/
│   ├── pipeline.py
│   ├── stage1_yolo/
│   ├── stage2_hrnet/
│   ├── datapredict/
│   └── analysis/
├── monomer/
│   ├── stage1_yolo/
│   ├── stage2_hrnet/
│   ├── pipelines/
│   │   ├── pipeline.py
│   │   ├── pipeline_two_point.py
│   │   └── pipeline_three_point.py
│   └── analysis/
├── multicodes/
│   ├── src/
│   ├── analysis/
│   ├── config/
│   ├── docs/
│   └── assets/
├── docs/
├── data/
└── models/
```

`data/` and `models/` contain only instructions. Actual images and weights are
mounted or placed outside Git.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/PRONIUBITION/DNA-Origami-Pipeline.git
cd DNA-Origami-Pipeline

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

python -m pip install --upgrade pip
pip install -r dimer/stage2_hrnet/requirements.txt
pip install -r multicodes/requirements.txt
```

YOLOv5 is maintained as an external dependency because the model stack is
distributed under its own license. Clone the official repository into an
external location and pin the reviewed commit used for an experiment:

```bash
git clone https://github.com/ultralytics/yolov5.git /path/to/yolov5
git -C /path/to/yolov5 checkout <reviewed-commit>
```

Set paths through environment variables:

```bash
export YOLOV5_DIR=/path/to/yolov5
export YOLO_WEIGHTS=/path/to/yolo_best.pt
export HRNET_WEIGHTS=/path/to/hrnet_best.pth
export DNA_DATA_ROOT=/path/to/data
export DNA_OUTPUT_ROOT=/path/to/outputs
```

On Windows PowerShell, use `$env:NAME = "value"`.

## Quick Start

### Dimer workflow

```bash
python dimer/pipeline.py \
  --input /path/to/afm/images \
  --yolo-weights /path/to/yolo_best.pt \
  --hrnet-weights /path/to/hrnet_best.pth \
  --output outputs/dimer
```

The stage-one model produces segmentation masks and expanded instance crops.
The stage-two model predicts four keypoints. The pipeline then applies grouping
rules and exports class sequences, annotated images, and a detailed JSON
record.

### Monomer workflow

The unified entry point selects the two-point implementation for types `0` and
`1`, and the three-point implementation for `00`, `01`, `10`, and `11`.

```bash
python monomer/pipelines/pipeline.py \
  --type 00 \
  --input /path/to/afm/images \
  --yolo-weights /path/to/yolo_best.pt \
  --hrnet-weights /path/to/hrnet_type00.pth \
  --yolov5-dir /path/to/yolov5 \
  --output outputs/monomer
```

Each monomer is labeled with its expected code only when the required number of
high-confidence keypoints is present. Otherwise it is labeled `invalid`.
Masked, grayscale crops and keypoint visualizations are saved separately.

### Multicode workflow

```bash
python multicodes/src/pipeline.py \
  --input /path/to/afm/chains \
  --output outputs/multicode_run \
  --mode auto
```

For batch recognition through the Qwen-compatible API:

```bash
export DASHSCOPE_API_KEY=your_key_here
python multicodes/src/recognize_crops.py --groups sample_group --skip-existing
```

Credentials are read from the environment or from an untracked
`multicodes/config/config.json`. Never commit a real API key.

Feature extraction and bit generation:

```bash
python multicodes/src/build_all_features.py
python multicodes/src/build_random_stream.py --order interleave
```

NIST STS execution requires a separately compiled official C implementation:

```bash
python multicodes/src/run_official_sts.py \
  --input outputs/random_stream.bin \
  --name combined
```

## Data and Annotation Formats

### YOLO segmentation

Each AFM image requires a normalized YOLO polygon label:

```text
class_id x1 y1 x2 y2 ... xn yn
```

Coordinates are normalized to `[0, 1]`. Empty label files denote images
without a target. Train/validation splits should use image-level partitioning
to prevent augmented copies of one source image from leaking across sets.

### HRNet keypoints

HRNet expects one heatmap per defined keypoint. Label files may use LabelMe
JSON. The number of keypoints is fixed by the selected YAML:

- types `0` and `1`: two keypoints;
- types `00`, `01`, `10`, and `11`: three keypoints;
- dimer/four-point experiments: four keypoints.

### Multicode features

Each recognized cross is described by a fixed-order feature vector. Features
include local dot morphology and alignment residuals, plus cross-level
geometry. All feature definitions and encoding options are implemented in
`multicodes/src/cross_features_v2.py`,
`multicodes/src/dot_features.py`,
`multicodes/src/digit_encoding.py`, and
`multicodes/src/build_random_stream.py`.

## Training

### YOLOv5-Seg

Stage-one configurations are in `dimer/stage1_yolo/configs` and
`monomer/stage1_yolo/configs`. Paths in the example YAML files are relative to
the configuration file and must be replaced for a new dataset.

Typical settings for AFM data include low photometric augmentation, bounded
translation and scale changes, and explicit validation of mask boundaries after
box expansion. Use the settings recorded in the selected experiment rather
than mixing hyperparameters from several training rounds.

### HRNet

Dimer:

```bash
cd dimer/stage2_hrnet
python train.py --config configs/config.yaml
```

Monomer:

```bash
cd monomer/stage2_hrnet
python train.py --type 00
```

Prediction:

```bash
python predict.py \
  --type 00 \
  --weights /path/to/model.pth \
  --input /path/to/crops \
  --output /path/to/results
```

Use a validation set that does not contain augmented copies of training images
when reporting generalization performance. Training-set-derived tests are
appropriate for debugging overfitting but must not be presented as independent
test performance.

## Evaluation

The repository distinguishes three evaluation levels:

1. **Detection and segmentation**: precision, recall, F1, box mAP, and mask
   mAP.
2. **Keypoints**: percentage of correct keypoints (PCK) and object keypoint
   similarity (OKS).
3. **Randomness**: NIST SP 800-22 pass rates, uniformity of p-values, and
   clearly documented stream length and partitioning.

Random bitstreams must be partitioned only at biologically or
experimentally defined image boundaries. Splitting a bitstream into arbitrary
blocks can create dependence between streams and inflate apparent pass rates.

## Reproducibility

For every reported experiment, archive:

- source-image identifiers and the train/validation/test split;
- YOLO and HRNet configuration files;
- model weight SHA-256 checksums;
- random seeds;
- software and GPU versions;
- exact NIST STS input files and partition definitions;
- generated metrics, logs, and figure-generation commands.

Do not commit raw data or weights directly to Git. Use institutional storage or
a release service that can issue persistent identifiers and checksums.

## Security and Privacy

- API keys must be supplied through environment variables or an untracked
  local configuration file.
- AFM data and annotations may contain unpublished research information.
  Review institutional data-sharing requirements before making the repository
  public.
- Report security issues privately rather than opening a public issue.

## Limitations

- Performance depends on image contrast, scan quality, and consistency between
  AFM acquisition sessions.
- Optical or visual-model assistance is not a substitute for validated
  annotation in publication-grade experiments.
- Statistically valid random-bit expansion requires independent source
  measurements; deterministic pseudo-random expansion must be reported
  separately and cannot be presented as measurement-derived entropy.
- The code is a research workflow and is not intended for clinical use.

## Citation

If this repository contributes to a publication, cite the associated
manuscript when available and include the repository URL, release version, and
commit hash.

The metadata file `CITATION.cff` can be updated with the final manuscript
details before release.

## License

Original code in this repository is released under the MIT License. Third-party
software and assets retain their own licenses; see
`THIRD_PARTY_NOTICES.md`.
