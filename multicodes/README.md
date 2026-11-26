# Multicode Workflow

The multicode module converts AFM images of linked DNA origami crosses into
discrete code sequences and measurement-derived random bitstreams.

## Stages

1. Pair chain images with canonical single-cross templates.
2. Split chains using template matching, periodic centers, or their union.
3. Recognize each cropped cross manually or with Qwen-compatible visual
   assistance.
4. Extract aligned geometric and image descriptors.
5. Convert the feature table into balanced binary sequences.
6. Evaluate bitstreams with NIST SP 800-22.
7. Export Excel tables, pass-rate charts, and p-value uniformity charts.

## Configuration

Copy the example configuration locally:

```bash
cp multicodes/config/config.example.json multicodes/config/config.json
export DASHSCOPE_API_KEY=your_key_here
```

`multicodes/config/config.json` is ignored by Git.

## Run

```bash
python multicodes/src/pipeline.py \
  --input /path/to/chains \
  --output /path/to/results \
  --mode auto
```

Batch recognition:

```bash
python multicodes/src/recognize_crops.py \
  --groups group_a group_b \
  --skip-existing
```

Features and bitstreams:

```bash
python multicodes/src/build_all_features.py
python multicodes/src/build_random_stream.py --order interleave
```

NIST STS:

```bash
python multicodes/src/run_official_sts.py \
  --input /path/to/random_stream.bin \
  --name combined
```

The official `nist_sts/` checkout and compiled `assess.exe` are external
prerequisites and are ignored by Git.

## Reproducibility Rules

- Preserve the source image name for every feature record.
- Define stream partitions at image boundaries, not arbitrary bit offsets.
- Record stream lengths, test selection codes, p-value thresholds, and the NIST
  STS version.
- Do not describe deterministic pseudo-random expansion as measured entropy.
