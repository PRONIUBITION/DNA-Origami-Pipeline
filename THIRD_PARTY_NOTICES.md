# Third-Party Notices

This repository contains original orchestration, image-analysis, encoding, and
evaluation code. It does not redistribute the source or weights of the
following third-party projects.

## Ultralytics YOLOv5

The stage-one detectors use YOLOv5-Seg. Install YOLOv5 from its official
repository and pin a reviewed commit in the deployment environment.

YOLOv5 is distributed under the AGPL-3.0 license unless a separate commercial
license is obtained. Users are responsible for complying with that license.

## HRNet

The keypoint models are implemented locally in `dimer/stage2_hrnet/models` and
`monomer/stage2_hrnet/models`. Any separately downloaded HRNet pretrained
weights remain subject to their original license.

## NIST Statistical Test Suite

The official NIST SP 800-22 Statistical Test Suite is downloaded and compiled
separately. It is not vendored in this repository. Users must follow NIST's
distribution terms.

## Qwen API

The multicodes visual-recognition workflow can call an external model API.
API access, model terms, data-retention rules, and costs are controlled by the
service provider. No credentials are included in this repository.
