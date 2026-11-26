# Dimer Improvement Plan

## Baseline Findings

The main error modes observed in development were:

- confusion between mirrored two-bit codes;
- unstable near/far decisions when keypoints were displaced;
- whole-pixel heatmap quantization;
- segmentation boxes clipping peripheral white dots.

The first corrective step is therefore to improve keypoint localization before
adding a more complex grouping rule.

## Priority 0: Immediate Improvements

1. Add sub-pixel local fitting around each heatmap peak.
2. Reduce the Gaussian sigma after validating the target dot scale.
3. Revisit coordinate-regression and negative-sample loss weights.
4. Test horizontal-flip and rotation test-time averaging.
5. Expand boxes and tune mask dilation against annotated edge cases.

## Priority 1: Short-Term Retraining

1. Increase input or heatmap resolution when GPU memory permits.
2. Compare HRNet-W32 and larger variants under identical splits.
3. Increase training length only after resolving localization errors.
4. Rebalance rare code classes.
5. Select checkpoints using code accuracy and keypoint metrics, not only
   validation loss.

## Priority 2: Robustness

1. Use geometric orientation or center-direction vectors instead of relying
   entirely on PCA.
2. Add conservative cutout, elastic deformation, and noise augmentation.
3. Audit labels for missing or duplicated dots.
4. Evaluate all parameters on an independent validation set.
5. Preserve an `invalid` path for incomplete structures.

## Priority 3: Expanded Evaluation

1. Test mask-plus-intensity input channels.
2. Expand the annotated four-point dataset.
3. Compare model ensembles only if members fail differently.
4. Report confidence intervals and sample counts with every metric.

## Validation Rules

- Split by source image, not by cropped instance.
- Freeze thresholds before evaluating the held-out test set.
- Distinguish training-set diagnostics from independent test results.
- Never infer statistically valid randomness from deterministic model output.
