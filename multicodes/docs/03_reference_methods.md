# Reference Methods

This document records the reference materials and the machine-learning methods
added for this repository.

## Materials

All short DNA strands were purchased from Sangon (ULTRAPAGE purification,
Shanghai, China). Biotin-modified oligonucleotides were obtained from Sangon
and purified by high-performance liquid chromatography. All staple sequences
used in this study are listed in Supplementary Data 1. All chemicals were
supplied by Sigma. M13mp18 DNA was purchased from BIORULER and used as
received. All solutions were prepared with Milli-Q water
(resistivity = 18.2 MOhm cm).

## Synthesis and Purification of DNA Origami Nodes

The DNA origami nodes were prepared by a one-pot process. A 2.5 nM M13mp18
scaffold solution was mixed with 25 nM original staples, 25 nM biotinylated
staples, and 25 nM connector strands in 1x TAE/Mg2+ buffer (40 mM Tris, 20 mM
acetic acid, 2 mM EDTA, and 12.5 mM magnesium acetate, pH 8.0). The mixture was
annealed in a thermal cycler from 95 to 25 C over 2 h. The prepared nodes were
purified four times with 100 kDa MWCO filters and redispersed in
1x TAE/Mg2+ buffer.

## Annealing for Multimer Assembly

Individual DNA origami code units were mixed at the molar concentrations
required by the target assembly. The mixed samples were annealed for 12 h in a
thermocycler (Thermo Fisher Veriti Fast): 45 to 37 C over 4 h
(0.1 C per 3 min), 37 to 25 C over 8 h (0.1 C per 4 min), followed by cooling
and holding at 4 C.

## Streptavidin Labeling

Commercial streptavidin powder was dissolved in 1x PBS/Mg2+ buffer
(10 mM sodium dihydrogen phosphate, 2 mM disodium hydrogen phosphate,
137 mM sodium chloride, 2.7 mM potassium chloride, and 12.5 mM magnesium
acetate, pH 7.4). Streptavidin was added at a 5:1 molar ratio relative to each
biotinylated site. The reaction proceeded at 37 C for 2 h. Labeled origami was
used directly for AFM characterization.

## UV Irradiation

To reduce distortion of DNA nanostructure multimers, samples were exposed to
254 nm UV light for 8 min with a handheld UV analyzer (WFH-204B, 6 W) before
AFM imaging.

## AFM Characterization

Samples were diluted with 1x TAE/Mg2+ to approximately 1 nM. A 4 uL drop was
deposited on freshly cleaved mica for 2 min at room temperature. The substrate
was rinsed three times with ultrapure water, dried, and imaged in air with a
MultiMode 8 AFM and NanoScope V controller (Bruker) in ScanAsyst mode.

## Gel Electrophoresis

Samples were resolved on 0.8-1.2% agarose gels for 100 min at 70 V in an
ice-water bath. The running buffer was 1x TAE/Mg2+. Gels were stained with
0.5 ug/mL ethidium bromide and imaged with a GelDoc XR+ system and Image Lab
5.1 (Bio-Rad).

## Glycerol Gradient Preparation

Glycerol gradients were prepared from 50% to 70% glycerol in 5 mL
ultracentrifuge tubes. Each layer contained 667 uL with a 5% glycerol
concentration decrement. Five-layer gradients were incubated overnight at 4 C
to form a quasi-continuous gradient.

## Ultracentrifugation and Fraction Collection

A 100 uL sample was loaded into a 1.5 mL ultracentrifuge tube and adjusted to
400 uL with 1x TAE/Mg2+ buffer. The tube was centrifuged with a Beckman
Optima XPN-100 at 3,000 x g for 10 min in three cycles. Fractions of
approximately 60 uL were collected into separate tubes, and 1.5 uL from each
fraction was used for quantification.

## Dimer Machine-Learning Analysis

Dimer segmentation and keypoint detection used a two-stage deep-learning
pipeline. A COCO-pretrained YOLOv5s-Seg model was trained on 640 x 640 AFM
images for dimer, monomer, and invalid classes. The reference run used batch
size 4, 500 epochs, freezing of the first 10 layers, and early-stopping
patience 30. Low-intensity photometric augmentation was used because the
images are grayscale. Geometric augmentation included translation, scale,
horizontal flipping, and mosaic.

At inference, low-confidence detections were removed, duplicate boxes were
suppressed, and boxes were expanded before mask-based cropping. The resulting
instances were converted to 8-bit grayscale and resized to 256 x 256.

An ImageNet-pretrained HRNet-W32 predicted four heatmaps:
`left_dot_1`, `left_dot_2`, `right_dot_1`, and `right_dot_2`. Training used
Adam, cosine annealing, batch size 4, 100 epochs, mixed precision, and early
stopping. Peaks were extracted with a heatmap threshold and a
non-maximum-suppression distance. The point-distance rule assigned one bit per
side. PCK and COCO-style OKS were used for keypoint evaluation.

## Multicode Machine-Learning Analysis

The multicode dataset contained AFM chain images paired with single-cross
templates. Straight chains were segmented by normalized cross-correlation
template matching. Curved chains were segmented with autocorrelation-based
period estimation and fixed-width center windows. Recognized crops were
assigned four-bit codes manually or with a vision-language model using the
orientation reference and English rule document.

After alignment to the canonical template, 62 features were extracted:

- 21 white-dot features;
- 26 cross-structure features;
- 15 surface-image features.

Feature vectors were converted to bits without hashing. The shortest exact
decimal representation of each feature was concatenated, interpreted as an
integer, and mapped losslessly into approximately 3.32 bits per decimal digit.

The resulting streams were evaluated with the official NIST SP 800-22 C
implementation. Partitions were defined at the source-image level. Block
lengths for the approximate-entropy and serial tests were selected to match the
available sequence length. Rank was evaluated on the full stream because short
subsequences generated too few matrices for a continuous p-value
distribution.

## Data Availability

The main data supporting the associated study should be provided with the
paper and Supplementary Information. Source images and annotations are not
included in this repository. Other source data may be requested from the
corresponding authors, subject to institutional requirements.

## Code Availability

This repository contains the analysis code. External model weights, source
images, and the NIST STS distribution are obtained separately.
