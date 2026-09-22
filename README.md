

# ICPhS 2027: When representations change: Speaker variation across multidimensional spaces

This repository contains the analysis code for our ICPhS 2027 study examining how observed speaker variability changes across different multidimensional speech representations.

The study compares two hand-crafted representations—rhythmic features and eGeMAPS—with learned representations extracted from multiple layers of wav2vec 2.0 XLSR-53. The central question is whether patterns of between- and within-speaker variability observed in one representational space are preserved in another.

## Data

The analysis uses the Persian component of the study corpus, comprising 30 speakers completing seven speaking tasks, for a total of 210 recordings.

Three types of speech representation are examined:

- **Rhythm:** 28 temporal, F0, and intensity features
- **eGeMAPS:** 88 acoustic features
- **XLSR-53:** 1024-dimensional embeddings extracted from Transformer layers 3, 6, 9, 12, 15, 18, 21, and 24

Speaker identifiers are harmonized across the three feature sets before analysis.

## Analysis

Features are z-standardized separately within each representation and reduced using principal component analysis (PCA). The minimum number of principal components explaining at least 80% of the variance is retained.

Speaker variability is then examined from three complementary perspectives:

1. **Within-speaker task variability**  
   The dispersion of each speaker's seven task recordings around their centroid in PCA space.

2. **Between-speaker peripherality**  
   The Mahalanobis distance of each speaker from the task-specific population center, averaged across tasks.

3. **Pairwise speaker organization**  
   Representational similarity analysis (RSA) based on pairwise Euclidean distances among task-centered speaker centroids.

Correspondence across representations is assessed using Spearman rank correlations. Permutation tests are used for statistical inference, with speaker-level bootstrap confidence intervals for task variability and peripherality. False discovery rate (FDR) correction is applied to layer-wise XLSR-53 comparisons.

The analysis also directly compares XLSR-53 layers to examine how speaker structure changes across the learned representation.

## Repository structure

```text
scripts/
├── ICPhS (2).ipynb
├── build_16k_manifest.py
├── run_wav2vec2_layers.sh
├── nkululeko_configs/
├── nkululeko_results/
└── ...
