# Experimental Design

## Overview

This document describes the experimental design for validating the manifold–trajectory framework for swallowing analysis. Experiments are organized into two tracks: simulated data validation (with known ground truth) and real data analysis (with clinical relevance).

---

## Track 1: Simulated Experiments

### Design Rationale

Simulated experiments establish that the mathematical framework produces the predicted results under controlled conditions. Each experiment tests a specific theoretical prediction from the paper, using synthetic data where the ground-truth manifold structure, trajectory geometry, and clinical phenotype are all known by construction.

### Synthetic Data Generation

**Manifold construction**: A smooth 5-dimensional manifold is embedded in 28-dimensional ambient space (14 landmarks × 2D coordinates) using a random orthogonal embedding with nonlinear coupling terms. The nonlinearity ensures the manifold has genuine curvature, not just a linear subspace.

**Trajectory generation**: Five trajectory types (healthy, fibrotic, weak, compensatory, neurogenic) are generated with distinct geometric signatures in intrinsic coordinates, then embedded onto the manifold with additive Gaussian measurement noise (σ = 0.02).

**Cohort size**: 50 trajectories per condition × 5 conditions = 250 trajectories. This provides adequate statistical power for group comparisons (Mann-Whitney U test power > 0.8 at α = 0.05 for moderate effect sizes).

### Experiment Specifications

| Exp | Prediction | Method | Success Criterion |
|-----|-----------|--------|-------------------|
| 01 | d ≪ N recoverable | PCA + Isomap residual variance | Elbow at d=5, >95% variance captured |
| 02 | Trajectory geometry distinguishes conditions | 3D PCA embedding | Visual separation of trajectory bundles |
| 03 | Geodesic length reflects effort | Arc length computation | Significant group differences (p < 0.001) |
| 04 | Curvature captures coordination | Covariant acceleration norm | Fibrotic < healthy < compensatory |
| 05 | Phases = geometric regions | Curvature peak detection | ≥4/5 phases detected in >80% of swallows |
| 06 | SRVF invariant to time warping | Distance to warped self | SRVF dist ≈ 0 while Euclidean dist >> 0 |
| 07 | Synchrony metrics detect decoupling | Phase lag + coupling strength | Significant condition effects on synchrony |
| 08 | Manifold metrics classify phenotypes | Random forest on metric features | CV accuracy > 80%, AUC > 0.85 |

### Statistical Analysis

- **Group comparisons**: Mann-Whitney U test (non-parametric) with Bonferroni correction for multiple comparisons.
- **Classification**: Stratified 5-fold cross-validation with Random Forest (100 trees, max depth 10).
- **Effect sizes**: Rank-biserial correlation for U tests; macro AUC for classification.
- **Confidence intervals**: Bootstrap 95% CI with 1000 resamples for key metrics.

---

## Track 2: Real Data Experiments

### Data Requirements

**Minimum**: 10 healthy controls + 10 patients with dysphagia (any etiology).
**Recommended**: 20+ per group, ideally including multiple dysphagia subtypes.
**Ideal for longitudinal**: Pre-treatment, mid-treatment, 3-month, and 12-month post-treatment time points.

### Landmark Extraction

Landmarks are extracted from midsagittal cine MRI frames. The minimum set of 14 landmarks covers: tongue dorsum (4 points), hyoid bone (2), larynx (2), epiglottis tip/base (2), pharyngeal wall (2), and UES margins (2).

**Extraction methods** (in order of preference):
1. Automated deep learning segmentation (e.g., nnU-Net trained on swallowing MRI)
2. Semi-automated tracking with manual correction
3. Manual annotation by trained raters (inter-rater reliability > 0.85 ICC required)

### Pipeline Steps

1. **Preprocessing**: Savitzky-Golay smoothing (window=7, order=3), centroid normalization, IQR-based outlier removal, cubic spline interpolation to 200 uniform time points.

2. **Manifold learning**: Isomap with k=8 neighbors, intrinsic dimension estimated via residual variance elbow detection.

3. **Metric computation**: Full metric suite (geodesic length, curvature, velocity, smoothness, phase detection, synchrony, bottleneck traversal).

4. **Clinical comparison**: Spearman correlation between manifold metrics and clinical scores (FOIS, MBSImP, PAS). Statistical significance at α = 0.05 with FDR correction.

5. **Group comparison**: Mann-Whitney U between healthy and patient groups for each metric. Effect size reporting with rank-biserial correlation.

### Evaluation Criteria

| Analysis | Primary Outcome | Success Criterion |
|----------|----------------|-------------------|
| Manifold dimensionality | Intrinsic dim estimate | d ≤ 10 with >90% variance |
| Metric–clinical correlation | Spearman ρ with FOIS/MBSImP | ρ > 0.4, p < 0.05 |
| Group discrimination | Mann-Whitney U | Significant difference in ≥3 metrics |
| Phenotype classification | Cross-validated accuracy | Accuracy > 70% (multi-class) |
| Longitudinal sensitivity | Effect size over time | Detectable change before clinical grade shift |

### Known Confounds and Mitigations

| Confound | Mitigation |
|----------|-----------|
| Bolus consistency variation | Standardize protocol (5 mL thin liquid) |
| Head motion | Centroid normalization; rigid registration |
| Landmark extraction error | Inter-rater reliability assessment; smoothing |
| Small sample size | Non-parametric tests; bootstrap CI; report effect sizes |
| Multiple comparisons | Bonferroni or FDR correction |
| Temporal resolution | Require ≥15 fps; report aliasing analysis |

---

## Reporting Standards

All experiments report:
1. Sample sizes and demographics
2. Effect sizes with confidence intervals
3. Exact p-values (not just significance thresholds)
4. Visualization of raw data alongside summary statistics
5. Code and configuration for full reproducibility
