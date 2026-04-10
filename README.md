# Swallowing as a Dynamical System: Manifold–Trajectory Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Computational validation of the manifold–trajectory framework for MRI-based swallowing analysis.**

This repository accompanies the paper *"Swallowing as a Dynamical System: A Manifold–Trajectory Framework for MRI-Based Analysis of Deglutition"* and provides:

1. **Core library** (`src/core/`) — Riemannian manifold construction, trajectory analysis, geodesic computation, curvature estimation, and elastic shape metrics (SRVF).
2. **Simulated experiments** (`experiments/simulated/`) — Synthetic swallowing trajectories that validate each theoretical prediction (manifold dimensionality, curvature–coordination relationship, geodesic length–efficiency relationship, phase region detection, phenotype discrimination).
3. **Real data experiments** (`experiments/real_data/`) — Pipeline for cine MRI swallowing data: landmark extraction, manifold learning, trajectory computation, and clinical metric comparison.
4. **Visualization** (`src/visualization/`) — Publication-quality figures for manifold embeddings, trajectories, curvature profiles, phase diagrams, and phenotype separation.

---

## Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Experiment 1: Simulated Data Validation](#experiment-1-simulated-data-validation)
- [Experiment 2: Real Data Analysis](#experiment-2-real-data-analysis)
- [Experimental Design](#experimental-design)
- [Results Summary](#results-summary)
- [Known Issues & Limitations](#known-issues--limitations)
- [Citation](#citation)
- [License](#license)

---

## Installation

```bash
git clone https://github.com/<your-org>/swallowing-manifold-analysis.git
cd swallowing-manifold-analysis
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.9
- NumPy, SciPy, scikit-learn
- geomstats (Riemannian geometry)
- fdasrsf (elastic shape analysis / SRVF)
- matplotlib, seaborn, plotly
- pandas
- nibabel, SimpleITK (for NIfTI/DICOM MRI I/O)
- PyYAML

---

## Repository Structure

```
swallowing-manifold-analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── configs/
│   ├── simulation_config.yaml      # Parameters for synthetic experiments
│   └── real_data_config.yaml       # Parameters for real data pipeline
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── manifold.py             # Manifold learning & embedding
│   │   ├── trajectory.py           # Trajectory representation & interpolation
│   │   ├── metrics.py              # Geodesic length, curvature, synchrony
│   │   ├── srvf.py                 # Square Root Velocity Function (elastic metrics)
│   │   ├── phase_detection.py      # Geometric phase region identification
│   │   └── phenotype.py            # Dysphagia phenotype classification
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── synthetic_manifold.py   # Generate synthetic swallowing manifolds
│   │   ├── trajectory_generator.py # Healthy & pathological trajectory synthesis
│   │   └── noise_models.py         # Measurement noise & physiological variability
│   ├── real_data/
│   │   ├── __init__.py
│   │   ├── landmark_extraction.py  # Extract anatomical landmarks from MRI
│   │   ├── preprocessing.py        # Temporal alignment, smoothing, normalization
│   │   └── clinical_scores.py      # Compute FOIS, MBSImP proxies from trajectories
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── manifold_plots.py       # 3D manifold embeddings
│   │   ├── trajectory_plots.py     # Trajectory overlays, phase coloring
│   │   ├── metric_plots.py         # Curvature, velocity, synchrony profiles
│   │   └── phenotype_plots.py      # Phenotype discrimination figures
│   └── utils/
│       ├── __init__.py
│       ├── io.py                   # Data I/O helpers
│       └── config.py               # Configuration loading
├── experiments/
│   ├── simulated/
│   │   ├── run_all_simulated.py    # Master script for all simulated experiments
│   │   ├── exp01_manifold_dim.py   # Intrinsic dimensionality validation
│   │   ├── exp02_trajectory_types.py  # Healthy vs pathological trajectories
│   │   ├── exp03_geodesic_length.py   # Length–efficiency relationship
│   │   ├── exp04_curvature.py      # Curvature–coordination mapping
│   │   ├── exp05_phase_regions.py  # Phase detection accuracy
│   │   ├── exp06_srvf_invariance.py   # Time-warp invariance validation
│   │   ├── exp07_synchrony.py      # Inter-structure synchrony metrics
│   │   └── exp08_phenotypes.py     # Phenotype discrimination power
│   └── real_data/
│       ├── run_real_data_pipeline.py  # End-to-end real data analysis
│       ├── step01_preprocess.py    # Landmark extraction & preprocessing
│       ├── step02_manifold_learn.py   # Learn manifold from real data
│       ├── step03_trajectory_analysis.py  # Compute trajectory metrics
│       ├── step04_clinical_comparison.py  # Compare with clinical scores
│       └── step05_longitudinal.py  # Longitudinal analysis (if available)
├── tests/
│   ├── test_manifold.py
│   ├── test_trajectory.py
│   ├── test_metrics.py
│   ├── test_srvf.py
│   └── test_phase_detection.py
├── notebooks/
│   ├── 01_tutorial_manifold_basics.ipynb
│   ├── 02_simulated_experiments_walkthrough.ipynb
│   └── 03_real_data_analysis.ipynb
├── data/
│   ├── simulated/                  # Generated by simulation scripts
│   └── real/
│       ├── raw/                    # Place raw MRI data here
│       └── processed/              # Processed landmark time series
├── results/
│   ├── figures/
│   └── tables/
└── docs/
    ├── EXPERIMENTAL_DESIGN.md
    ├── DATA_FORMAT.md
    └── TROUBLESHOOTING.md
```

---

## Quick Start

### Run all simulated experiments
```bash
python experiments/simulated/run_all_simulated.py --config configs/simulation_config.yaml
```

### Run real data pipeline
```bash
# 1. Place MRI landmark data in data/real/raw/ (see docs/DATA_FORMAT.md)
# 2. Run pipeline
python experiments/real_data/run_real_data_pipeline.py --config configs/real_data_config.yaml
```

---

## Experiment 1: Simulated Data Validation

Eight experiments validate each theoretical prediction from the paper using synthetic data with known ground truth.

| ID | Experiment | Paper Section | Validates |
|----|-----------|---------------|-----------|
| 01 | Manifold dimensionality | §2 | PCA/Isomap recover d ≪ N |
| 02 | Trajectory types | §2.3, §6 | Healthy vs pathological trajectory geometry |
| 03 | Geodesic length | §5.1 | Length correlates with simulated effort |
| 04 | Curvature profiles | §3.2 | Curvature distinguishes coordination patterns |
| 05 | Phase region detection | §4 | Geometric phases match ground-truth labels |
| 06 | SRVF invariance | §5.2 | Elastic distance invariant to time warping |
| 07 | Inter-structure synchrony | §5.3 | Coupling metrics detect desynchronization |
| 08 | Phenotype discrimination | §6 | Manifold metrics separate clinical phenotypes |

---

## Experiment 2: Real Data Analysis

### Data Requirements

The real data pipeline expects **anatomical landmark time series** extracted from cine MRI of swallowing. Accepted formats:

- **CSV**: Each row is a time frame; columns are landmark coordinates (x₁, y₁, x₂, y₂, ..., xₖ, yₖ) for K landmarks.
- **NIfTI**: 4D volumes (x, y, z, t) with segmentation masks.

### Supported Landmarks (minimum set)

| ID | Landmark | Structure |
|----|----------|-----------|
| 1–4 | Tongue dorsum contour (4 points) | Tongue |
| 5–6 | Hyoid bone (anterior, posterior) | Hyoid |
| 7–8 | Larynx (superior, inferior) | Larynx |
| 9–10 | Epiglottis (tip, base) | Epiglottis |
| 11–12 | Pharyngeal wall (superior, inferior) | Pharynx |
| 13–14 | UES (superior, inferior margins) | UES |

### Pipeline Steps

1. **Preprocessing** — Temporal smoothing, spatial normalization, outlier removal.
2. **Manifold learning** — Isomap/diffusion maps to estimate intrinsic dimension and embed trajectories.
3. **Trajectory analysis** — Geodesic length, curvature, velocity profiles, phase detection.
4. **Clinical comparison** — Correlate manifold metrics with FOIS/MBSImP scores (if available).
5. **Longitudinal analysis** — Track manifold deformation over treatment time points.

---

## Experimental Design

See [docs/EXPERIMENTAL_DESIGN.md](docs/EXPERIMENTAL_DESIGN.md) for the full protocol. Key design decisions:

- **Simulated data**: 5 manifold dimensions, 14 landmarks (28D ambient), 100 time frames per swallow, 200 swallows per condition, 5 conditions (healthy, fibrotic, weak, neurogenic, compensatory).
- **Real data**: Minimum 10 healthy controls + 10 patients recommended. Landmarks extracted at 15+ fps.
- **Statistical tests**: Permutation tests for group differences, bootstrap confidence intervals for metric estimates, Bonferroni correction for multiple comparisons.
- **Evaluation metrics**: AUC for phenotype classification, Spearman correlation for clinical score comparison, reconstruction error for manifold dimensionality.

---

## Results Summary

Results are generated in `results/figures/` and `results/tables/`. Key expected outputs:

- **Figure 1**: Scree plot showing intrinsic dimensionality (d ≈ 5 recoverable from 28D ambient space).
- **Figure 2**: 3D manifold embedding with healthy (smooth) and pathological (distorted) trajectories.
- **Figure 3**: Geodesic length distributions by phenotype (fibrotic > compensatory > healthy).
- **Figure 4**: Curvature profiles showing coordination complexity differences.
- **Figure 5**: Phase region detection accuracy vs ground truth.
- **Figure 6**: SRVF distance matrix showing time-warp invariance.
- **Figure 7**: Synchrony metrics separating coupled vs decoupled swallowing.
- **Figure 8**: Phenotype confusion matrix from manifold-based classification.
- **Table 1**: Manifold metric summary statistics by group.
- **Table 2**: Correlation of manifold metrics with simulated clinical scores.

---

## Known Issues & Limitations

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for details.

1. **Manifold learning sensitivity**: Isomap neighborhood parameter k requires tuning per dataset; diffusion maps are more robust but slower.
2. **Landmark extraction**: Automated landmark detection from MRI is not included; we assume pre-extracted landmarks. Integration with deep learning segmentation (e.g., nnU-Net) is planned.
3. **Sample size**: Real data experiments require sufficient subjects for statistical power; N < 10 per group yields unreliable phenotype classification.
4. **Temporal resolution**: Cine MRI below 10 fps may alias fast pharyngeal events, degrading curvature estimates.
5. **2D vs 3D**: Current implementation uses midsagittal 2D landmarks; 3D extension requires volumetric MRI at sufficient frame rate.
6. **SRVF on manifolds**: The current SRVF implementation uses Euclidean SRVF with manifold projection; a fully intrinsic transported SRVF (TSRVF) is under development.

---

## Citation

```bibtex
@article{swallowing_manifold_2026,
  title={Swallowing as a Dynamical System: A Manifold--Trajectory Framework
         for MRI-Based Analysis of Deglutition},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
