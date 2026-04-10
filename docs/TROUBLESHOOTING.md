# Troubleshooting Guide

## Common Issues

### 1. "No CSV files found in data/real/raw/"

**Cause**: No real data has been placed in the expected directory.

**Solution**: The pipeline will automatically generate demo data for testing. To use your own data, place CSV files in `data/real/raw/` following the format described in `docs/DATA_FORMAT.md`.

### 2. Isomap fails with "graph is not fully connected"

**Cause**: The neighborhood graph built by Isomap has disconnected components, typically because `n_neighbors` is too small for sparse data.

**Solutions**:
- Increase `n_neighbors` in the config (try 12–20).
- Switch to `method: diffusion_map` which handles disconnected graphs more gracefully.
- Check for outlier frames (large jumps in landmark positions) and increase smoothing.

### 3. Curvature values are extremely large or NaN

**Cause**: Numerical instability in finite differences, often due to near-zero velocity (stationary frames) or noise.

**Solutions**:
- Apply smoothing before computing curvature: `traj.smooth(window_length=9, polyorder=3)`.
- Increase interpolation points: `traj.interpolate(n_points=300)`.
- The code clips near-zero velocities, but extreme noise may still cause issues.

### 4. SRVF distance computation is slow

**Cause**: Dynamic programming alignment is O(N²) where N is the resampling resolution.

**Solutions**:
- Reduce `n_resample` in config (100 is usually sufficient for preliminary analysis).
- Reduce `n_dp_grid` (50 is a reasonable lower bound).
- For large cohorts, compute distances in parallel or use the unaligned SRVF distance as a fast approximation.

### 5. Phenotype classification accuracy is low

**Possible causes**:
- Insufficient sample size (< 20 per class).
- Overlapping phenotypes (e.g., mild fibrosis vs. mild weakness).
- Noisy landmark extraction.

**Solutions**:
- Check feature distributions: are they actually separating groups?
- Try `method: gradient_boosting` which may handle overlapping classes better.
- Examine feature importances to ensure informative features are being used.
- Consider binary classification (healthy vs. dysphagic) before multi-class.

### 6. Phase detection finds wrong number of phases

**Cause**: Curvature threshold is not well-calibrated for the data.

**Solutions**:
- Adjust `curvature_threshold` (lower values detect more transitions).
- The algorithm falls back to velocity-based segmentation if curvature peaks are insufficient.
- For real data, consider manually annotating a small training set to calibrate thresholds.

### 7. Memory errors with large cohorts

**Solutions**:
- Process trajectories in batches rather than stacking all landmarks at once.
- Use `method: pca` for manifold learning (much lower memory than Isomap).
- Reduce interpolation resolution.

### 8. Correlations with clinical scores are weak

**Possible causes**:
- Clinical scores (FOIS, MBSImP) are ordinal with limited resolution; manifold metrics are continuous.
- Small sample size limits statistical power.
- Landmark extraction noise masks true physiological signal.

**Solutions**:
- Use Spearman (rank) correlation, not Pearson.
- Report effect sizes, not just p-values.
- Consider that weak correlations may reflect genuine information gain: manifold metrics capture information orthogonal to what clinical scores measure.

## Environment Issues

### geomstats installation fails

```bash
pip install geomstats --no-deps
pip install autograd scipy numpy matplotlib
```

### fdasrsf compilation errors

fdasrsf requires a C compiler. On Ubuntu/Debian:
```bash
sudo apt-get install build-essential
pip install fdasrsf
```

On macOS:
```bash
xcode-select --install
pip install fdasrsf
```

### matplotlib "no display" error in headless environments

The scripts use `matplotlib.use("Agg")` for headless rendering. If you still see display errors, ensure this is called before any pyplot imports.
