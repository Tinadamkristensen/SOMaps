# Fish Oil White Matter Analysis - Enhanced Script

## Overview

This MATLAB script (`PUFA_WM_permute.m`) analyzes the effect of fish oil intervention on white matter regions of the brain using mixed-effects models. The script has been enhanced with three major features:

1. **Confidence Intervals for β coefficients**
2. **Standardized Effect Sizes**
3. **Permutation-Based Global Test**

## Requirements

- MATLAB 2023b or later
- Statistics and Machine Learning Toolbox
- Data file: `merged_COPSYCH_WM_DTI_FBA_2.xlsx`

## New Features

### 1. Confidence Intervals for β Coefficients

For each ROI-specific group effect, the script now computes 95% confidence intervals using:

```matlab
SE_beta = sqrt(c * covB * c')
CI_low  = beta - t_crit * SE_beta
CI_high = beta + t_crit * SE_beta
```

Where:
- `covB` is the coefficient covariance matrix from the mixed model
- `t_crit` is the critical t-value for 95% CI
- The contrast vector `c` defines the specific hypothesis being tested

**Output**: Each ROI now reports beta estimates with confidence intervals, e.g.:
```
FA_corpus_callosum   F = 5.234   p = 0.0231   beta = 0.01234 [0.00456, 0.02012]   std.effect = 0.234
```

### 2. Standardized Effect Sizes

Standardized effects are computed by dividing the raw beta coefficient by the residual standard deviation:

```matlab
stdEffect = beta / residSD
```

This provides a Cohen's d-like measure that is comparable across ROIs and studies.

**Benefits**:
- Scale-free interpretation
- Comparable across different measurement units
- Facilitates meta-analysis and effect size comparison

**Interpretation**:
- Small effect: |d| ≈ 0.2
- Medium effect: |d| ≈ 0.5
- Large effect: |d| ≈ 0.8

### 3. Permutation-Based Global Test

The script implements a non-parametric permutation test for the global fish oil effect:

**Algorithm**:
1. Preserve the observed F-statistic from the omnibus test
2. Permute group labels at the subject level (maintaining within-subject dependencies)
3. Refit the full mixed-effects model with permuted labels
4. Extract the F-statistic for the same global contrast
5. Repeat for 1,000 permutations
6. Calculate p-value as proportion of permuted F ≥ observed F

**Advantages**:
- No distributional assumptions
- Robust to violations of parametric assumptions
- Controls for multiple testing at the global level
- Maintains subject-level data structure

**Output**:
- Permutation p-value
- Visualization of null distribution with observed statistic
- Comparison with parametric p-value

## Visualizations

The enhanced script generates four key figures:

1. **Permutation Distribution**: Shows the null distribution of F-statistics with the observed value marked
2. **Beta Distribution**: Histogram of raw beta coefficients across ROIs
3. **Standardized Effects Distribution**: Histogram of standardized effect sizes
4. **Forest Plot**: Error bar plot showing beta estimates with 95% CIs for all ROIs

## Output Summary

The script provides:

- **Global Test Results**: Parametric and permutation-based p-values
- **ROI-Specific Results**: For each ROI:
  - F-statistic and p-value
  - Beta coefficient with 95% CI
  - Standardized effect size
- **FDR-Corrected Results**: Benjamini-Hochberg corrected p-values
- **Results Table**: Exportable table with all metrics

### Exporting Results

To save the results table to CSV:

```matlab
writetable(resultsTable, 'fishoil_results.csv');
```

## Usage

1. Update the data path in the script:
```matlab
db1 = readtable('/path/to/your/data/merged_COPSYCH_WM_DTI_FBA_2.xlsx');
```

2. Run the script in MATLAB:
```matlab
analyze_fishoil_white_matter
```

3. Review the console output for detailed statistics
4. Examine the generated figures for visual assessment
5. Export the results table if needed

## Interpretation Guide

### Confidence Intervals
- If CI excludes 0, the effect is statistically significant at α = 0.05
- Width of CI indicates precision of estimate
- Overlapping CIs suggest similar effect magnitudes

### Standardized Effects
- Positive values indicate fish oil increases FA
- Negative values indicate fish oil decreases FA
- Magnitude indicates strength of effect (see interpretation above)

### Permutation Test
- If permutation p < 0.05, there's a significant global effect
- Provides robustness check against parametric assumptions
- More conservative than parametric test in small samples

## Technical Details

### Mixed-Effects Model Specification

```matlab
FA ~ Group*ROI + NU_age + NU_sex + 
     NU_Tx_motion + NU_Ty_motion + NU_Tz_motion + 
     NU_Rx_motion + NU_Ry_motion + NU_Rz_motion + 
     (1|SubjectID)
```

- **Fixed Effects**: Group, ROI, their interaction, age, sex, and motion parameters
- **Random Effects**: Random intercept for each subject
- **Outcome**: Fractional Anisotropy (FA) values

### Computational Considerations

- Permutation test with 1,000 iterations takes ~5-10 minutes depending on dataset size
- Increase `nPerm` for more precise p-values (e.g., 5,000 or 10,000)
- Progress updates printed every 100 permutations

## References

- **Permutation Testing**: Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation tests for functional neuroimaging. *Human Brain Mapping*.
- **Effect Sizes**: Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
- **FDR Correction**: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society*.

## Troubleshooting

**Issue**: "Interaction term not found" warning
- **Cause**: Missing ROI interaction term (expected for first ROI)
- **Solution**: This is normal behavior; the first ROI serves as reference

**Issue**: Permutation test runs slowly
- **Cause**: Large dataset or complex model
- **Solution**: Reduce `nPerm` or use parallel computing (requires Parallel Computing Toolbox)

**Issue**: Some permutations produce NaN
- **Cause**: Model convergence issues with certain permutations
- **Solution**: These are automatically excluded; reported in final count

## License

This code is provided for research purposes. Please cite appropriately if used in publications.

## Contact

For questions or issues, please open an issue in the repository.
