# Quick Reference Guide: Enhanced Fish Oil Analysis

## Quick Start

```matlab
% 1. Open MATLAB 2023b or later
% 2. Navigate to the script directory
% 3. Update the data path (line 8):
db1 = readtable('/your/path/to/merged_COPSYCH_WM_DTI_FBA_2.xlsx');

% 4. Run the script:
analyze_fishoil_white_matter

% 5. (Optional) Export results:
writetable(resultsTable, 'fishoil_results.csv');
```

---

## What You'll Get

### Console Output

#### 1. Global Test Results (Parametric)
```
GLOBAL fish oil effect across all ROIs:
F(20, 1234.5) = 3.456, p = 0.0001
Global partial R^2 = 0.0532
```

#### 2. Permutation Test Results
```
--- PERMUTATION-BASED GLOBAL TEST ---
Running 1000 permutations...
  Permutation 100/1000
  Permutation 200/1000
  ...
  Permutation 1000/1000

Permutation-based global test results:
Observed F = 3.456
Permutation p-value = 0.0034
(Based on 1000 permutations)
```

#### 3. ROI-Specific Results with CIs and Standardized Effects
```
--- ROI-SPECIFIC EFFECTS WITH CONFIDENCE INTERVALS ---
FA_corpus_callosum      F = 5.234   p = 0.0231   beta = 0.01234 [0.00456, 0.02012]   std.effect = 0.234
FA_anterior_corona      F = 3.891   p = 0.0489   beta = 0.00892 [0.00012, 0.01772]   std.effect = 0.169
FA_posterior_corona     F = 2.345   p = 0.1256   beta = 0.00567 [-0.00156, 0.01290]   std.effect = 0.108
...
```

#### 4. Summary Statistics
```
ROI-level summary:
Mean beta = 0.00845
SD beta   = 0.00423
Mean standardized effect = 0.161
SD standardized effect   = 0.081
```

#### 5. FDR-Corrected Significant Results
```
FDR-significant ROIs (alpha = 0.05):
FA_corpus_callosum   raw p = 0.0231   FDR p = 0.0462   beta = 0.01234 [0.00456, 0.02012]   std.effect = 0.234
FA_anterior_corona   raw p = 0.0489   FDR p = 0.0489   beta = 0.00892 [0.00012, 0.01772]   std.effect = 0.169
```

---

## Figures Generated

### Figure 1: Permutation Distribution
- **Type**: Histogram
- **Shows**: Null distribution of F-statistics from permutation test
- **Key**: Red dashed line marks observed F-statistic
- **Interpretation**: If observed F is in far right tail, effect is significant

### Figure 2: Beta Distribution
- **Type**: Histogram
- **Shows**: Distribution of raw beta coefficients across all ROIs
- **Interpretation**: 
  - Center location indicates average effect direction
  - Spread indicates variability across ROIs

### Figure 3: Standardized Effects Distribution
- **Type**: Histogram  
- **Shows**: Distribution of standardized effect sizes
- **Interpretation**: Compare against Cohen's benchmarks (0.2, 0.5, 0.8)

### Figure 4: Forest Plot
- **Type**: Error bar plot (horizontal)
- **Shows**: Beta estimates with 95% confidence intervals for each ROI
- **Key**: Vertical line at zero; CIs not crossing zero are significant
- **Use**: Publication-ready visualization of all effects

---

## Results Table Structure

The `resultsTable` variable contains:

| Column | Description | Example |
|--------|-------------|---------|
| ROI | Region of interest name | 'FA_corpus_callosum' |
| Beta | Raw beta coefficient | 0.01234 |
| CI_Lower | Lower 95% confidence bound | 0.00456 |
| CI_Upper | Upper 95% confidence bound | 0.02012 |
| Standardized_Effect | Cohen's d-like effect size | 0.234 |
| F_statistic | F-test statistic | 5.234 |
| p_value | Raw p-value | 0.0231 |
| pFDR | FDR-corrected p-value | 0.0462 |

---

## Interpretation Guide

### Confidence Intervals
- **Excludes 0**: Effect is statistically significant (p < 0.05)
- **Includes 0**: Effect is not statistically significant
- **Width**: Narrower = more precise estimate
- **All positive** or **all negative**: Consistent effect direction

### Standardized Effects (Cohen's d interpretation)
- **|d| < 0.2**: Negligible to small effect
- **|d| ≈ 0.2**: Small effect
- **|d| ≈ 0.5**: Medium effect
- **|d| ≈ 0.8**: Large effect
- **|d| > 1.0**: Very large effect

### Permutation Test
- **p < 0.05**: Significant global effect (robust finding)
- **p close to parametric p**: Assumptions well-met
- **p > parametric p**: More conservative (possibly better estimate)
- **p << parametric p**: Unusual; check for issues

---

## Customization Options

### Adjust Number of Permutations
```matlab
% Line 115 in the script:
nPerm = 1000;  % Change this value

% Recommended:
% - Testing: 100-500
% - Standard analysis: 1000
% - Publication: 5000-10000
```

### Change Confidence Level
```matlab
% Line 166 in the script:
alpha = 0.05;  % Change for different CI (e.g., 0.01 for 99% CI)
```

### Modify FDR Threshold
```matlab
% Line 270 in the script:
alpha = 0.05;  % Change FDR threshold
```

---

## Common Questions

**Q: Why do permutation and parametric p-values differ?**
A: Permutation test doesn't assume normality or specific distributions. Differences suggest potential assumption violations in parametric test.

**Q: Should I use beta or standardized effect?**
A: 
- **Beta**: When original scale is meaningful (e.g., FA units)
- **Standardized**: For comparison across studies or ROIs with different scales

**Q: How many permutations do I need?**
A:
- Minimum: 1,000 for p-value precision to ±0.01
- Better: 5,000 for p-value precision to ±0.004
- Publication: 10,000 for high precision

**Q: What if no ROIs survive FDR correction?**
A: This is possible with modest effects. Consider:
- Report uncorrected p-values with caution
- Focus on global effect if significant
- Report effect sizes regardless of significance

**Q: Can I run this on multiple outcome variables?**
A: Yes, but modify the script to loop over different FA measures or other DTI metrics.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Interaction term not found" | Expected for reference ROI | Normal; can be ignored |
| Permutation p = NaN | All permutations failed | Check data quality; reduce model complexity |
| Script runs very slowly | Large dataset | Reduce nPerm or use parallel computing |
| Memory error | Too many observations | Process ROIs in batches |
| Convergence warnings | Model complexity | Simplify random effects or check for collinearity |

---

## Citation

If you use this enhanced script in your research, please consider citing:

```
Enhanced mixed-effects analysis script with confidence intervals, 
standardized effects, and permutation testing for neuroimaging data.
GitHub: Tinadamkristensen/SOMaps (2026)
```

---

## Support

For questions, issues, or suggestions:
1. Open an issue on GitHub
2. Check the detailed README (FISHOIL_ANALYSIS_README.md)
3. Review changes summary (CHANGES_SUMMARY.md)

---

**Last Updated**: January 2026
**MATLAB Version**: Tested on 2023b
**Script Version**: 1.0
