# Fix for Correlation Scatterplot Mismatch

## Problem Description

The original MATLAB script had a mismatch between the correlation values shown in the correlation matrix heatmap and the values displayed on the scatterplot.

## Root Cause

The issue was caused by using **different correlation methods** in two parts of the code:

1. **Correlation Matrix (Lines ~130-140)**: Used `partialcorr()` to compute partial correlations between ROI and cognitive variables, controlling for age and sex as covariates.

2. **Scatterplot (Original Line ~240)**: Used simple Pearson correlation `corr(X_res, Y_res)` on the residualized data, which does NOT produce the same result as `partialcorr()`.

## The Fix

**Changed line ~240 from:**
```matlab
[r,p] = corr(X_res, Y_res);
```

**To:**
```matlab
[r, p] = partialcorr(X, Y, C);
```

## Why This Works

### Understanding Partial Correlation

When computing partial correlations controlling for covariates:

- `partialcorr(X, Y, C)` computes the correlation between X and Y while statistically controlling for the covariates C
- This is equivalent to correlating the residuals of X and Y after regressing out C, **but the mathematical computation differs slightly from simple Pearson correlation on residuals**

### Key Insight

While residualizing and then computing Pearson correlation is conceptually similar to partial correlation, `partialcorr()` uses a specific mathematical formula:

```
r_partial = (r_XY - r_XC * r_YC) / sqrt((1 - r_XC²) * (1 - r_YC²))
```

This can produce slightly different numerical results than `corr(X_res, Y_res)` due to:
1. Numerical precision differences
2. Different handling of degrees of freedom
3. Different p-value calculations

## Solution Summary

To ensure consistency between the heatmap and scatterplot:

1. **Use the same method**: Apply `partialcorr(X, Y, C)` directly on the original (non-residualized) data with covariates
2. **Keep visualization on residuals**: Continue plotting X_res vs Y_res to show the partial relationship visually
3. **Display matching statistics**: Show the rho and p-value computed by `partialcorr()`, which will now match the correlation matrix

## Implementation

The fixed script (`correlation_analysis_fixed.m`) now:
- Computes partial correlation using `partialcorr(X, Y, C)` 
- Displays rho and p-values that match the correlation matrix exactly
- Maintains the residual-based scatterplot for visual interpretation
- Ensures statistical consistency throughout the analysis

## Usage

Replace your original scatterplot section (starting around line ~200) with the corrected version from `correlation_analysis_fixed.m`.

The key change is on the line where correlation statistics are computed:

```matlab
% FIX: Use partialcorr to match the correlation matrix calculation
% This computes the partial correlation controlling for age and sex
[r, p] = partialcorr(X, Y, C);
```

## Verification

To verify the fix works:

1. Run the corrected script
2. Find the correlation value in the heatmap for your chosen ROI and cognitive variable
3. Check the scatterplot - the displayed rho and p-value should now match the heatmap value exactly
