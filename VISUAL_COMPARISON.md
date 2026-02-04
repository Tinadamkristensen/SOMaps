# Visual Comparison: Before vs After Fix

## The Issue Visualized

### Before Fix ❌
```
Correlation Matrix Heatmap          Scatterplot
┌─────────────────────┐            ┌─────────────────────┐
│ FA_Splenium ×       │            │                     │
│ CBCL_act            │            │  ρ = 0.23, p=0.045  │ ← Different!
│                     │            │                     │
│ ρ = 0.28, p=0.012   │            │                     │
└─────────────────────┘            └─────────────────────┘
    Uses partialcorr()                 Uses corr(residuals)
```

### After Fix ✅
```
Correlation Matrix Heatmap          Scatterplot
┌─────────────────────┐            ┌─────────────────────┐
│ FA_Splenium ×       │            │                     │
│ CBCL_act            │            │  ρ = 0.28, p=0.012  │ ← MATCHES!
│                     │            │                     │
│ ρ = 0.28, p=0.012   │            │                     │
└─────────────────────┘            └─────────────────────┘
    Uses partialcorr()                 Uses partialcorr()
```

## Code Comparison

### Original Code (Incorrect)
```matlab
% In correlation matrix section (around line 130)
[rho, pval] = partialcorr( ...
    roiData(validRows), ...
    cognitiveData(validRows,:), ...
    covariates(validRows,:) );

% ...later in scatterplot section (around line 240)
Y_res = Y - C * (C \ Y);
X_res = X - C * (C \ X);
[r,p] = corr(X_res, Y_res);  ← INCONSISTENT METHOD
```

### Fixed Code (Correct)
```matlab
% In correlation matrix section (around line 130)
[rho, pval] = partialcorr( ...
    roiData(validRows), ...
    cognitiveData(validRows,:), ...
    covariates(validRows,:) );

% ...later in scatterplot section (around line 247)
Y_res = Y - C * (C \ Y);
X_res = X - C * (C \ X);
[r, p] = partialcorr(X, Y, C);  ← CONSISTENT METHOD
```

## Why Are They Different?

### Method 1: corr(residuals) - What the original code did
```matlab
Y_res = Y - C * (C \ Y);
X_res = X - C * (C \ X);
r = corr(X_res, Y_res);
```
- Manually residualizes X and Y
- Then computes Pearson correlation
- **Not mathematically equivalent to partialcorr()**

### Method 2: partialcorr() - What the fix uses
```matlab
r = partialcorr(X, Y, C);
```
- Uses the partial correlation formula directly
- Properly accounts for covariate relationships
- **Mathematically correct for partial correlation**

## Mathematical Difference

**Partial correlation formula:**
```
           r_XY - r_XC * r_YC
r_partial = ─────────────────────────────────
           √(1 - r_XC²) × √(1 - r_YC²)
```

This is NOT the same as:
```
r_residuals = corr(X - X̂_from_C, Y - Ŷ_from_C)
```

While conceptually similar, they produce different numerical results!

## Key Takeaway

✅ **Always use the same correlation method throughout your analysis**

If you compute partial correlations in your matrix using `partialcorr()`, you must use `partialcorr()` for individual comparisons too, not `corr()` on residuals.

## Testing Your Fix

1. Run the corrected script
2. Note the value in the heatmap: e.g., ρ = 0.28, p = 0.012
3. Generate scatterplot for the same variable pair
4. Check the displayed text: Should show ρ = 0.28, p = 0.012
5. **Values should match exactly!**
