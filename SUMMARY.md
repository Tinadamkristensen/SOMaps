# MATLAB Correlation Analysis Fix - Summary

## Files in This Repository

### 1. `correlation_analysis_fixed.m`
The complete corrected MATLAB script with the fix applied. This is a ready-to-use version of your script with the correlation mismatch resolved.

### 2. `CORRELATION_FIX_README.md`
Comprehensive documentation that explains:
- The problem in detail
- Root cause analysis
- Why the fix works
- Mathematical explanation of partial correlation
- Verification steps

### 3. `QUICK_FIX.md`
A quick reference guide showing:
- The exact line that needs to be changed
- Before/after comparison
- Complete corrected scatterplot section for easy copy-paste

## The Problem
The scatterplot displayed different rho and p-values than the correlation matrix heatmap for the same variable pair.

## The Solution
**Line 247** in `correlation_analysis_fixed.m`:
```matlab
[r, p] = partialcorr(X, Y, C);
```

**Previously was:**
```matlab
[r,p] = corr(X_res, Y_res);
```

## Why This Matters
- **Correlation matrix**: Uses `partialcorr()` to control for age and sex
- **Scatterplot (before fix)**: Used `corr()` on residuals, giving inconsistent results
- **Scatterplot (after fix)**: Uses `partialcorr()`, matching the matrix exactly

## How to Apply the Fix

### Option 1: Use the Complete Fixed Script
1. Replace your script with `correlation_analysis_fixed.m`
2. Update the file path on line 3 to point to your data
3. Run the script

### Option 2: Apply Minimal Change to Your Existing Script
1. Open your current script
2. Find the line: `[r,p] = corr(X_res, Y_res);` (around line 240)
3. Replace it with: `[r, p] = partialcorr(X, Y, C);`
4. Save and run

## Verification
After applying the fix:
1. Run your script
2. Note the correlation value in the heatmap for your chosen ROI × cognitive variable
3. Generate the scatterplot
4. The displayed rho and p-value should now **exactly match** the heatmap value

## Technical Details
See `CORRELATION_FIX_README.md` for detailed technical explanation of why `partialcorr()` and `corr()` on residuals produce different results.

## Questions?
The fix ensures statistical consistency by using the same correlation method (`partialcorr`) throughout your analysis, which is the standard approach for partial correlation analysis with covariates.
