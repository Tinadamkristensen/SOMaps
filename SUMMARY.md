# MATLAB Correlation Analysis Fix - Summary

## Files in This Repository

### MATLAB Scripts

#### 1. `spectrum_correlation_analysis.m` (RECOMMENDED)
The complete MATLAB script for SPECTRUM study data with bubble plot visualization. Features:
- Uses **FD_ prefix** for ROI variables (not FA_)
- Uses **COG_ prefix** for cognitive variables (DART, Verbal_Memory, etc.)
- **Bubble plot visualization** for the correlation matrix
- **FDR correction** using Benjamini-Hochberg method
- **Significant Correlations Table** export to CSV
- **Automatic scatterplot generation** for all FDR-significant correlations
- Consistent use of partial correlations throughout

#### 2. `correlation_analysis_fixed.m`
Original example script with FA_ ROI variables. Includes:
- **Bubble plot visualization** for the correlation matrix showing effect size, directionality, and significance levels
- **FDR correction** using Benjamini-Hochberg method
- Consistent use of partial correlations throughout

### Documentation Files

#### 3. `CORRELATION_FIX_README.md`
Comprehensive documentation that explains:
- The problem in detail
- Root cause analysis
- Why the fix works
- Mathematical explanation of partial correlation
- Verification steps

#### 4. `QUICK_FIX.md`
A quick reference guide showing:
- The exact line that needs to be changed
- Before/after comparison
- Complete corrected scatterplot section for easy copy-paste

## The Problem
The scatterplot displayed different rho and p-values than the correlation matrix heatmap for the same variable pair.

## The Solution
**Line 255** in `correlation_analysis_fixed.m`:
```matlab
[r, p] = partialcorr(X, Y, C);
```

**Previously was:**
```matlab
[r,p] = corr(X_res, Y_res);
```

## Correlation Matrix Visualization
The script now uses a **bubble plot** instead of a traditional heatmap:
- **Circle size**: Represents effect size (absolute correlation magnitude)
- **Color**: Represents directionality (positive/negative correlation) using turbo colormap
- **Transparency**: Represents significance level:
  - FDR-corrected significant (q < 0.05): Full opacity (vivid)
  - Raw significant (p < 0.05): 50% opacity (intermediate)
  - Non-significant: 15% opacity (very transparent)

## Why This Matters
- **Correlation matrix**: Uses `partialcorr()` to control for age and sex
- **Scatterplot (before fix)**: Used `corr()` on residuals, giving inconsistent results
- **Scatterplot (after fix)**: Uses `partialcorr()`, matching the matrix exactly

## How to Apply the Fix

### Option 1: Use the Complete Fixed Script
1. Replace your script with `correlation_analysis_fixed.m`
2. Update the file path on line 9 to point to your data
3. Run the script

### Option 2: Apply Minimal Change to Your Existing Script
1. Open your current script
2. Find the line: `[r,p] = corr(X_res, Y_res);` (around line 240)
3. Replace it with: `[r, p] = partialcorr(X, Y, C);`
4. Save and run

## Verification
After applying the fix:
1. Run your script
2. Note the correlation value in the bubble plot for your chosen ROI × cognitive variable
3. Generate the scatterplot
4. The displayed rho and p-value should now **exactly match** the bubble plot value

## Technical Details
See `CORRELATION_FIX_README.md` for detailed technical explanation of why `partialcorr()` and `corr()` on residuals produce different results.

## Questions?
The fix ensures statistical consistency by using the same correlation method (`partialcorr`) throughout your analysis, which is the standard approach for partial correlation analysis with covariates.
