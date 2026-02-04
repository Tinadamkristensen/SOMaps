# Cognitive profiles across the psychosis continuum

This is a code base to reproduce our paper on _Cognitive profiles across the psychosis continuum_, available https://www.sciencedirect.com/science/article/pii/S0165178124004530.

## MATLAB Correlation Analysis Fix

This repository also includes a fix for a MATLAB correlation analysis script where scatterplot correlation values didn't match the correlation matrix. The script now features an advanced **bubble plot visualization** for the correlation matrix. See:

- **[SUMMARY.md](SUMMARY.md)** - Overview of the fix and bubble plot visualization
- **[BUBBLE_PLOT_GUIDE.md](BUBBLE_PLOT_GUIDE.md)** - Guide to the bubble plot visualization
- **[BUBBLE_PLOT_VISUAL.md](BUBBLE_PLOT_VISUAL.md)** - Visual examples and interpretation
- **[QUICK_FIX.md](QUICK_FIX.md)** - Quick reference for applying the fix
- **[CORRELATION_FIX_README.md](CORRELATION_FIX_README.md)** - Detailed technical explanation
- **[MOD_SEQ_RAW_EXPLANATION.md](MOD_SEQ_RAW_EXPLANATION.md)** - Explanation of mod_seq_raw transformation in moderation analysis

### MATLAB Scripts
- **[spectrum_correlation_analysis.m](spectrum_correlation_analysis.m)** - Complete script for SPECTRUM study data (FD_ ROIs, COG_ variables)
- **[correlation_analysis_fixed.m](correlation_analysis_fixed.m)** - Original example script (FA_ ROIs)

### Bubble Plot Features
- **Circle size** = Effect size (correlation magnitude)
- **Color** = Directionality (positive/negative)
- **Transparency** = Significance level (FDR-corrected, raw p-value, or non-significant)
- **FDR correction** using Benjamini-Hochberg method across all correlations
