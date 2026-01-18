# Bubble Plot Visualization Guide

## Overview

The correlation matrix is now visualized as a **bubble plot** instead of a traditional heatmap. This provides more information at a glance by encoding three dimensions of data:

1. **Circle Size** → Effect size (correlation magnitude)
2. **Color** → Directionality (positive/negative correlation)
3. **Transparency** → Statistical significance level

## Visual Encoding

### Circle Size (Effect Size)
- **Larger circles** = Stronger correlations (higher |ρ|)
- **Smaller circles** = Weaker correlations (lower |ρ|)
- Size is proportional to the absolute value of the correlation coefficient

### Color (Directionality)
Uses the **turbo colormap** to represent correlation direction:
- **Red/warm colors** = Positive correlations (ρ > 0)
- **Blue/cool colors** = Negative correlations (ρ < 0)
- **Mid-range colors** = Near-zero correlations

Color scale ranges from -0.5 to +0.5 for correlation values.

### Transparency (Significance Level)

Three levels of transparency indicate statistical significance:

| Significance Level | Alpha Value | Visual Appearance | Meaning |
|-------------------|-------------|-------------------|---------|
| **FDR-corrected** | 1.0 (100%) | Vivid, solid color | q < 0.05 (Benjamini-Hochberg) |
| **Raw p-value** | 0.5 (50%) | Semi-transparent | p < 0.05 (uncorrected) |
| **Non-significant** | 0.15 (15%) | Very faint | p ≥ 0.05 |

## Advantages Over Traditional Heatmap

1. **Effect Size Visibility**: Immediately see which correlations are strong vs. weak
2. **Multiple Significance Levels**: Distinguish between FDR-corrected and raw significance
3. **Cleaner Display**: Non-significant correlations fade into background
4. **Publication Ready**: Professional appearance suitable for manuscripts

## FDR Correction

The script applies **Benjamini-Hochberg FDR correction** across all ROI × cognitive variable correlations using MATLAB's `mafdr()` function:

```matlab
pvals_vec = pMatrix(:);                             % vectorize all p-values
qvals_vec = mafdr(pvals_vec, 'BHFDR', true);        % Benjamini–Hochberg
qMatrix   = reshape(qvals_vec, size(pMatrix));      % reshape back
```

This controls for multiple comparisons across the entire correlation matrix.

## Legend

The plot includes an inline legend showing example circles at three transparency levels:
- **q < 0.05 (FDR)** - Full opacity
- **p < 0.05 (raw)** - 50% opacity  
- **n.s.** - 15% opacity (non-significant)

## Code Location

The bubble plot code is in `correlation_analysis_fixed.m` starting at line ~146:
```matlab
%% Bubble plot: Correlation matrix with effect size and significance
```

## Customization Options

You can adjust these parameters in the code:

- `max_bubble_size = 1000;` - Maximum circle size in points
- Transparency levels (lines 196-204):
  - FDR: `alpha_val = 1.0;`
  - Raw: `alpha_val = 0.5;`
  - Non-sig: `alpha_val = 0.15;`
- Color scale: `caxis([-0.5 0.5]);` - Adjust range for correlation values
- Figure size: `figure('Position', [100, 100, 1200, 800]);`

## Example Interpretation

**Large, vivid red circle**: Strong positive correlation with FDR-corrected significance
**Medium, semi-transparent blue circle**: Moderate negative correlation with raw significance  
**Small, very faint circle**: Weak correlation, not statistically significant

This visualization makes it easy to identify the most important relationships in your data at a glance.
