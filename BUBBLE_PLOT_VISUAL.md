# Bubble Plot - Visual Example

## What the Bubble Plot Looks Like

```
                    Cognitive Measures
                    ├─────┬─────┬─────┬─────┐
                    │ Cog1│ Cog2│ Cog3│ Cog4│
    ┌───────────────┼─────┼─────┼─────┼─────┤
R   │ ROI_1         │  ●  │  ◉  │  ○  │  ●  │
O   │               │ FDR │ Raw │ n.s.│ FDR │
I   ├───────────────┼─────┼─────┼─────┼─────┤
s   │ ROI_2         │  ◉  │  ○  │  ●  │  ○  │
    │               │ Raw │ n.s.│ FDR │ n.s.│
    ├───────────────┼─────┼─────┼─────┼─────┤
    │ ROI_3         │  ○  │  ●  │  ◉  │  ●  │
    │               │ n.s.│ FDR │ Raw │ FDR │
    └───────────────┴─────┴─────┴─────┴─────┘

Legend:
  ● = Large, vivid circle (FDR-corrected significant, strong effect)
  ◉ = Medium, semi-transparent (Raw p-value significant, moderate effect)
  ○ = Small, very faint (Non-significant, weak effect)
```

## Color Scheme

The circles use a **turbo colormap** gradient:

```
Negative Correlations        Zero        Positive Correlations
(Cool colors: Blue/Purple)              (Warm colors: Red/Yellow/Orange)
    ◄────────────────────────0───────────────────────►
   -0.5                               +0.5

Example colors:
  🔵 Strong negative (ρ ≈ -0.5)
  🟦 Moderate negative (ρ ≈ -0.25)
  🟩 Near zero (ρ ≈ 0)
  🟨 Moderate positive (ρ ≈ +0.25)
  🔴 Strong positive (ρ ≈ +0.5)
```

## Size and Transparency Examples

### FDR-Corrected Significant (q < 0.05)
```
Effect Size:    Small           Medium          Large
                 ●               ●●●             ●●●●●
                (15%)           (30%)           (50%)
Transparency:   ████████████████████████████████████  (100% opacity)
```

### Raw Significant (p < 0.05)
```
Effect Size:    Small           Medium          Large
                 ●               ●●●             ●●●●●
                (15%)           (30%)           (50%)
Transparency:   ████████░░░░░░░░████████░░░░░░░░  (50% opacity)
```

### Non-Significant (p ≥ 0.05)
```
Effect Size:    Small           Medium          Large
                 ●               ●●●             ●●●●●
                (15%)           (30%)           (50%)
Transparency:   ███░░░░░░░░░░░░░███░░░░░░░░░░░░░  (15% opacity)
```

## Complete Visual Guide

### Reading the Plot

1. **Find correlations of interest**: Look for large circles
2. **Check significance**: Vivid colors = FDR-corrected, faint = not significant
3. **Assess direction**: Red/warm = positive, blue/cool = negative
4. **Estimate effect size**: Larger circle = stronger correlation

### Quick Identification

**What to look for first:**
- ✨ **Large, vivid red/blue circles** = Strong, FDR-significant correlations (top priority)
- 📊 **Medium, semi-transparent circles** = Moderate correlations with raw significance
- 👻 **Small, very faint circles** = Weak, non-significant relationships (can ignore)

### Example Scenarios

**Scenario 1: Strong Positive FDR-Significant**
```
Circle: ●●●●● (large)
Color:  🔴 (vivid red)
Alpha:  ████████████ (100%)
→ ρ ≈ +0.45, q < 0.05
```

**Scenario 2: Moderate Negative Raw-Significant**
```
Circle: ●●● (medium)
Color:  🔵 (semi-transparent blue)
Alpha:  ██████░░░░░░ (50%)
→ ρ ≈ -0.25, p < 0.05, q > 0.05
```

**Scenario 3: Weak Non-Significant**
```
Circle: ● (small)
Color:  🟩 (very faint green)
Alpha:  ██░░░░░░░░░░ (15%)
→ ρ ≈ 0.10, p > 0.05
```

## Advantages for Publication

1. **Information-Dense**: Three data dimensions in one plot
2. **Visually Intuitive**: Size and brightness naturally communicate importance
3. **Statistical Rigor**: Shows FDR correction, not just raw p-values
4. **Professional**: Clean, modern appearance suitable for high-impact journals
5. **Accessible**: Easy to understand without detailed statistical knowledge

## Comparison to Traditional Heatmap

| Feature | Traditional Heatmap | Bubble Plot |
|---------|-------------------|-------------|
| Effect Size | Color only | Circle size |
| Directionality | Color | Color |
| Significance | Stars overlay | Transparency |
| FDR vs Raw p | Not distinguished | Clear distinction |
| Visual Clutter | Stars can obscure | Clean appearance |
| At-a-glance | Moderate | Excellent |

The bubble plot provides richer information while maintaining visual clarity.
