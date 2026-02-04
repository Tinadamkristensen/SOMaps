# mod_seq_raw Quick Reference Card

## TL;DR

```matlab
mod_seq_raw = linspace(min(mod_raw), max(mod_raw), 100)';
```

**What it is**: A sequence of 100 evenly-spaced values from min to max of moderator  
**What it's NOT**: A transformation of your data  
**Purpose**: Create smooth prediction lines in plots

## One-Line Summary

Creates 100 evenly-spaced x-axis values for plotting smooth model predictions across the moderator's range.

## Function Breakdown

| Component | What it does |
|-----------|--------------|
| `linspace(...)` | Generates evenly-spaced sequence |
| `min(mod_raw)` | Starting point of sequence |
| `max(mod_raw)` | Ending point of sequence |
| `100` | Number of points to generate |
| `'` | Transpose to column vector |

## Example

```matlab
% Your data (5 irregular points)
mod_raw = [1.2, 3.5, 2.1, 8.9, 5.3]'

% Generated sequence (100 even points)
mod_seq_raw = linspace(1.2, 8.9, 100)'
% Result: [1.2, 1.28, 1.36, 1.44, ..., 8.82, 8.9]
```

## Common Usage Pattern

```matlab
% 1. Extract raw data
mod_raw = db1.(moderatorVar);

% 2. Create plotting sequence
mod_seq_raw = linspace(min(mod_raw), max(mod_raw), 100)';

% 3. Center for model
mod_seq_c = mod_seq_raw - mean(mod_raw, 'omitnan');

% 4. Get predictions
[FA_pred, FA_CI] = predict(model, ...);

% 5. Plot smooth line with raw x-axis
plot(mod_seq_raw, FA_pred);

% 6. Overlay actual data
scatter(mod_raw, actual_values);
```

## Why 100 Points?

- Provides smooth visual curve
- Standard choice for plotting
- Can be changed (e.g., 50 or 200) if needed
- Trade-off: more points = smoother but slower computation

## Key Insight

Think of it as creating a "plotting grid" that spans your data's range, allowing you to visualize the model's predictions as a smooth curve rather than a jagged line connecting only your actual data points.

## See Also

- **[MOD_SEQ_RAW_EXPLANATION.md](MOD_SEQ_RAW_EXPLANATION.md)** - Full technical explanation
- **[MOD_SEQ_RAW_VISUAL_GUIDE.md](MOD_SEQ_RAW_VISUAL_GUIDE.md)** - Visual diagrams and illustrations

## Common Misconceptions

❌ **WRONG**: "mod_seq_raw transforms my data"  
✅ **RIGHT**: "mod_seq_raw creates a new sequence for plotting"

❌ **WRONG**: "I should use mod_seq_raw instead of mod_raw"  
✅ **RIGHT**: "Use mod_seq_raw for prediction lines, mod_raw for data points"

❌ **WRONG**: "mod_seq_raw contains my actual measurements"  
✅ **RIGHT**: "mod_seq_raw is a generated sequence for visualization"

## Quick Comparison

```
mod_raw:     Your actual data points (scattered)
             ●    ●     ●        ●  ●

mod_seq_raw: Smooth sequence for plotting (even)
             ●●●●●●●●●●●●●●●●●●●●●●●●
             
Both span:   Same range [min, max]
```
