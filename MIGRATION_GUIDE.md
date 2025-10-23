# Migration Guide: Jupyter Notebook to CLI

## Overview

PhenoHunter has been converted from a Jupyter notebook-based tool to a professional Python CLI application. This guide will help you transition from the old notebook interface to the new command-line interface.

## Why the Change?

The CLI version offers several advantages over the Jupyter notebook:

- **Better Performance**: No browser overhead, runs faster
- **Automation**: Easy to integrate into scripts and pipelines
- **Server-Friendly**: Run on remote servers without GUI
- **Version Control**: Easier to track changes in plain Python files
- **Professional**: Standard tool for production environments
- **Reproducible**: Scripts can be version-controlled and shared

## Installation

### Old Way (Notebook)
```python
# In a Jupyter cell
!pip install -r requirements.txt
# Then run notebook cells
```

### New Way (CLI)
```bash
# Install as a package
pip install -e .

# Or install from PyPI (when available)
pip install phenohunt
```

## Command Comparison

### Training Models

**Old (Notebook):**
```python
ph = create_phenohunter()
df = pd.read_csv('strains.csv')
ph.load_strain_database(df)
ph.train_vae(epochs=369)
ph.train_effect_predictors()
```

**New (CLI):**
```bash
phenohunt train --data strains.csv --epochs 369
```

### Generating F1 Hybrid

**Old (Notebook):**
```python
f1_result = ph.generate_f1_hybrid(
    parent1_name='Blue Dream',
    parent2_name='OG Kush',
    parent1_weight=0.6,
    n_samples=100
)
ph.visualize_breeding_result(f1_result)
```

**New (CLI):**
```bash
phenohunt cross \
    --data strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --ratio 0.6 \
    --output f1_hybrid.csv \
    --visualize
```

### Generating F2 Population

**Old (Notebook):**
```python
f2_population = ph.generate_f2_population(f1_result, n_offspring=10)
ranked = sorted(f2_population, key=lambda x: x.stability_score, reverse=True)
```

**New (CLI):**
```bash
phenohunt f2 \
    --data strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --count 10 \
    --trait "Analgesic" \
    --output f2_population.csv
```

### Backcrossing

**Old (Notebook):**
```python
bx1_result = ph.backcross(
    f1_result,
    parent_name='Blue Dream',
    backcross_generation=1
)
```

**New (CLI):**
```bash
phenohunt backcross \
    --data strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --backcross-to "Blue Dream" \
    --generation 1 \
    --output bx1.csv
```

## Feature Comparison

| Feature | Notebook | CLI | Notes |
|---------|----------|-----|-------|
| Train Models | ✅ | ✅ | CLI is faster |
| F1 Hybrids | ✅ | ✅ | Same functionality |
| F2 Population | ✅ | ✅ | CLI adds ranking |
| Backcross | ✅ | ✅ | Same functionality |
| Visualizations | ✅ | ✅ | CLI saves to files |
| Interactive Widgets | ✅ | ❌ | CLI uses flags/options |
| Batch Processing | ❌ | ✅ | CLI is scriptable |
| Remote Execution | ⚠️ | ✅ | CLI is better |
| COA Import | ✅ | ✅ | CLI uses import command |

## Automation Examples

One of the biggest advantages of the CLI is automation. Here are some examples:

### Batch Processing Multiple Crosses

```bash
#!/bin/bash
# Generate multiple crosses automatically

parents=(
    "Blue Dream,OG Kush"
    "Sour Diesel,Girl Scout Cookies"
    "Wedding Cake,Gelato"
)

for pair in "${parents[@]}"; do
    IFS=',' read -r p1 p2 <<< "$pair"
    echo "Processing: $p1 x $p2"

    phenohunt cross \
        --data strains.csv \
        --parent1 "$p1" \
        --parent2 "$p2" \
        --output "results/${p1}_x_${p2}.csv"
done
```

### Scheduled Training Pipeline

```bash
#!/bin/bash
# Cron job to retrain models daily with new data

# Download latest strain data
curl -o latest_strains.csv https://your-api.com/strains

# Train models
phenohunt train \
    --data latest_strains.csv \
    --epochs 369 \
    --save models/latest.pth

# Send notification
echo "Training complete" | mail -s "PhenoHunt Update" admin@example.com
```

### CI/CD Integration

```yaml
# .github/workflows/validate-crosses.yml
name: Validate Breeding Predictions

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install PhenoHunt
        run: pip install -e .
      - name: Run validation
        run: |
          phenohunt cross \
            --data tests/test_strains.csv \
            --parent1 "Test Strain A" \
            --parent2 "Test Strain B" \
            --output results.csv
```

## Python API Still Available

If you still need programmatic access (e.g., for custom scripts), the Python API is still available:

```python
from phenohunter_scientific import create_phenohunter
import pandas as pd

# Same as before!
ph = create_phenohunter()
df = pd.read_csv('strains.csv')
ph.load_strain_database(df)
ph.train_vae(epochs=369)

# All the same methods work
f1_result = ph.generate_f1_hybrid(
    parent1_name='Blue Dream',
    parent2_name='OG Kush'
)
```

## What About the Notebook?

The Jupyter notebook (`PhenoHunter.ipynb`) is now **deprecated** but remains in the repository for reference. It will not receive updates.

### If You Still Need Notebooks

You can create your own Jupyter notebooks that use the Python API:

```python
# In your notebook
from phenohunter_scientific import create_phenohunter
import pandas as pd

ph = create_phenohunter()
# Use as before...
```

## Getting Help

- **CLI Help**: `phenohunt --help`
- **Command Help**: `phenohunt cross --help`
- **Documentation**: See [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt/issues)

## Quick Start

Try the example workflow to get familiar with the CLI:

```bash
# Generate sample data
python examples/create_sample_data.py

# Run example workflow
bash examples/example_cli_workflow.sh
```

## Feedback

We welcome feedback on the new CLI interface! Please open an issue if you encounter any problems or have suggestions for improvements.

---

**Migration Date**: 2025-10-23
**Version**: 3.0.0
**Status**: Stable
