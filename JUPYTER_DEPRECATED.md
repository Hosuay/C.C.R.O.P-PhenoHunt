# ⚠️ DEPRECATION NOTICE: Jupyter Notebook Interface

**Status**: DEPRECATED as of Version 3.0.0
**Date**: 2025-10-23
**Replacement**: [CLI Interface](MIGRATION_GUIDE.md)

## What Happened?

The PhenoHunter Jupyter notebook interface (`PhenoHunter.ipynb`) has been **deprecated** and replaced with a professional command-line interface (CLI).

## Why Was This Changed?

The Jupyter notebook had several limitations:

1. **Not Production-Ready**: Difficult to use in server environments
2. **Hard to Automate**: Couldn't easily integrate into pipelines
3. **Version Control Issues**: Notebooks don't work well with Git
4. **Performance Overhead**: Browser-based execution is slower
5. **Limited Scalability**: Can't handle batch processing efficiently

The new CLI addresses all these issues while maintaining the same core functionality.

## What Should I Do?

### Option 1: Use the CLI (Recommended)

Install PhenoHunt as a CLI tool:

```bash
pip install -e .
```

Run commands from your terminal:

```bash
# Train models
phenohunt train --data strains.csv --epochs 369

# Generate hybrid
phenohunt cross --parent1 "Blue Dream" --parent2 "OG Kush" --output f1.csv
```

See the [Migration Guide](MIGRATION_GUIDE.md) for detailed instructions.

### Option 2: Use the Python API

If you prefer programmatic access, the Python API is still available:

```python
from phenohunter_scientific import create_phenohunter
import pandas as pd

ph = create_phenohunter()
df = pd.read_csv('strains.csv')
ph.load_strain_database(df)
ph.train_vae(epochs=369)

f1_result = ph.generate_f1_hybrid(
    parent1_name='Blue Dream',
    parent2_name='OG Kush'
)
```

You can use this in your own Jupyter notebooks, Python scripts, or applications.

### Option 3: Keep Using the Old Notebook (Not Recommended)

The old notebook is still in the repository at `PhenoHunter.ipynb`, but:

- ❌ Will NOT receive updates
- ❌ Will NOT receive bug fixes
- ❌ May break with new dependencies
- ❌ Missing new features added to CLI

**We strongly recommend migrating to the CLI or Python API.**

## Timeline

- **Version 2.x**: Jupyter notebook was primary interface
- **Version 3.0.0**: CLI introduced, notebook deprecated
- **Version 4.0.0+**: Notebook may be removed from repository

## Features in CLI Not in Notebook

The CLI version includes several enhancements:

- ✅ **Faster Execution**: No browser overhead
- ✅ **Batch Processing**: Process multiple crosses at once
- ✅ **Shell Scripting**: Integrate into bash/zsh scripts
- ✅ **CI/CD Integration**: Use in automated pipelines
- ✅ **Remote Execution**: SSH-friendly interface
- ✅ **Better Error Messages**: More informative output
- ✅ **Progress Tracking**: See what's happening in real-time

## Need Help?

- Read the [Migration Guide](MIGRATION_GUIDE.md)
- Check [README.md](README.md) for CLI documentation
- Run `phenohunt --help` for command help
- Open an [issue](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt/issues) if you encounter problems

## Feedback

We welcome feedback on this decision. If you have strong reasons for needing the Jupyter interface, please open an issue and let us know.

---

Thank you for understanding as we move PhenoHunt toward a more professional, production-ready architecture!
