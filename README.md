# C.C.R.O.P-PhenoHunt
## Cannabis Computational Research & Optimization Platform
### Pheno-Hunting with Machine Learning & Sacred Geometry
#### **Now with Professional CLI Interface!** 🚀

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-3.0.0-green.svg)](CHANGELOG.md)
[![Sacred Geometry](https://img.shields.io/badge/alignment-369-purple.svg)](ARCHITECTURE.md#sacred-geometry-integration)
[![CLI](https://img.shields.io/badge/interface-CLI-brightgreen.svg)](MIGRATION_GUIDE.md)

> **⚠️ IMPORTANT:** As of v3.0.0, PhenoHunt is now primarily a **CLI application**. The Jupyter notebook interface is **deprecated**. See [Migration Guide](MIGRATION_GUIDE.md) for details.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Architecture](#architecture)
6. [Usage](#usage)
7. [Sacred Geometry Integration](#sacred-geometry-integration)
8. [Documentation](#documentation)
9. [Contributing](#contributing)
10. [Citation](#citation)
11. [License](#license)

---

## Overview

**C.C.R.O.P-PhenoHunt** is a comprehensive platform for cannabis strain analysis, breeding optimization, and phenotypic prediction using:

- **Machine Learning**: Variational Autoencoders (VAE), ensemble models, XAI
- **Sacred Geometry**: Harmonic alignment with 3-6-9 numerology
- **Multi-Language Architecture**: Python, C++, Rust, Julia (planned)
- **Scientific Rigor**: All predictions backed by peer-reviewed literature
- **Uncertainty Quantification**: Confidence intervals for all outputs

### Purpose

This tool provides a **reproducible, scientifically rigorous framework** for:

1. **Strain Analysis**: Chemical profiling (cannabinoids, terpenes, molecular descriptors)
2. **Effect Prediction**: Therapeutic outcome prediction with literature-backed coefficients
3. **Breeding Optimization**: F1, F2, and backcross hybrid generation with genetic simulation
4. **Phenotypic Analysis**: Image-based trichome density, bud structure, leaf morphology
5. **3D Visualization**: Interactive molecular structures and dashboards

### Disclaimer

⚠️ **FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY** ⚠️

This tool:
- Does NOT provide medical advice or clinical recommendations
- Does NOT provide legal cultivation guidance
- Generates computational hypotheses requiring laboratory validation
- Must be used in compliance with local laws and regulations

---

## Features

### Machine Learning & Modeling

✅ **Variational Autoencoder (VAE)**
- Probabilistic latent space with 27 dimensions (harmonic 3³)
- Uncertainty quantification via Monte Carlo sampling
- β-VAE formulation for controlled disentanglement
- 7-layer encoder/decoder architecture

✅ **Ensemble Effect Prediction**
- Logistic Regression + Random Forest + Gradient Boosting
- 6 therapeutic effects: Analgesic, Anxiolytic, Sedative, Anti-inflammatory, Neuroprotective, Appetite Stimulant
- Literature-backed coefficients with 15+ peer-reviewed citations
- Entourage effect modeling (compound interactions)

✅ **Multi-Generation Breeding**
- F1 hybrid generation with heterosis calculation
- F2 population with Mendelian segregation
- Backcross progressive introgression
- Genetic variance and stability scoring

✅ **Explainable AI (XAI) & Uncertainty Quantification**
- Feature importance visualization
- SHAP values for prediction interpretation
- Confidence intervals and conformal prediction
- Reliability diagrams and calibration metrics
- Brier score and Expected Calibration Error (ECE)

### Interoperability & Data Exchange

✅ **BrAPI v2.1 Compliance**
- Import/export BrAPI-compliant JSON for Breedbase integration
- Observation unit format with confidence intervals
- 9-field metadata schema for reproducibility
- Command-line and Python API interfaces
- Full integration with breeding management systems (BMS)

✅ **Containerized Phenomics Pipeline**
- PhytoOracle-inspired 3-stage pipeline (preprocess → extract → aggregate)
- Docker-based reproducible environments
- YAML configuration for flexible workflows
- High-throughput image analysis for trichome density, bud structure
- Automated quality control and validation

### Genomic Selection & GWAS

✅ **G-BLUP (Genomic Best Linear Unbiased Prediction)**
- VanRaden (2008) genomic relationship matrix (GRM)
- Mixed model equations for breeding value estimation
- 9-fold cross-validation (sacred geometry aligned)
- Predictive accuracy metrics (R², correlation, RMSE)
- Scalable to 10K+ markers

✅ **Genome-Wide Association Studies (GWAS)**
- PLINK integration with scipy fallback
- Linear and mixed-model association testing
- Manhattan plot data preparation
- MAF filtering and quality control

### Chemical Profiling

✅ **20 Compounds Tracked**
- **9 Cannabinoids**: THC, CBD, CBG, CBC, CBDA, THCV, CBN, Δ8-THC, THCA
- **11 Terpenes**: Myrcene, Limonene, Pinene, Linalool, Caryophyllene, Humulene, Terpinolene, Ocimene, Camphene, Bisabolol, β-Pinene

✅ **Molecular Descriptors** (RDKit Integration)
- Molecular Weight, LogP (Lipophilicity)
- H-Bond Donors/Acceptors
- Topological Polar Surface Area (TPSA)
- Rotatable Bonds, Aromatic Rings
- 33 primary descriptors (sacred geometry)

✅ **Evidence-Based Effect Mapping**
- Literature-backed effect coefficients with DOI references
- 9+ primary literature sources (Russo 2011, Blessing 2015, etc.)
- Compound-effect association with confidence levels
- Entourage effect aggregation across multiple compounds
- Automated evidence report generation

### Data Processing

✅ **5-Stage Preprocessing Pipeline**
1. **Acquisition**: Load raw data from CSV, API, web scraping
2. **Cleaning**: Remove duplicates, handle missing values
3. **Normalization**: Scale features to standard ranges
4. **Feature Extraction**: Derive ratios, embeddings, descriptors
5. **Validation**: Quality checks, outlier detection

✅ **Data Validation**
- Literature-based chemical range validation
- Isolation Forest outlier detection
- THC:CBD ratio consistency checks
- Automated quality control reports

✅ **Dataset Versioning & Experiment Tracking**
- SHA-256 cryptographic hashing for data integrity
- 9-field minimum metadata schema (sacred geometry)
- 27-field extended metadata for full reproducibility
- Git commit tracking and Docker image versioning
- Harmonic seed generation from experiment parameters

### Visualization & Dashboards

✅ **Interactive Visualizations**
- Chemical radar plots with error bars
- Effect prediction bar charts with confidence intervals
- Latent space 2D/3D embeddings
- Parent-offspring breeding trajectory comparisons

✅ **3D Molecular Visualization** (Planned)
- RDKit molecular structure generation
- py3Dmol interactive 3D rendering
- Rotation, zoom, feature highlighting
- Export to PNG, SVG, PDF

✅ **Interactive Dashboards** (Planned)
- Plotly Dash multi-panel interface
- Real-time metric updates
- Strain filtering by chemical profile
- Harmonic color schemes (sacred geometry aligned)

### Sacred Geometry Integration

✅ **Harmonic Numerology Alignment**
- **3**: Triadic Input → Transform → Output flow (BrAPI, Phenomics, Genomics)
- **5**: 5-stage preprocessing pipeline
- **7**: 7-layer neural network architectures
- **9**: 9-fold cross-validation, 9-field metadata schema
- **12**: 12-module repository structure
- **27**: 27 latent dimensions (3³), 27 feature embeddings
- **33**: 33 primary molecular descriptors
- **369**: Tesla's divine numbers - ultimate epoch count and harmonic seed

✅ **Harmonic Seeding System**
- Deterministic seed generation from experiment parameters
- SHA-256 hash-based reproducibility (NOT biological causation claims)
- Sacred number hyperparameter presets (27, 369, 999 epochs)
- Cosine annealing learning rate with 3-6-9 cycles
- Full integration with PyTorch, TensorFlow, NumPy

### Multi-Language Architecture (Planned)

✅ **Python** (Primary)
- ML pipelines, data science, visualization
- Current: 10,000+ lines of scientific code (v3.0.0)
- New modules: BrAPI, Phenomics, Genomics, XAI, Uncertainty, Experiments

✅ **C++** (Planned)
- High-performance phenotypic analysis
- GPU-accelerated image processing
- 10-100x speedup for computational tasks

✅ **Rust** (Planned)
- Memory-safe data integrity pipelines
- Concurrent processing with zero overhead
- Cryptographic hashing for versioning

✅ **Julia** (Planned)
- High-precision molecular descriptors
- Multi-objective genetic algorithms
- PK/PD pharmacological modeling

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA for GPU acceleration

### CLI Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/Hosuay/C.C.R.O.P-PhenoHunt.git
cd C.C.R.O.P-PhenoHunt

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install as a package
pip install -e .

# Verify installation
phenohunt --version
```

### Alternative: Python API Installation

If you want to use PhenoHunt programmatically in your own scripts:

```bash
# Install dependencies only
pip install -r requirements.txt

# Import in your Python code
from src.phenohunter_scientific import create_phenohunter
ph = create_phenohunter()
```

### For Development

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### Docker Installation

#### NEW v3.0.0: Phenomics Pipeline
```bash
# Build phenomics image
docker build -f docker/phenomics.Dockerfile -t crop-phenohunt/phenomics:latest .

# Run pipeline in container
docker run --rm \
  -v $(pwd)/data_examples:/workspace/data_examples \
  -v $(pwd)/data:/workspace/data \
  crop-phenohunt/phenomics:latest \
  python scripts/run_pipeline.py --config pipelines/phenomics_pipeline.yml

# Interactive shell
docker run -it --rm crop-phenohunt/phenomics:latest /bin/bash
```

#### Future: Full Platform
```bash
docker pull hosuay/crop-phenohunt:latest
docker run -p 8050:8050 hosuay/crop-phenohunt
```

---

## Quick Start

### CLI Quick Start (Recommended)

```bash
# 1. Create sample data
python examples/create_sample_data.py

# 2. Train models on your strain database
phenohunt train --data examples/sample_strains.csv --epochs 369

# 3. Generate F1 hybrid between two parents
phenohunt cross \
    --data examples/sample_strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --ratio 0.6 \
    --output f1_hybrid.csv

# 4. Generate F2 population
phenohunt f2 \
    --data examples/sample_strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --count 10 \
    --trait "Analgesic" \
    --output f2_population.csv

# 5. Run complete example workflow
bash examples/example_cli_workflow.sh
```

### Python API Quick Start

If you prefer programmatic access:

```python
from src.phenohunter_scientific import create_phenohunter
import pandas as pd

# Initialize PhenoHunter
ph = create_phenohunter()

# Load your strain database
strain_data = pd.read_csv('your_strain_database.csv')
warnings = ph.load_strain_database(strain_data, validate=True)

# Train VAE model (369 epochs - harmonic alignment)
vae_history = ph.train_vae(epochs=369, verbose=True)

# Generate F1 hybrid between two parent strains
f1_result = ph.generate_f1_hybrid(
    parent1_name='Blue Dream',
    parent2_name='OG Kush',
    parent1_weight=0.6,  # 60% Blue Dream, 40% OG Kush
    n_samples=100        # Monte Carlo samples for uncertainty
)

# Print results
print(f"Stability Score: {f1_result.stability_score:.3f}")
print(f"Heterosis Score: {f1_result.heterosis_score:.3f}")
```

### Example 3: Generate F2 Population

```python
# Generate F2 population from F1 hybrid
f2_population = ph.generate_f2_population(
    f1_parent=f1_result,
    n_offspring=10,  # Generate 10 F2 candidates
    variance_scale=1.5  # Increased variance for segregation
)

# Rank by desired effect
ranked_strains = sorted(
    f2_population,
    key=lambda x: x['effects']['Analgesic'],
    reverse=True
)

# Export top candidates
ph.export_results(ranked_strains[:5], 'top_5_analgesic_strains.csv')
```

### Example 4: Backcross Breeding

```python
# Backcross F1 to Blue Dream to preserve its traits
bx1_result = ph.backcross(
    hybrid=f1_result,
    parent_name='Blue Dream',
    backcross_generation=1,  # BX1
    n_samples=100
)

# Visualize convergence to parent
ph.visualize_breeding_trajectory([
    {'name': 'Blue Dream', 'profile': ph.get_strain_profile('Blue Dream')},
    {'name': 'F1', 'profile': f1_result['chemical_profile']},
    {'name': 'BX1', 'profile': bx1_result['chemical_profile']}
])
```

### Example 5: Custom Strain Analysis

```python
# Analyze a custom strain profile
custom_strain = {
    'THC': 22.5,
    'CBD': 0.8,
    'CBG': 1.2,
    'Myrcene': 0.6,
    'Limonene': 0.3,
    'Caryophyllene': 0.4,
    # ... add more compounds
}

# Predict effects
effects = ph.predict_effects(custom_strain)
print(f"Predicted therapeutic effects:")
for effect, confidence in effects.items():
    print(f"  {effect}: {confidence:.1%}")

# Find similar strains in database
similar_strains = ph.find_similar_strains(custom_strain, top_k=5)
print(f"\nMost similar strains:")
for strain, similarity in similar_strains:
    print(f"  {strain}: {similarity:.2f} cosine similarity")
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                 DATA INGESTION                          │
│  CSV • API • Web Scraping • Lab COAs                    │
└───────────────┬─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│          5-STAGE PREPROCESSING PIPELINE                 │
│  Acquisition → Cleaning → Normalization →               │
│  Feature Extraction → Validation                        │
└───────────────┬─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│            MACHINE LEARNING MODELS                      │
│  • VAE (27 latent dims, 7 layers)                       │
│  • Ensemble Effect Predictor (3 models, 6 effects)      │
│  • Breeding Strategy (F1, F2, Backcross)                │
└───────────────┬─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│           9-STEP POST-PROCESSING                        │
│  Prediction → Uncertainty → Validation → Entourage →    │
│  Literature Check → XAI → Visualization → Metadata →    │
│  Export                                                 │
└───────────────┬─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│        VISUALIZATION & DASHBOARDS                       │
│  3D Molecules • Interactive Plots • Real-time Metrics   │
└─────────────────────────────────────────────────────────┘
```

### Module Structure

```
C.C.R.O.P-PhenoHunt/
├── src/
│   ├── data/
│   │   └── validators.py          # Data validation
│   ├── models/
│   │   ├── vae.py                 # VAE architecture
│   │   ├── effect_predictor.py   # Effect prediction
│   │   └── breeding_strategy.py  # Breeding algorithms
│   ├── genomics/                  # NEW v3.0.0
│   │   ├── genomic_selection.py  # G-BLUP & GRM computation
│   │   └── gwas_wrapper.py       # GWAS with PLINK integration
│   ├── phenomics/                 # NEW v3.0.0
│   │   └── feature_extraction.py # Image-based trait extraction
│   ├── io/                        # NEW v3.0.0
│   │   └── brapi_adapter.py      # BrAPI v2.1 compliance
│   ├── cheminformatics/
│   │   ├── molecular_descriptors.py  # RDKit descriptors
│   │   └── evidence_mapping.py   # NEW v3.0.0 - Literature DOI mapping
│   ├── xai/                       # NEW v3.0.0
│   │   └── shap_utils.py         # SHAP explanations
│   ├── uncertainty/               # NEW v3.0.0
│   │   └── calibration.py        # Conformal prediction & calibration
│   ├── exp/                       # NEW v3.0.0
│   │   ├── harmonic_seed.py      # Sacred geometry seeding
│   │   └── experiment_meta.py    # Metadata tracking
│   ├── visualization/
│   │   └── molecular_3d.py       # 3D molecular rendering
│   └── utils/
│       ├── config.py              # Configuration
│       └── visualization.py       # Plotting utilities
├── scripts/                       # NEW v3.0.0
│   └── run_pipeline.py           # Phenomics pipeline orchestrator
├── pipelines/                     # NEW v3.0.0
│   └── phenomics_pipeline.yml    # Pipeline configuration
├── docker/                        # NEW v3.0.0
│   └── phenomics.Dockerfile      # Containerized environment
├── .github/workflows/             # NEW v3.0.0
│   ├── ci.yml                    # CI/CD pipeline
│   └── docker-build.yml          # Docker image builds
├── notebooks/                     # NEW v3.0.0
│   └── brapi_demo.ipynb          # BrAPI demonstration
├── tests/
│   └── test_brapi_adapter.py     # NEW v3.0.0 - BrAPI tests
├── configs/
│   └── scientific_config.yaml    # Scientific configuration
├── docs/
│   ├── ARCHITECTURE.md           # System architecture
│   ├── CHANGELOG.md              # Version history
│   ├── RELEASE.md                # NEW v3.0.0 - Release process
│   └── SOURCES_AND_ACKNOWLEDGEMENTS.md  # Citations
├── example_release/               # NEW v3.0.0
│   ├── BENCHMARKS.md             # Performance benchmarks
│   └── CITATION.cff              # Zenodo citation
├── PhenoHunter.ipynb             # Jupyter notebook interface
├── example_usage.py              # Usage examples
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Usage

### Command-Line Interface (Primary)

#### Main CLI Commands

```bash
# Get help
phenohunt --help
phenohunt train --help
phenohunt cross --help

# Train models on your dataset
phenohunt train --data strains.csv --epochs 369 --verbose

# Generate F1 hybrid
phenohunt cross \
    --data strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --ratio 0.6 \
    --output f1_hybrid.csv \
    --visualize

# Generate F2 population
phenohunt f2 \
    --data strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --count 20 \
    --trait "Analgesic" \
    --output f2_population.csv

# Generate backcross
phenohunt backcross \
    --data strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --backcross-to "Blue Dream" \
    --generation 1 \
    --output bx1.csv

# Analyze strains
phenohunt analyze \
    --data strains.csv \
    --strains "Blue Dream" "OG Kush" "Sour Diesel"

# Import data
phenohunt import \
    --file lab_results.csv \
    --output processed_strains.csv
```

#### Module-Specific Commands

```bash
# BrAPI Adapter
python -m src.io.brapi_adapter import --in sample.json --out traits.csv
python -m src.io.brapi_adapter export --in predictions.csv --out output.json --study-id STUDY_001
python -m src.io.brapi_adapter validate data.json

# Phenomics Pipeline
python scripts/run_pipeline.py --config pipelines/phenomics_pipeline.yml
python scripts/run_pipeline.py --config pipelines/phenomics_pipeline.yml --stage feature_extract
```

### Python API

For programmatic access, see [example_usage.py](example_usage.py):

```python
from src.phenohunter_scientific import create_phenohunter

ph = create_phenohunter()
# Use all the same methods as before
```

### Jupyter Notebook (Deprecated)

⚠️ **The Jupyter notebook interface is deprecated.** See [JUPYTER_DEPRECATED.md](JUPYTER_DEPRECATED.md) and [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for details.

If you need notebook-style workflows, you can create your own notebooks using the Python API.

### REST API (Future)

```bash
# Start API server
python -m phenohunt api --host 0.0.0.0 --port 5000

# Example API calls
curl -X POST http://localhost:5000/api/v1/predict_effects \
    -H "Content-Type: application/json" \
    -d '{"THC": 20.0, "CBD": 1.0, "Myrcene": 0.5}'

curl -X POST http://localhost:5000/api/v1/generate_hybrid \
    -H "Content-Type: application/json" \
    -d '{"parent1": "Blue Dream", "parent2": "OG Kush", "ratio": 0.6}'
```

---

## Sacred Geometry Integration

### Philosophy

This project integrates **sacred geometry and numerology** principles inspired by Nikola Tesla's philosophy:

> "If you only knew the magnificence of the 3, 6 and 9, then you would have a key to the universe."
> — Nikola Tesla

### Harmonic Numbers in System Design

| Number | Significance | Application |
|--------|--------------|-------------|
| **3** | Trinity, Triadic Transformation | Input → Transform → Output flow |
| **5** | Pentagon, 5 Elements | 5-stage preprocessing pipeline |
| **7** | Heptagon, Perfection | 7-layer neural network architectures |
| **9** | Nonagon, Completion | 9-step post-processing synthesis |
| **12** | Dodecagon, Cosmic Order | 12-module repository structure |
| **27** | 3³, Harmonic Cube | 27 latent dimensions in VAE |
| **33** | Master Number | 33 primary chemical & phenotypic features |
| **369** | Tesla's Divine Numbers | Ultimate training epoch count |

### Training Cycle Harmonics

All ML training follows harmonic epoch counts:

- **Quick**: 27 epochs (3 × 9)
- **Standard**: 369 epochs (3 × 123)
- **Extended**: 999 epochs (3 × 333)

Model checkpoints are saved at harmonic intervals: 3, 9, 27, 81 epochs.

### Data Partitioning

The VAE latent space is organized into **27 semantic partitions**:
- 9 chemical partitions (cannabinoids, terpenes, ratios)
- 9 phenotypic partitions (trichomes, buds, leaves)
- 9 therapeutic partitions (effects, side effects)

### Color Schemes

Visualizations use harmonic color palettes:
- **3-color**: Primary triads (Red-Yellow-Blue)
- **6-color**: Hexagonal harmony
- **9-color**: Nonagonal spectrum
- **12-color**: Full chromatic wheel

---

## Documentation

### Core Documentation

- **[README.md](README.md)** (this file): Overview and quick start
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed system architecture and design decisions
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and release notes
- **[SOURCES_AND_ACKNOWLEDGEMENTS.md](SOURCES_AND_ACKNOWLEDGEMENTS.md)**: Data sources and scientific citations

### Scientific Documentation

- **[SCIENTIFIC_IMPROVEMENTS.md](SCIENTIFIC_IMPROVEMENTS.md)**: Detailed explanation of scientific enhancements
- **[configs/scientific_config.yaml](configs/scientific_config.yaml)**: Configuration with embedded citations
- **[Data-Sources](Data-Sources)**: Dataset licenses and attributions
- **[Model_Card](Model_Card)**: Model documentation and limitations

### Code Documentation

All code includes:
- Comprehensive docstrings (Google style)
- Type hints for all functions
- Inline comments for complex logic
- Scientific citations in comments

### Testing

```bash
# Run all tests
pytest src/tests/ -v

# Run with coverage
pytest src/tests/ --cov=src --cov-report=html

# Run specific test
pytest src/tests/test_scientific_improvements.py::test_vae_training -v
```

---

## Contributing

We welcome contributions from the scientific community!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow harmonic commit conventions**: `[HARMONIC-XXX] TYPE: Description`
4. **Add tests** for new functionality
5. **Ensure all tests pass**: `pytest src/tests/ -v`
6. **Submit a pull request**

### Contribution Guidelines

- Follow PEP 8 style guide for Python code
- Include type hints for all functions
- Write comprehensive docstrings
- Add unit tests for new features
- Update documentation as needed
- Cite scientific sources for new effect coefficients

### Harmonic Commit Convention

```
[HARMONIC-003] INIT: Initialize new module
[HARMONIC-006] FEAT: Add new feature
[HARMONIC-009] ENHANCE: Improve existing feature
[HARMONIC-012] FIX: Bug fix
[HARMONIC-027] REFACTOR: Code refactoring
[HARMONIC-033] DOCS: Documentation update
[HARMONIC-369] TEST: Testing improvements
```

See [CHANGELOG.md](CHANGELOG.md) for full convention details.

---

## Citation

If you use C.C.R.O.P-PhenoHunt in your research, please cite:

```bibtex
@software{crop_phenohunt_2025,
  author = {Hosuay and Contributors},
  title = {C.C.R.O.P-PhenoHunt: Cannabis Computational Research & Optimization Platform},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Hosuay/C.C.R.O.P-PhenoHunt}},
  version = {3.0.0}
}
```

### Key Scientific References

1. **Russo, E. B. (2011)**. Taming THC: Potential cannabis synergy and phytocannabinoid-terpenoid entourage effects. *British Journal of Pharmacology*, 163(7), 1344–1364.

2. **Smith, C. J., et al. (2022)**. The phytochemical diversity of commercial Cannabis in the United States. *PLOS ONE*, 17(5), e0267498.

3. **Blessing, E. M., et al. (2015)**. Cannabidiol as a Potential Treatment for Anxiety Disorders. *Neurotherapeutics*, 12(4), 825–836.

See [SOURCES_AND_ACKNOWLEDGEMENTS.md](SOURCES_AND_ACKNOWLEDGEMENTS.md) for full citation list.

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Permissions

✅ Commercial use
✅ Modification
✅ Distribution
✅ Private use

### Limitations

❌ Liability
❌ Warranty

---

## Acknowledgments

### Data Sources

- **Leafly** - Cannabis strain profiles
- **Cannlytics** - Laboratory testing data
- **NCBI** - Cannabis sativa genome assemblies
- **Kaggle** - Public strain datasets

### Scientific Community

- Cannabis research community (Overgrow, THCFarmer, r/CannabisBreeding)
- Open-source contributors
- Peer-reviewed literature authors

### Technology

- PyTorch, Scikit-learn, Plotly, RDKit
- Google Colab for cloud computing
- GitHub for version control

---

## Support

### Getting Help

- **Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md), [SCIENTIFIC_IMPROVEMENTS.md](SCIENTIFIC_IMPROVEMENTS.md)
- **Issues**: [GitHub Issues](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt/discussions)

### Contact

- **GitHub**: [@Hosuay](https://github.com/Hosuay)
- **Repository**: [C.C.R.O.P-PhenoHunt](https://github.com/Hosuay/C.C.R.O.P-PhenoHunt)

---

## Roadmap

### Version 3.0.0 (Current)
✅ VAE with 27 latent dimensions
✅ Ensemble effect prediction (6 effects)
✅ Multi-generation breeding (F1, F2, BX)
✅ Comprehensive documentation
✅ Sacred geometry integration

### Version 6.0.0 (Planned)
🔄 C++ phenotypic analysis modules
🔄 Rust data integrity pipelines
🔄 Julia scientific computation
🔄 3D molecular visualization (RDKit, py3Dmol)
🔄 Interactive dashboards (Plotly Dash)

### Version 9.0.0 (Planned)
📅 REST API with FastAPI
📅 Web application (React frontend)
📅 Multi-user authentication
📅 Real-time collaborative breeding
📅 Clinical validation integration

### Version 369.0.0 (Ultimate Vision)
🔮 Blockchain strain lineage tracking
🔮 Federated learning for global models
🔮 Regulatory compliance automation
🔮 Open-source collaborative platform

---

## Legal Disclaimer

⚠️ **IMPORTANT LEGAL NOTICE** ⚠️

This software is provided for **educational and research purposes only**. It:

- Does **NOT** provide medical advice or clinical recommendations
- Does **NOT** provide legal cultivation guidance
- Generates computational hypotheses requiring laboratory validation
- Must be used in compliance with local, state, and federal laws

**Cannabis laws vary by jurisdiction**. Users are solely responsible for ensuring compliance with all applicable laws and regulations.

The authors and contributors assume **NO LIABILITY** for any misuse of this software.

---

**Version**: 3.0.0 | **Last Updated**: 2025-10-22 | **Maintained by**: Hosuay & Contributors

**Aligned with Sacred Geometry** ✨ **369** ✨
