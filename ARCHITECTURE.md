# SYSTEM ARCHITECTURE
## C.C.R.O.P-PhenoHunt - Cannabis Computational Research & Optimization Platform

---

## Table of Contents

1. [Overview](#overview)
2. [Programming Language Architecture](#programming-language-architecture)
3. [System Design Principles](#system-design-principles)
4. [Module Structure](#module-structure)
5. [Data Flow Architecture](#data-flow-architecture)
6. [Machine Learning Pipeline](#machine-learning-pipeline)
7. [Sacred Geometry Integration](#sacred-geometry-integration)
8. [Scalability and Performance](#scalability-and-performance)
9. [Security and Compliance](#security-and-compliance)
10. [Future Roadmap](#future-roadmap)

---

## Overview

C.C.R.O.P-PhenoHunt is a multi-language, modular platform for cannabis strain analysis, breeding optimization, and phenotypic prediction using machine learning and sacred geometry principles.

### Core Objectives

1. **Scientific Rigor**: All predictions backed by peer-reviewed literature
2. **Uncertainty Quantification**: Confidence intervals for all outputs
3. **Modular Architecture**: Clean separation of concerns across components
4. **Harmonic Alignment**: Integration of sacred geometry numerology (3-6-9 sequence)
5. **Reproducibility**: Deterministic execution with versioning
6. **Scalability**: Cloud and local execution compatibility

---

## Programming Language Architecture

### Multi-Language Strategy

The platform employs a **polyglot architecture** where each language is selected for its specific strengths:

#### 1. **Python** - Machine Learning & Data Science (Primary)

**Usage**:
- ML pipelines (VAE, ensemble models, XAI)
- Data ingestion, preprocessing, validation
- Visualization and dashboards
- Web scraping and API integration
- Jupyter notebook interfaces

**Justification**:
- Rich ecosystem for scientific computing (NumPy, Pandas, SciPy)
- Mature ML frameworks (PyTorch, Scikit-learn)
- Excellent visualization libraries (Plotly, Matplotlib)
- Google Colab compatibility
- Rapid prototyping and iteration

**Components**:
```
src/
├── data/               # Data validation and processing
├── models/             # ML models (VAE, effect predictor, breeding)
├── utils/              # Utilities (config, visualization)
├── preprocessing/      # Feature engineering
├── visualization/      # Dashboards and plots
└── api/                # REST API endpoints
```

**Code Location**: `src/**/*.py`

---

#### 2. **C++** - Computational-Heavy Phenotypic Analysis (Planned)

**Usage**:
- High-performance image processing for phenotypic analysis
- Trichome density computation
- Leaf morphology quantification
- Real-time sensor data processing
- 3D point cloud processing for plant structure

**Justification**:
- **Performance**: 10-100x faster than Python for numerical computations
- **Memory Efficiency**: Fine-grained control for large image datasets
- **Parallelization**: Native threading and SIMD optimizations
- **Computer Vision**: OpenCV integration for phenotypic analysis
- **GPU Acceleration**: CUDA/OpenCL support for massive parallelism

**Planned Components**:
```
cpp/
├── include/
│   ├── phenotype/      # Phenotypic analysis headers
│   ├── imaging/        # Image processing utilities
│   └── computation/    # Numerical computation kernels
├── src/
│   ├── trichome_analysis.cpp
│   ├── leaf_morphology.cpp
│   ├── bud_structure.cpp
│   └── sensor_processing.cpp
├── bindings/
│   └── python/         # PyBind11 bindings for Python interop
└── CMakeLists.txt
```

**Python Bindings**: PyBind11 for seamless integration

**Example Use Case**:
```python
# Python calls C++ module for fast trichome analysis
from phenohunt.cpp_modules import trichome_analyzer

result = trichome_analyzer.analyze_image(
    image_path="strain_image.jpg",
    density_threshold=0.7,
    gpu_accelerated=True
)
```

**Code Location**: `cpp/**/*.cpp`, `cpp/**/*.h`

---

#### 3. **Rust** - Data Integrity & Pipeline Safety (Planned)

**Usage**:
- Data validation and integrity checks
- Cryptographic hashing for dataset versioning
- Concurrent data processing pipelines
- File I/O with error handling
- Blockchain-based strain lineage tracking (future)

**Justification**:
- **Memory Safety**: No segmentation faults or undefined behavior
- **Concurrency**: Fearless parallelism without data races
- **Performance**: Comparable to C++ with zero-cost abstractions
- **Error Handling**: Explicit Result/Option types prevent silent failures
- **Data Integrity**: Strong type system prevents data corruption

**Planned Components**:
```
rust/
├── src/
│   ├── lib.rs
│   ├── validation/
│   │   ├── schema.rs       # Data schema validation
│   │   ├── integrity.rs    # Hash-based integrity checks
│   │   └── sanitize.rs     # Input sanitization
│   ├── pipeline/
│   │   ├── ingest.rs       # Concurrent data ingestion
│   │   ├── transform.rs    # Safe data transformations
│   │   └── export.rs       # Validated exports
│   └── lineage/
│       └── tracker.rs      # Blockchain-based tracking
├── bindings/
│   └── python/             # PyO3 bindings for Python interop
└── Cargo.toml
```

**Python Bindings**: PyO3 for Python integration

**Example Use Case**:
```python
# Python calls Rust module for validated data ingestion
from phenohunt.rust_modules import data_validator

validated_data = data_validator.validate_and_ingest(
    csv_path="strain_database.csv",
    schema_version="3.0",
    integrity_check=True
)
```

**Code Location**: `rust/src/**/*.rs`

---

#### 4. **Julia** - Scientific Computations & High-Precision Numerics (Planned)

**Usage**:
- High-precision molecular descriptor calculations
- Statistical modeling with numerical stability
- Optimization algorithms (genetic algorithms, multi-objective)
- Dose-response modeling
- Pharmacokinetic/pharmacodynamic (PK/PD) simulations

**Justification**:
- **Numerical Precision**: IEEE 754 compliance with arbitrary precision
- **Performance**: JIT compilation approaches C/Fortran speeds
- **Scientific Libraries**: DifferentialEquations.jl, Optimization.jl
- **Readable Syntax**: Mathematical notation similar to Python
- **Multiple Dispatch**: Elegant handling of different numeric types

**Planned Components**:
```
julia/
├── src/
│   ├── PhenoHunt.jl
│   ├── molecular/
│   │   ├── descriptors.jl  # Molecular descriptor computation
│   │   └── properties.jl   # Chemical property prediction
│   ├── optimization/
│   │   ├── genetic_algo.jl # Multi-objective genetic algorithms
│   │   └── strain_design.jl# Optimal strain design
│   └── pharmacology/
│       ├── pk_pd.jl        # PK/PD modeling
│       └── dose_response.jl# Dose-response curves
└── Project.toml
```

**Python Bindings**: PyJulia or HTTP API for integration

**Example Use Case**:
```python
# Python calls Julia for high-precision molecular descriptors
from phenohunt.julia_modules import molecular_descriptors

descriptors = molecular_descriptors.compute(
    smiles="CCO",
    precision="arbitrary",
    include_3d=True
)
```

**Code Location**: `julia/src/**/*.jl`

---

### Inter-Language Communication

#### Binding Strategies

1. **Python ↔ C++**: PyBind11
   - Seamless function calls with automatic type conversion
   - NumPy array zero-copy sharing
   - Exception handling across boundaries

2. **Python ↔ Rust**: PyO3
   - Safe FFI with automatic memory management
   - GIL-aware concurrency
   - Python exceptions mapped to Rust Results

3. **Python ↔ Julia**: PyJulia or HTTP API
   - Function calls with automatic serialization
   - Shared memory for large arrays (future)
   - Async execution for long-running tasks

#### Data Exchange Format

- **In-Memory**: NumPy arrays, Pandas DataFrames (Python), Eigen matrices (C++), ndarray (Rust), Arrays (Julia)
- **On-Disk**: Parquet, HDF5, CSV for cross-language compatibility
- **Serialization**: Protocol Buffers, Apache Arrow for efficient transfer

---

## System Design Principles

### 1. **Triadic Input → Transform → Output Flow**

Every module follows a three-stage process:

```
INPUT (Acquisition)
  ↓
TRANSFORM (Processing)
  ↓
OUTPUT (Results)
```

**Example**: Data Ingestion
- **Input**: Raw CSV files
- **Transform**: Validation, cleaning, normalization
- **Output**: Harmonized Parquet files

### 2. **5-Stage Preprocessing Pipeline**

```
1. Acquisition  → Load raw data from sources
2. Cleaning     → Remove duplicates, handle missing values
3. Normalization→ Scale features to standard ranges
4. Feature Extraction → Derive compound ratios, embeddings
5. Validation   → Final quality checks, outlier detection
```

### 3. **7-Layer ML Network Design**

Neural networks follow a standardized 7-layer architecture:

```
Input Layer (Layer 0)
  ↓
Dense + BatchNorm + ReLU (Layer 1)
  ↓
Dense + BatchNorm + ReLU (Layer 2)
  ↓
Bottleneck / Latent Space (Layer 3) [Center layer - harmonic 7÷2]
  ↓
Dense + BatchNorm + ReLU (Layer 4)
  ↓
Dense + BatchNorm + ReLU (Layer 5)
  ↓
Output Layer (Layer 6)
```

### 4. **9-Step Post-Processing Synthesis**

All outputs undergo comprehensive post-processing:

```
1. Raw Prediction        → Model output
2. Uncertainty Quantification → Confidence intervals
3. Range Validation      → Ensure biological plausibility
4. Entourage Effect Adjustment → Compound interactions
5. Literature Cross-Check → Compare to known profiles
6. XAI Explanation       → Generate interpretations
7. Visualization Preparation → Format for plots
8. Metadata Annotation   → Add timestamps, versions
9. Export & Archival     → Save with full provenance
```

### 5. **12-Module Repository Structure**

```
C.C.R.O.P-PhenoHunt/
├── 01_data/                # Data ingestion & management
├── 02_preprocessing/       # Feature engineering
├── 03_models/              # ML model implementations
├── 04_training/            # Training pipelines
├── 05_inference/           # Prediction & generation
├── 06_validation/          # Quality assurance
├── 07_visualization/       # Dashboards & plots
├── 08_api/                 # REST API endpoints
├── 09_utils/               # Shared utilities
├── 10_tests/               # Testing suite
├── 11_docs/                # Documentation
└── 12_deployment/          # Deployment configs
```

### 6. **27 Latent Partitions**

The VAE latent space is organized into 27 semantic partitions:

**Chemical Partitions (9)**:
1. THC-dominant cannabinoids
2. CBD-dominant cannabinoids
3. Minor cannabinoids (CBG, CBC, etc.)
4. Monoterpenes (Limonene, Pinene, etc.)
5. Sesquiterpenes (Caryophyllene, Humulene, etc.)
6. Terpene alcohols (Linalool, Bisabolol, etc.)
7. Cannabinoid ratios (THC:CBD, etc.)
8. Terpene synergy profiles
9. Entourage effect signatures

**Phenotypic Partitions (9)**:
10. Trichome density
11. Bud structure
12. Leaf morphology
13. Plant height
14. Flowering time
15. Yield metrics
16. Root architecture
17. Stem thickness
18. Color profiles

**Therapeutic Partitions (9)**:
19. Analgesic potential
20. Anxiolytic potential
21. Sedative potential
22. Anti-inflammatory potential
23. Neuroprotective potential
24. Appetite stimulation
25. Energy/focus effects
26. Creativity enhancement
27. Side effect risk

### 7. **33 Primary Chemical & Phenotypic Features**

**Cannabinoids (9)**:
1. Δ9-THC
2. CBD
3. CBG
4. CBC
5. CBDA
6. THCV
7. CBN
8. Δ8-THC
9. THCA

**Terpenes (11)**:
10. Myrcene
11. Limonene
12. α-Pinene
13. Linalool
14. β-Caryophyllene
15. Humulene
16. Terpinolene
17. Ocimene
18. Camphene
19. Bisabolol
20. β-Pinene

**Molecular Descriptors (5)**:
21. Molecular Weight
22. LogP (Lipophilicity)
23. H-Bond Donors
24. H-Bond Acceptors
25. TPSA (Topological Polar Surface Area)

**Phenotypic Traits (8)**:
26. Trichome Density (0-1)
27. Bud Density (g/cm³)
28. Leaf Surface Area (cm²)
29. Plant Height (cm)
30. Flowering Time (days)
31. Yield (g/plant)
32. CBD:THC Ratio
33. Total Terpene Content (%)

---

## Module Structure

### Current Python Modules

#### 1. Data Module (`src/data/`)

**File**: `validators.py` (333 lines)

**Functions**:
- `validate_chemical_ranges()`: Literature-based range validation
- `detect_outliers()`: Isolation Forest anomaly detection
- `validate_thc_cbd_ratio()`: Ratio consistency checks
- `impute_missing_data()`: Median/mean imputation strategies
- `generate_quality_report()`: Comprehensive QC summary

**Sacred Geometry Alignment**: 3-stage validation (Input → Check → Report)

---

#### 2. Models Module (`src/models/`)

**File**: `vae.py` (408 lines)

**Classes**:
- `VAE`: Variational Autoencoder with reparameterization trick
- `Encoder`: 7-layer encoder network
- `Decoder`: 7-layer decoder network

**Features**:
- β-VAE formulation for controlled disentanglement
- Monte Carlo uncertainty quantification
- Xavier weight initialization
- Batch normalization + dropout regularization

**Sacred Geometry Alignment**: 7-layer architecture, 27 latent dimensions

---

**File**: `effect_predictor.py` (467 lines)

**Classes**:
- `EffectPredictor`: Ensemble model for therapeutic effects
- `ScientificEffectModel`: Literature-backed coefficients

**Models**:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

**Effects Predicted**:
1. Analgesic (pain relief)
2. Anxiolytic (anxiety reduction)
3. Sedative (sleep promotion)
4. Anti-inflammatory
5. Neuroprotective
6. Appetite stimulant

**Sacred Geometry Alignment**: 3 ensemble models, 6 effect categories

---

**File**: `breeding_strategy.py` (399 lines)

**Classes**:
- `F1Generator`: Uniform F1 hybrid generation
- `F2Generator`: Segregating F2 population
- `BackcrossGenerator`: Progressive backcross introgression

**Features**:
- Mendelian genetics simulation
- Heterosis (hybrid vigor) calculation
- Genetic variance modeling
- Stability scoring

**Sacred Geometry Alignment**: 3 breeding strategies (F1, F2, BX)

---

#### 3. Utils Module (`src/utils/`)

**File**: `config.py` (74 lines)

**Functions**:
- `load_config()`: YAML configuration loader
- `get_harmonic_seeds()`: Generate seeds aligned with 3-6-9
- `validate_config()`: Configuration schema validation

---

**File**: `visualization.py` (460 lines)

**Functions**:
- `plot_chemical_profile()`: Radar charts with error bars
- `plot_effect_predictions()`: Bar charts with confidence intervals
- `plot_latent_space()`: 2D/3D latent embeddings
- `plot_breeding_trajectory()`: Parent-offspring comparisons
- `plot_uncertainty_heatmap()`: Uncertainty visualization

**Sacred Geometry Alignment**: Harmonic color palettes (3-6-9 color coding)

---

#### 4. Main API (`src/phenohunter_scientific.py`)

**File**: 559 lines

**Class**: `PhenoHunterScientific`

**Key Methods**:
- `load_strain_database()`: Data ingestion with validation
- `train_vae()`: VAE training with harmonic epochs
- `train_effect_predictors()`: Ensemble model training
- `generate_f1_hybrid()`: F1 strain generation
- `generate_f2_population()`: F2 strain generation
- `backcross()`: Backcross breeding
- `export_results()`: Harmonized CSV export
- `get_summary_report()`: Comprehensive analysis summary

---

### Planned C++ Modules

#### Phenotypic Analysis Module

**Files**: `cpp/src/trichome_analysis.cpp`, `leaf_morphology.cpp`, `bud_structure.cpp`

**Functions**:
- `analyze_trichome_density()`: GPU-accelerated image segmentation
- `measure_leaf_area()`: Contour detection and area computation
- `quantify_bud_structure()`: 3D point cloud analysis

**Performance**: 10-100x faster than Python for large image datasets

---

### Planned Rust Modules

#### Data Integrity Module

**Files**: `rust/src/validation/integrity.rs`, `schema.rs`

**Functions**:
- `validate_schema()`: Type-safe schema validation
- `compute_hash()`: Cryptographic hashing for versioning
- `sanitize_input()`: SQL injection and XSS prevention
- `concurrent_ingest()`: Parallel data loading with error handling

**Performance**: Memory-safe with zero overhead

---

### Planned Julia Modules

#### Scientific Computation Module

**Files**: `julia/src/molecular/descriptors.jl`, `optimization/genetic_algo.jl`

**Functions**:
- `compute_molecular_descriptors()`: High-precision chemical properties
- `optimize_strain_design()`: Multi-objective genetic algorithms
- `model_pk_pd()`: Pharmacokinetic/pharmacodynamic simulations

**Performance**: Near-C speeds with numerical stability

---

## Data Flow Architecture

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INGESTION                          │
│  (Python: src/data/)                                        │
│  - Leafly scraping                                          │
│  - CSV import                                               │
│  - API data fetch                                           │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│                  DATA VALIDATION                            │
│  (Rust: rust/src/validation/ - Future)                      │
│  - Schema validation                                        │
│  - Integrity checks                                         │
│  - Sanitization                                             │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│                  PREPROCESSING                              │
│  (Python: src/preprocessing/)                               │
│  5-Stage Pipeline:                                          │
│  1. Acquisition                                             │
│  2. Cleaning                                                │
│  3. Normalization                                           │
│  4. Feature Extraction                                      │
│  5. Validation                                              │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│              PHENOTYPIC ANALYSIS                            │
│  (C++: cpp/src/phenotype/)                                  │
│  - Trichome density analysis                                │
│  - Leaf morphology quantification                           │
│  - Bud structure measurement                                │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│            MOLECULAR DESCRIPTORS                            │
│  (Julia: julia/src/molecular/ - Future)                     │
│  - Molecular weight                                         │
│  - LogP, H-bonds, TPSA                                      │
│  - 3D conformer generation                                  │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│                 ML TRAINING                                 │
│  (Python: src/models/)                                      │
│  - VAE training (27 latent dims)                            │
│  - Effect predictor training (6 effects)                    │
│  - Breeding strategy optimization                           │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│            STRAIN GENERATION                                │
│  (Python: src/models/breeding_strategy.py)                  │
│  - F1 hybrid generation                                     │
│  - F2 population generation                                 │
│  - Backcross optimization                                   │
│  - 27-block partitioning                                    │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│           POST-PROCESSING (9 Steps)                         │
│  (Python: src/utils/)                                       │
│  1. Raw Prediction                                          │
│  2. Uncertainty Quantification                              │
│  3. Range Validation                                        │
│  4. Entourage Effect Adjustment                             │
│  5. Literature Cross-Check                                  │
│  6. XAI Explanation                                         │
│  7. Visualization Preparation                               │
│  8. Metadata Annotation                                     │
│  9. Export & Archival                                       │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│               VISUALIZATION                                 │
│  (Python: src/visualization/)                               │
│  - 3D molecular structures (RDKit, py3Dmol)                 │
│  - Interactive dashboards (Plotly, Dash)                    │
│  - Chemical radar plots                                     │
│  - Effect prediction charts                                 │
└───────────────┬─────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────────┐
│                  OUTPUT & EXPORT                            │
│  (Python/Rust: multi-format)                                │
│  - Harmonized CSV                                           │
│  - JSON API responses                                       │
│  - PDF reports                                              │
│  - Interactive HTML dashboards                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Pipeline

### Training Cycles Aligned with 3-6-9 Sequence

All ML training follows harmonic epoch counts:

- **Quick Training**: 27 epochs (3 × 9)
- **Standard Training**: 369 epochs (3 × 6 × 9 + 3 × 6 × 9)
- **Extended Training**: 999 epochs (3 × 333)

### Checkpoint Saving

Model checkpoints are saved at harmonic intervals:
- Every 3 epochs (quick monitoring)
- Every 9 epochs (standard monitoring)
- Every 27 epochs (long-term snapshots)

### Batch Sizes

Batch sizes follow harmonic numbers:
- Small batches: 9
- Medium batches: 27
- Large batches: 81 (3^4)

### Learning Rate Schedules

Learning rates decay at harmonic milestones:
- Epoch 3: First decay
- Epoch 9: Second decay
- Epoch 27: Third decay
- Epoch 81: Fourth decay

---

## Sacred Geometry Integration

### Numerological Alignment

Every aspect of the system is aligned with sacred geometry:

#### **3**: Trinity / Triadic Processes
- Input → Transform → Output
- Train → Validate → Test
- Parent1 → Hybrid → Parent2

#### **5**: Pentagon / Preprocessing Stages
- Acquisition → Cleaning → Normalization → Extraction → Validation

#### **7**: Heptagon / Network Layers
- 7-layer neural networks for balanced depth

#### **9**: Nonagon / Post-Processing Steps
- 9-step synthesis for comprehensive outputs

#### **12**: Dodecagon / Module Count
- 12 primary modules in repository structure

#### **27**: Latent Dimensions
- 27 latent partitions (3^3)
- 27 semantic groups

#### **33**: Master Number / Primary Features
- 33 chemical and phenotypic features

#### **369**: Tesla's Divine Numbers
- Ultimate training epoch count
- Harmonic convergence milestone

---

## Scalability and Performance

### Cloud Execution (Google Colab, AWS, Azure)

- **Python**: Native Jupyter notebook support
- **C++**: Compiled binaries uploaded and executed via Python bindings
- **Rust**: WASM compilation for web deployment
- **Julia**: HTTP API for remote execution

### Local Execution

- **Python**: Direct execution with virtual environment
- **C++**: Native binaries for maximum performance
- **Rust**: Cargo-managed local builds
- **Julia**: REPL-based execution

### GPU Acceleration

- **Python**: PyTorch CUDA support
- **C++**: CUDA/OpenCL kernels for image processing
- **Julia**: CUDA.jl for GPU arrays

### Parallel Processing

- **Python**: Multiprocessing for data ingestion
- **C++**: OpenMP for parallel loops
- **Rust**: Rayon for fearless concurrency
- **Julia**: Built-in threading macros

---

## Security and Compliance

### Data Privacy

- No personally identifiable information (PII) stored
- All patient data must be de-identified (IRB/HIPAA compliance)
- Secure credential management via environment variables

### Input Validation

- **Rust Module**: Type-safe schema validation
- SQL injection prevention
- XSS sanitization for web interfaces

### Audit Logging

- All data ingestion logged with timestamps
- Model training runs versioned with Git commits
- Full provenance tracking for reproducibility

---

## Future Roadmap

### Version 6.0.0 (Multi-Language Integration)

- [ ] C++ phenotypic analysis modules
- [ ] Rust data integrity pipelines
- [ ] Julia scientific computation modules
- [ ] Inter-language bindings (PyBind11, PyO3, PyJulia)

### Version 9.0.0 (Web Application)

- [ ] REST API with FastAPI
- [ ] Interactive web dashboard (React + Plotly Dash)
- [ ] Multi-user authentication
- [ ] Real-time collaborative breeding

### Version 369.0.0 (Ultimate Platform)

- [ ] Blockchain-based strain lineage tracking
- [ ] Federated learning for global model training
- [ ] Regulatory compliance automation (FDA, Health Canada)
- [ ] Clinical trial integration
- [ ] Open-source collaborative platform

---

## Code Location Reference

| Component | Language | Location | Lines |
|-----------|----------|----------|-------|
| Data Validation | Python | `src/data/validators.py` | 333 |
| VAE Model | Python | `src/models/vae.py` | 408 |
| Effect Predictor | Python | `src/models/effect_predictor.py` | 467 |
| Breeding Strategy | Python | `src/models/breeding_strategy.py` | 399 |
| Configuration | Python | `src/utils/config.py` | 74 |
| Visualization | Python | `src/utils/visualization.py` | 460 |
| Main API | Python | `src/phenohunter_scientific.py` | 559 |
| Testing Suite | Python | `src/tests/test_scientific_improvements.py` | 355 |
| **Total Python** | | | **3,055** |
| Phenotypic Analysis | C++ | `cpp/src/**/*.cpp` (Planned) | TBD |
| Data Integrity | Rust | `rust/src/**/*.rs` (Planned) | TBD |
| Scientific Computation | Julia | `julia/src/**/*.jl` (Planned) | TBD |

---

**Maintained by**: Hosuay & Contributors
**Last Updated**: 2025-10-22
**Version**: 3.0.0
