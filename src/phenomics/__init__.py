"""
Phenomics Module for C.C.R.O.P-PhenoHunt
========================================

High-throughput image-based trait extraction pipeline inspired by PhytoOracle.

Modules:
    - feature_extraction: Extract morphological features from plant images
    - preprocessing: Image normalization and quality control
    - aggregation: Multi-image trait aggregation and statistics

Sacred Geometry Alignment:
    - 3-stage pipeline: Preprocess → Extract → Aggregate
    - 9 core phenotypic traits tracked
    - 27-dimensional feature embedding space
"""

__version__ = "1.0.0"
__all__ = ['FeatureExtractor', 'ImagePreprocessor', 'TraitAggregator']
