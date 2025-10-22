"""
I/O Module for C.C.R.O.P-PhenoHunt
==================================

Handles data import/export for various formats and APIs.

Modules:
    - brapi_adapter: BrAPI-compliant data exchange for Breedbase/BMS integration

Sacred Geometry Alignment:
    - 3 primary I/O operations: Load, Transform, Export
    - 9-field metadata schema for full reproducibility
"""

from .brapi_adapter import (
    load_brapi_traits,
    export_pheno_predictions,
    create_brapi_observation_unit,
    validate_brapi_json
)

__all__ = [
    'load_brapi_traits',
    'export_pheno_predictions',
    'create_brapi_observation_unit',
    'validate_brapi_json'
]

__version__ = "1.0.0"
