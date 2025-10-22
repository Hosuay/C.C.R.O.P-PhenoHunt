"""
Cheminformatics Module for C.C.R.O.P-PhenoHunt

This module provides molecular descriptor computation and chemical structure analysis
for cannabis compounds using RDKit and other cheminformatics libraries.

Sacred Geometry Alignment: 33 molecular descriptors (master number)
"""

from typing import Dict, List, Optional
import logging

__version__ = "3.0.0"
__author__ = "Hosuay & Contributors"

# Configure module logger
logger = logging.getLogger(__name__)

__all__ = [
    "molecular_descriptors",
    "structure_generator",
    "chemical_properties"
]
