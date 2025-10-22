"""
BrAPI Adapter for C.C.R.O.P-PhenoHunt
=====================================

Provides BrAPI (Breeding API) compliance for interoperability with:
    - Breedbase: Open-source plant breeding database
    - BMS (Breeding Management System): CIMMYT/IBP breeding platform
    - PhytoOracle: High-throughput phenomics pipeline

Implements BrAPI v2.1 specification:
    https://brapi.org/specification

Sacred Geometry Integration:
    - 3 core operations: Load, Transform, Export
    - 9-field metadata schema
    - 27-dimensional trait embedding (when available)

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0
License: MIT
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BrAPIAdapter:
    """
    BrAPI v2.1 compliant adapter for phenotype/genotype data exchange.

    Attributes:
        version (str): BrAPI specification version
        metadata (dict): Experiment metadata with harmonic numerology
    """

    def __init__(self, version: str = "2.1", metadata: Optional[Dict] = None):
        """
        Initialize BrAPI adapter.

        Args:
            version: BrAPI specification version (default: "2.1")
            metadata: Optional metadata dict with experiment info
        """
        self.version = version
        self.metadata = metadata or self._create_default_metadata()
        logger.info(f"BrAPI Adapter initialized (v{version})")

    def _create_default_metadata(self) -> Dict[str, Any]:
        """Create default 9-field metadata schema (sacred geometry alignment)."""
        return {
            'timestamp': datetime.now().isoformat(),
            'source': 'C.C.R.O.P-PhenoHunt',
            'version': '3.0.0',
            'brapi_version': self.version,
            'data_license': 'CC-BY-4.0',
            'contact': 'https://github.com/Hosuay/C.C.R.O.P-PhenoHunt',
            'citation': 'Hosuay et al. (2025). C.C.R.O.P-PhenoHunt',
            'sacred_geometry_seed': None,  # To be populated if harmonic seeding used
            'reproducibility_hash': None   # SHA-256 of input data
        }


def load_brapi_traits(
    json_path: Union[str, Path],
    validate: bool = True,
    return_metadata: bool = False
) -> Union[pd.DataFrame, tuple]:
    """
    Load BrAPI-compliant trait observations from JSON.

    Parses BrAPI ObservationUnit or Phenotype endpoints and converts
    to pandas DataFrame for downstream analysis.

    Args:
        json_path: Path to BrAPI JSON file
        validate: Whether to validate against BrAPI schema
        return_metadata: If True, return (df, metadata) tuple

    Returns:
        DataFrame with columns: [plant_id, trait_name, value, unit, timestamp]
        Or tuple: (DataFrame, metadata_dict) if return_metadata=True

    Example:
        >>> df = load_brapi_traits('sample_traits.json')
        >>> print(df.head())
           plant_id trait_name    value  unit           timestamp
        0  P001     THC_content   22.5   %    2025-01-15T10:00:00
        1  P001     CBD_content    1.2   %    2025-01-15T10:00:00

    References:
        - BrAPI v2.1 Phenotyping: https://brapi.org/specification
        - Breedbase integration: https://breedbase.org
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"BrAPI JSON not found: {json_path}")

    logger.info(f"Loading BrAPI traits from: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Validate BrAPI structure
    if validate:
        _validate_brapi_structure(data)

    # Extract observation units
    observations = []

    # Handle different BrAPI response formats
    if 'result' in data:
        result = data['result']
        if 'data' in result:
            obs_units = result['data']
        else:
            obs_units = [result]
    else:
        obs_units = data if isinstance(data, list) else [data]

    # Parse each observation unit
    for obs_unit in obs_units:
        plant_id = obs_unit.get('observationUnitDbId', obs_unit.get('observationUnitName', 'unknown'))

        # Extract observations
        obs_list = obs_unit.get('observations', [])

        for obs in obs_list:
            observations.append({
                'plant_id': plant_id,
                'trait_name': obs.get('observationVariableName', obs.get('observationVariableDbId', 'unknown')),
                'value': obs.get('value'),
                'unit': obs.get('observationUnit', ''),
                'timestamp': obs.get('observationTimeStamp', datetime.now().isoformat()),
                'collector': obs.get('collector', ''),
                'season': obs.get('season', '')
            })

    df = pd.DataFrame(observations)

    # Convert numeric values
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    logger.info(f"Loaded {len(df)} trait observations for {df['plant_id'].nunique()} plants")

    # Extract metadata
    metadata = data.get('metadata', {})

    if return_metadata:
        return df, metadata
    return df


def export_pheno_predictions(
    predictions_df: pd.DataFrame,
    out_path: Union[str, Path],
    observation_level: str = "plant",
    study_db_id: Optional[str] = None,
    include_confidence: bool = True,
    metadata: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Export phenotype predictions as BrAPI ObservationUnit JSON.

    Converts model predictions to BrAPI v2.1 format for import into
    Breedbase, BMS, or other breeding management systems.

    Args:
        predictions_df: DataFrame with columns [plant_id, trait_name, predicted_value, (optional: std, confidence_lower, confidence_upper)]
        out_path: Output JSON file path
        observation_level: BrAPI observation level (default: "plant")
        study_db_id: Optional study database ID
        include_confidence: Whether to include confidence intervals
        metadata: Optional metadata dict (uses harmonic defaults if None)

    Returns:
        Dict with BrAPI-compliant JSON structure

    Example:
        >>> predictions = pd.DataFrame({
        ...     'plant_id': ['P001', 'P001', 'P002'],
        ...     'trait_name': ['THC_content', 'CBD_content', 'THC_content'],
        ...     'predicted_value': [22.5, 1.2, 18.3],
        ...     'confidence_lower': [20.1, 0.8, 16.5],
        ...     'confidence_upper': [24.9, 1.6, 20.1]
        ... })
        >>> result = export_pheno_predictions(predictions, 'predictions.json')

    References:
        - BrAPI ObservationUnits: https://brapi.org/specification#tag/Observation-Units
    """
    out_path = Path(out_path)

    # Initialize adapter with metadata
    adapter = BrAPIAdapter(metadata=metadata)

    # Group by plant_id
    plant_groups = predictions_df.groupby('plant_id')

    observation_units = []

    for plant_id, group in plant_groups:
        observations = []

        for _, row in group.iterrows():
            obs = {
                'observationDbId': f"{plant_id}_{row['trait_name']}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'observationUnitDbId': str(plant_id),
                'observationVariableDbId': row['trait_name'],
                'observationVariableName': row['trait_name'],
                'value': float(row['predicted_value']) if not pd.isna(row['predicted_value']) else None,
                'observationTimeStamp': datetime.now().isoformat(),
                'collector': 'C.C.R.O.P-PhenoHunt ML Pipeline',
                'season': metadata.get('season', '') if metadata else ''
            }

            # Add confidence intervals if available
            if include_confidence and 'confidence_lower' in row and 'confidence_upper' in row:
                obs['additionalInfo'] = {
                    'confidenceInterval': {
                        'lower': float(row['confidence_lower']) if not pd.isna(row['confidence_lower']) else None,
                        'upper': float(row['confidence_upper']) if not pd.isna(row['confidence_upper']) else None,
                        'level': 0.95  # 95% confidence interval
                    }
                }

                # Include standard deviation if available
                if 'std' in row:
                    obs['additionalInfo']['standardDeviation'] = float(row['std']) if not pd.isna(row['std']) else None

            observations.append(obs)

        # Create observation unit
        obs_unit = {
            'observationUnitDbId': str(plant_id),
            'observationUnitName': str(plant_id),
            'observationLevel': observation_level,
            'observations': observations
        }

        if study_db_id:
            obs_unit['studyDbId'] = study_db_id

        observation_units.append(obs_unit)

    # Construct BrAPI response
    brapi_response = {
        '@context': ['https://brapi.org/jsonld/context/metadata.jsonld'],
        'metadata': {
            'pagination': {
                'pageSize': len(observation_units),
                'currentPage': 0,
                'totalCount': len(observation_units),
                'totalPages': 1
            },
            'status': [],
            'datafiles': []
        },
        'result': {
            'data': observation_units
        }
    }

    # Add custom metadata
    brapi_response['metadata']['additionalInfo'] = adapter.metadata

    # Write to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(brapi_response, f, indent=2)

    logger.info(f"Exported {len(observation_units)} observation units to {out_path}")

    return brapi_response


def create_brapi_observation_unit(
    plant_id: str,
    trait_dict: Dict[str, float],
    study_db_id: Optional[str] = None,
    observation_level: str = "plant",
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a single BrAPI ObservationUnit object.

    Args:
        plant_id: Unique plant identifier
        trait_dict: Dictionary mapping trait names to values
        study_db_id: Optional study database ID
        observation_level: Observation level (default: "plant")
        timestamp: ISO timestamp (uses current time if None)

    Returns:
        BrAPI-compliant observation unit dict

    Example:
        >>> obs_unit = create_brapi_observation_unit(
        ...     plant_id='P001',
        ...     trait_dict={'THC': 22.5, 'CBD': 1.2, 'Myrcene': 0.6}
        ... )
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    observations = []
    for trait_name, value in trait_dict.items():
        obs = {
            'observationDbId': f"{plant_id}_{trait_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'observationUnitDbId': plant_id,
            'observationVariableDbId': trait_name,
            'observationVariableName': trait_name,
            'value': float(value) if not pd.isna(value) else None,
            'observationTimeStamp': timestamp
        }
        observations.append(obs)

    obs_unit = {
        'observationUnitDbId': plant_id,
        'observationUnitName': plant_id,
        'observationLevel': observation_level,
        'observations': observations
    }

    if study_db_id:
        obs_unit['studyDbId'] = study_db_id

    return obs_unit


def validate_brapi_json(json_path: Union[str, Path]) -> tuple[bool, List[str]]:
    """
    Validate BrAPI JSON structure.

    Args:
        json_path: Path to BrAPI JSON file

    Returns:
        Tuple of (is_valid, list_of_errors)

    Example:
        >>> is_valid, errors = validate_brapi_json('sample.json')
        >>> if not is_valid:
        ...     print("Validation errors:", errors)
    """
    json_path = Path(json_path)
    errors = []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return False, [f"File not found: {json_path}"]

    # Check for required BrAPI fields
    if 'metadata' not in data and 'result' not in data:
        errors.append("Missing required 'metadata' or 'result' fields (not BrAPI compliant)")

    # Validate observation units if present
    if 'result' in data:
        result = data['result']
        if 'data' in result:
            obs_units = result['data']
            for i, obs_unit in enumerate(obs_units):
                if 'observationUnitDbId' not in obs_unit:
                    errors.append(f"Observation unit {i} missing 'observationUnitDbId'")
                if 'observations' not in obs_unit:
                    errors.append(f"Observation unit {i} missing 'observations' array")

    is_valid = len(errors) == 0

    if is_valid:
        logger.info(f"BrAPI validation passed: {json_path}")
    else:
        logger.warning(f"BrAPI validation failed with {len(errors)} errors")

    return is_valid, errors


def _validate_brapi_structure(data: Dict) -> None:
    """Internal validation for BrAPI structure."""
    if not isinstance(data, dict):
        raise ValueError("BrAPI data must be a JSON object (dict)")

    # Check for basic BrAPI structure (flexible for different endpoints)
    has_result = 'result' in data
    has_metadata = 'metadata' in data

    if not (has_result or has_metadata):
        logger.warning("Data may not be BrAPI compliant (missing 'result' or 'metadata')")


# Command-line interface
def main():
    """CLI for BrAPI adapter operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description='BrAPI Adapter for C.C.R.O.P-PhenoHunt',
        epilog='Example: python -m src.io.brapi_adapter import --in sample.json --out predictions.csv'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import BrAPI JSON to DataFrame')
    import_parser.add_argument('--in', dest='input', required=True, help='Input BrAPI JSON file')
    import_parser.add_argument('--out', dest='output', required=True, help='Output CSV file')
    import_parser.add_argument('--validate', action='store_true', help='Validate BrAPI structure')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export predictions to BrAPI JSON')
    export_parser.add_argument('--in', dest='input', required=True, help='Input predictions CSV')
    export_parser.add_argument('--out', dest='output', required=True, help='Output BrAPI JSON file')
    export_parser.add_argument('--study-id', help='Study database ID')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate BrAPI JSON')
    validate_parser.add_argument('file', help='BrAPI JSON file to validate')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.command == 'import':
        # Load BrAPI and export to CSV
        df = load_brapi_traits(args.input, validate=args.validate)
        df.to_csv(args.output, index=False)
        print(f"✓ Imported {len(df)} observations to {args.output}")

    elif args.command == 'export':
        # Load CSV and export to BrAPI
        df = pd.read_csv(args.input)
        required_cols = ['plant_id', 'trait_name', 'predicted_value']
        if not all(col in df.columns for col in required_cols):
            print(f"✗ Error: CSV must have columns: {required_cols}")
            return

        export_pheno_predictions(df, args.output, study_db_id=args.study_id)
        print(f"✓ Exported to {args.output}")

    elif args.command == 'validate':
        is_valid, errors = validate_brapi_json(args.file)
        if is_valid:
            print(f"✓ {args.file} is valid BrAPI JSON")
        else:
            print(f"✗ {args.file} has validation errors:")
            for error in errors:
                print(f"  - {error}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
