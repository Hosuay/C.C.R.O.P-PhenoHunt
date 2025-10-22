"""
Unit Tests for BrAPI Adapter
=============================

Tests BrAPI v2.1 compliance for import/export operations.

Sacred Geometry Alignment:
    - 3 test categories: Load, Export, Validate
    - 9 total test cases for comprehensive coverage

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0
"""

import pytest
import json
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime

from src.io.brapi_adapter import (
    load_brapi_traits,
    export_pheno_predictions,
    create_brapi_observation_unit,
    validate_brapi_json,
    BrAPIAdapter
)


class TestBrAPIAdapter:
    """Test suite for BrAPI adapter functionality."""

    @pytest.fixture
    def sample_brapi_json(self, tmp_path):
        """Create sample BrAPI JSON for testing."""
        data = {
            "metadata": {
                "pagination": {"pageSize": 2, "currentPage": 0, "totalCount": 2, "totalPages": 1},
                "status": [],
                "datafiles": []
            },
            "result": {
                "data": [
                    {
                        "observationUnitDbId": "P001",
                        "observationUnitName": "Plant_001",
                        "observationLevel": "plant",
                        "observations": [
                            {
                                "observationDbId": "OBS001",
                                "observationVariableDbId": "THC",
                                "observationVariableName": "THC_content",
                                "value": "22.5",
                                "observationUnit": "%",
                                "observationTimeStamp": "2025-01-15T10:00:00Z"
                            },
                            {
                                "observationDbId": "OBS002",
                                "observationVariableDbId": "CBD",
                                "observationVariableName": "CBD_content",
                                "value": "1.2",
                                "observationUnit": "%",
                                "observationTimeStamp": "2025-01-15T10:00:00Z"
                            }
                        ]
                    },
                    {
                        "observationUnitDbId": "P002",
                        "observationUnitName": "Plant_002",
                        "observationLevel": "plant",
                        "observations": [
                            {
                                "observationDbId": "OBS003",
                                "observationVariableDbId": "THC",
                                "observationVariableName": "THC_content",
                                "value": "18.3",
                                "observationUnit": "%",
                                "observationTimeStamp": "2025-01-15T10:00:00Z"
                            }
                        ]
                    }
                ]
            }
        }

        json_file = tmp_path / "sample_brapi.json"
        with open(json_file, 'w') as f:
            json.dump(data, f)

        return json_file

    @pytest.fixture
    def sample_predictions_df(self):
        """Create sample predictions DataFrame."""
        return pd.DataFrame({
            'plant_id': ['P001', 'P001', 'P002'],
            'trait_name': ['THC_content', 'CBD_content', 'THC_content'],
            'predicted_value': [22.5, 1.2, 18.3],
            'std': [1.5, 0.3, 2.1],
            'confidence_lower': [20.1, 0.8, 16.5],
            'confidence_upper': [24.9, 1.6, 20.1]
        })

    # ========== Load Tests (3 tests) ==========

    def test_load_brapi_traits_basic(self, sample_brapi_json):
        """Test basic loading of BrAPI traits."""
        df = load_brapi_traits(sample_brapi_json, validate=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 2 observations for P001, 1 for P002
        assert 'plant_id' in df.columns
        assert 'trait_name' in df.columns
        assert 'value' in df.columns
        assert df['plant_id'].nunique() == 2

    def test_load_brapi_traits_with_metadata(self, sample_brapi_json):
        """Test loading with metadata return."""
        df, metadata = load_brapi_traits(sample_brapi_json, return_metadata=True)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert len(df) == 3

    def test_load_brapi_traits_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_brapi_traits('nonexistent_file.json')

    # ========== Export Tests (3 tests) ==========

    def test_export_pheno_predictions_basic(self, sample_predictions_df, tmp_path):
        """Test basic export of phenotype predictions."""
        out_file = tmp_path / "predictions.json"
        result = export_pheno_predictions(sample_predictions_df, out_file)

        # Check file was created
        assert out_file.exists()

        # Check structure
        assert 'metadata' in result
        assert 'result' in result
        assert 'data' in result['result']

        # Check observation units
        obs_units = result['result']['data']
        assert len(obs_units) == 2  # P001 and P002

        # Check confidence intervals
        first_obs = obs_units[0]['observations'][0]
        assert 'additionalInfo' in first_obs
        assert 'confidenceInterval' in first_obs['additionalInfo']

    def test_export_pheno_predictions_no_confidence(self, tmp_path):
        """Test export without confidence intervals."""
        df = pd.DataFrame({
            'plant_id': ['P001'],
            'trait_name': ['THC_content'],
            'predicted_value': [22.5]
        })

        out_file = tmp_path / "predictions_no_ci.json"
        result = export_pheno_predictions(df, out_file, include_confidence=False)

        assert out_file.exists()
        obs = result['result']['data'][0]['observations'][0]
        assert 'additionalInfo' not in obs or 'confidenceInterval' not in obs.get('additionalInfo', {})

    def test_export_with_study_id(self, sample_predictions_df, tmp_path):
        """Test export with study database ID."""
        out_file = tmp_path / "predictions_study.json"
        study_id = "STUDY_2025_001"

        result = export_pheno_predictions(
            sample_predictions_df,
            out_file,
            study_db_id=study_id
        )

        # Check study ID is included
        for obs_unit in result['result']['data']:
            assert 'studyDbId' in obs_unit
            assert obs_unit['studyDbId'] == study_id

    # ========== Utility Tests (3 tests) ==========

    def test_create_observation_unit(self):
        """Test creation of single observation unit."""
        trait_dict = {'THC': 22.5, 'CBD': 1.2, 'Myrcene': 0.6}
        obs_unit = create_brapi_observation_unit('P001', trait_dict)

        assert obs_unit['observationUnitDbId'] == 'P001'
        assert len(obs_unit['observations']) == 3
        assert all('observationVariableName' in obs for obs in obs_unit['observations'])

    def test_validate_brapi_json_valid(self, sample_brapi_json):
        """Test validation of valid BrAPI JSON."""
        is_valid, errors = validate_brapi_json(sample_brapi_json)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_brapi_json_invalid(self, tmp_path):
        """Test validation of invalid BrAPI JSON."""
        invalid_json = tmp_path / "invalid.json"
        with open(invalid_json, 'w') as f:
            json.dump({"invalid": "structure"}, f)

        is_valid, errors = validate_brapi_json(invalid_json)

        assert is_valid is False
        assert len(errors) > 0


class TestBrAPIAdapterClass:
    """Test BrAPIAdapter class functionality."""

    def test_adapter_initialization(self):
        """Test adapter initialization with default metadata."""
        adapter = BrAPIAdapter()

        assert adapter.version == "2.1"
        assert isinstance(adapter.metadata, dict)
        assert 'timestamp' in adapter.metadata
        assert 'source' in adapter.metadata
        assert adapter.metadata['source'] == 'C.C.R.O.P-PhenoHunt'

    def test_adapter_custom_metadata(self):
        """Test adapter with custom metadata."""
        custom_meta = {
            'experiment': 'F1_Hybrid_Generation',
            'sacred_geometry_seed': 369
        }
        adapter = BrAPIAdapter(metadata=custom_meta)

        assert adapter.metadata == custom_meta

    def test_default_metadata_schema(self):
        """Test that default metadata follows 9-field schema (sacred geometry)."""
        adapter = BrAPIAdapter()
        metadata = adapter.metadata

        # Check 9 required fields
        required_fields = [
            'timestamp', 'source', 'version', 'brapi_version',
            'data_license', 'contact', 'citation',
            'sacred_geometry_seed', 'reproducibility_hash'
        ]

        for field in required_fields:
            assert field in metadata, f"Missing required metadata field: {field}"


class TestBrAPIIntegration:
    """Integration tests for full BrAPI workflow."""

    def test_round_trip_import_export(self, sample_brapi_json, tmp_path):
        """Test round-trip: import BrAPI → export BrAPI."""
        # Load original data
        df_original = load_brapi_traits(sample_brapi_json)

        # Rename columns to match prediction format
        df_predictions = df_original.rename(columns={'value': 'predicted_value'})
        df_predictions['confidence_lower'] = df_predictions['predicted_value'] * 0.9
        df_predictions['confidence_upper'] = df_predictions['predicted_value'] * 1.1

        # Export
        out_file = tmp_path / "roundtrip.json"
        result = export_pheno_predictions(df_predictions, out_file)

        # Verify export
        assert out_file.exists()
        assert len(result['result']['data']) == df_original['plant_id'].nunique()

        # Re-import and compare
        df_reimported = load_brapi_traits(out_file)
        assert len(df_reimported) == len(df_original)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
