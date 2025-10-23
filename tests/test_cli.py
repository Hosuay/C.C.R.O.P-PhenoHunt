#!/usr/bin/env python3
"""
Tests for PhenoHunter CLI.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestCLIUtils(unittest.TestCase):
    """Test CLI utility functions."""

    def test_colors_import(self):
        """Test that Colors class imports correctly."""
        from cli_utils import Colors
        self.assertIsNotNone(Colors.RED)
        self.assertIsNotNone(Colors.GREEN)

    def test_print_functions(self):
        """Test print utility functions."""
        from cli_utils import print_success, print_error, print_warning, print_info

        # These should not raise exceptions
        print_success("Test success")
        print_error("Test error")
        print_warning("Test warning")
        print_info("Test info")

    def test_format_chemical(self):
        """Test chemical formatting."""
        from cli_utils import format_chemical

        result = format_chemical("THC", 20.5)
        self.assertIn("THC", result)
        self.assertIn("20.50", result)

        result_with_std = format_chemical("CBD", 1.2, 0.3)
        self.assertIn("CBD", result_with_std)
        self.assertIn("1.20", result_with_std)
        self.assertIn("0.30", result_with_std)


class TestVersion(unittest.TestCase):
    """Test version module."""

    def test_version_import(self):
        """Test version information imports."""
        from __version__ import __version__, VERSION_INFO

        self.assertIsInstance(__version__, str)
        self.assertIsInstance(VERSION_INFO, dict)
        self.assertEqual(VERSION_INFO['version'], __version__)


class TestPhenoHunter(unittest.TestCase):
    """Test core PhenoHunter functionality."""

    def test_import(self):
        """Test that PhenoHunter can be imported."""
        from phenohunter_scientific import create_phenohunter

        # Should not raise an exception
        ph = create_phenohunter()
        self.assertIsNotNone(ph)

    def test_feature_columns(self):
        """Test that feature columns are defined."""
        from phenohunter_scientific import create_phenohunter

        ph = create_phenohunter()
        # Feature columns should be None initially
        self.assertIsNone(ph.feature_columns)


class TestDataValidation(unittest.TestCase):
    """Test data validation functions."""

    def setUp(self):
        """Set up test fixtures."""
        import pandas as pd
        from phenohunter_scientific import create_phenohunter

        self.ph = create_phenohunter()

        # Create sample data
        self.sample_data = pd.DataFrame([
            {
                'strain_name': 'Test Strain',
                'type': 'hybrid',
                'thc_pct': 20.0,
                'cbd_pct': 1.0,
                'cbg_pct': 0.5,
                'cbc_pct': 0.2,
                'cbda_pct': 0.1,
                'thcv_pct': 0.3,
                'cbn_pct': 0.4,
                'delta8_thc_pct': 0.1,
                'thca_pct': 18.0,
                'myrcene_pct': 1.0,
                'limonene_pct': 0.5,
                'pinene_pct': 0.3,
                'linalool_pct': 0.2,
                'caryophyllene_pct': 0.4,
                'humulene_pct': 0.2,
                'terpinolene_pct': 0.1,
                'ocimene_pct': 0.05,
                'camphene_pct': 0.03,
                'bisabolol_pct': 0.08
            }
        ])

    def test_load_strain_database(self):
        """Test loading strain database."""
        warnings = self.ph.load_strain_database(self.sample_data, validate=True)

        self.assertIsInstance(warnings, dict)
        self.assertIsNotNone(self.ph.strain_data)
        self.assertEqual(len(self.ph.strain_data), 1)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
