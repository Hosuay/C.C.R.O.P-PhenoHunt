"""
Configuration Management
"""

import yaml
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate scientific configuration."""

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config YAML file

        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")

        # Validate
        ConfigLoader._validate_config(config)

        return config

    @staticmethod
    def _validate_config(config: Dict):
        """Validate configuration structure."""
        required_sections = ['compounds', 'effects', 'model', 'validation', 'breeding']

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        logger.info("Configuration validation passed")

    @staticmethod
    def get_feature_list(config: Dict) -> list:
        """Extract list of all chemical features from config."""
        features = []

        for category in ['cannabinoids', 'terpenes']:
            if category in config['compounds']:
                for level in ['major', 'minor']:
                    if level in config['compounds'][category]:
                        for compound in config['compounds'][category][level]:
                            features.append(f"{compound['name']}_pct")

        return features

    @staticmethod
    def get_compound_info(config: Dict, compound_name: str) -> Dict:
        """Get full information for a compound."""
        for category in ['cannabinoids', 'terpenes']:
            for level in ['major', 'minor']:
                for compound in config['compounds'][category][level]:
                    if compound['name'] == compound_name.replace('_pct', ''):
                        return compound

        return None
