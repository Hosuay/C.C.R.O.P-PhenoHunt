"""
Experiment Metadata Module
===========================

Comprehensive metadata tracking for reproducible experiments.

Sacred Geometry:
    - 9-field minimum metadata schema
    - 27-field extended metadata for full reproducibility

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import platform
import sys

logger = logging.getLogger(__name__)


class ExperimentMetadata:
    """
    Tracks experiment metadata for full reproducibility.

    Sacred Geometry: 9-field minimum schema.
    """

    def __init__(self, experiment_name: str, description: str = ""):
        """
        Initialize experiment metadata.

        Args:
            experiment_name: Name of experiment
            description: Brief description
        """
        self.experiment_name = experiment_name
        self.description = description
        self.start_time = datetime.now()
        self.metadata = self._initialize_metadata()

    def _initialize_metadata(self) -> Dict[str, Any]:
        """Initialize 9-field minimum metadata schema."""
        return {
            # Core fields (9 minimum)
            'experiment_name': self.experiment_name,
            'description': self.description,
            'timestamp': self.start_time.isoformat(),
            'version': '3.0.0',
            'platform': platform.system(),
            'python_version': sys.version,
            'sacred_geometry_seed': None,
            'data_hash': None,
            'git_commit': self._get_git_commit()
        }

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except Exception:
            return None

    def add_parameter(self, key: str, value: Any) -> None:
        """Add experiment parameter."""
        if 'parameters' not in self.metadata:
            self.metadata['parameters'] = {}
        self.metadata['parameters'][key] = value

    def add_harmonic_seed(self, seed: int, numerology_params: Optional[Dict] = None) -> None:
        """
        Add harmonic seed and numerology parameters.

        Args:
            seed: Harmonic seed value
            numerology_params: Parameters used to generate seed
        """
        self.metadata['sacred_geometry_seed'] = seed
        if numerology_params:
            self.metadata['numerology_params'] = numerology_params

        logger.info(f"Harmonic seed added: {seed}")

    def add_data_hash(self, data_path: Optional[Path] = None, data_content: Optional[str] = None) -> str:
        """
        Add data hash for reproducibility.

        Args:
            data_path: Path to data file
            data_content: Or direct data content

        Returns:
            SHA-256 hash string
        """
        if data_path:
            with open(data_path, 'rb') as f:
                data_bytes = f.read()
        elif data_content:
            data_bytes = data_content.encode()
        else:
            raise ValueError("Must provide data_path or data_content")

        hash_obj = hashlib.sha256(data_bytes)
        data_hash = hash_obj.hexdigest()

        self.metadata['data_hash'] = f"sha256:{data_hash}"
        logger.info(f"Data hash added: {data_hash[:16]}...")

        return data_hash

    def add_docker_image(self, image_name: str, image_tag: str = "latest") -> None:
        """Add Docker image info for containerized reproducibility."""
        self.metadata['docker_image'] = f"{image_name}:{image_tag}"

    def add_results(self, results: Dict[str, Any]) -> None:
        """Add experiment results."""
        self.metadata['results'] = results

    def add_metrics(self, metrics: Dict[str, float]) -> None:
        """Add performance metrics."""
        self.metadata['metrics'] = metrics

    def finalize(self) -> Dict[str, Any]:
        """Finalize metadata and add duration."""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['duration_seconds'] = (datetime.now() - self.start_time).total_seconds()

        return self.metadata

    def save(self, output_path: Path) -> None:
        """
        Save metadata to JSON file.

        Args:
            output_path: Output JSON file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_metadata = self.finalize()

        with open(output_path, 'w') as f:
            json.dump(final_metadata, f, indent=2, default=str)

        logger.info(f"Metadata saved to {output_path}")

    def get_extended_metadata(self) -> Dict[str, Any]:
        """
        Get extended 27-field metadata for comprehensive tracking.

        Sacred Geometry: 27 fields (3³).
        """
        extended = self.metadata.copy()

        # Add 18 more fields for total of 27
        try:
            import torch
            extended['pytorch_version'] = torch.__version__
            extended['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                extended['cuda_version'] = torch.version.cuda
                extended['gpu_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            extended['pytorch_version'] = None
            extended['cuda_available'] = False

        try:
            import numpy as np
            extended['numpy_version'] = np.__version__
        except ImportError:
            extended['numpy_version'] = None

        try:
            import pandas as pd
            extended['pandas_version'] = pd.__version__
        except ImportError:
            extended['pandas_version'] = None

        # System info
        extended['cpu_count'] = platform.processor()
        extended['architecture'] = platform.machine()

        # Add more fields as needed
        extended['hostname'] = platform.node()

        return extended


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Experiment Metadata Demo")
    print("=" * 60)

    # Create metadata
    exp = ExperimentMetadata(
        experiment_name="F1_Hybrid_VAE_Training",
        description="Train VAE on F1 hybrid population with harmonic hyperparameters"
    )

    # Add parameters
    exp.add_parameter('latent_dim', 27)
    exp.add_parameter('epochs', 369)
    exp.add_parameter('batch_size', 9)

    # Add harmonic seed
    exp.add_harmonic_seed(
        seed=123456789,
        numerology_params={'latent_dim': 27, 'epochs': 369}
    )

    # Add data hash
    exp.add_data_hash(data_content="Sample training data")

    # Add Docker info
    exp.add_docker_image('crop-phenohunt/vae', 'v3.0.0')

    # Add results
    exp.add_metrics({
        'train_loss': 0.123,
        'val_loss': 0.156,
        'r2_score': 0.85
    })

    # Save
    exp.save(Path('/tmp/experiment_metadata.json'))

    print("\n9-field minimum metadata:")
    for key in ['experiment_name', 'timestamp', 'version', 'platform',
                'python_version', 'sacred_geometry_seed', 'data_hash',
                'git_commit', 'docker_image']:
        if key in exp.metadata:
            print(f"  {key}: {exp.metadata[key]}")

    print(f"\n✓ Metadata saved to /tmp/experiment_metadata.json")
    print("\n✓ Experiment metadata demo complete")
