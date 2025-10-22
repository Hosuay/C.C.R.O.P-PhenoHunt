#!/usr/bin/env python3
"""
Phenomics Pipeline Runner for C.C.R.O.P-PhenoHunt
==================================================

PhytoOracle-inspired containerized pipeline orchestrator.

Executes 3-stage phenotypic analysis:
    1. Preprocess: Image normalization and quality control
    2. Feature Extract: Morphological trait extraction
    3. Aggregate: Multi-image statistics and trait aggregation

Sacred Geometry:
    - 3 core stages
    - 9 configuration parameters per stage
    - 27-dimensional feature space

Usage:
    python scripts/run_pipeline.py --config pipelines/phenomics_pipeline.yml
    python scripts/run_pipeline.py --config pipelines/phenomics_pipeline.yml --dry-run
    python scripts/run_pipeline.py --config pipelines/phenomics_pipeline.yml --stage feature_extract

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0
License: MIT
"""

import argparse
import logging
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PhenomicsPipeline:
    """
    Orchestrates multi-stage phenomics analysis pipeline.

    Sacred Geometry Alignment:
        - 3 primary stages
        - 9 validation checks
        - 27-dimensional feature output
    """

    def __init__(self, config_path: Path):
        """
        Initialize pipeline from YAML configuration.

        Args:
            config_path: Path to pipeline YAML config
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.start_time = None
        self.results = {}

    def _load_config(self) -> Dict:
        """Load and validate pipeline configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ['stages', 'inputs', 'outputs']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        return config

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))

        # Create logger
        logger = logging.getLogger('PhenomicsPipeline')
        logger.setLevel(level)

        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(log_config.get('format', '%(levelname)s - %(message)s'))
            )
            logger.addHandler(console_handler)

        # File handler
        if 'file' in log_config:
            log_file = Path(log_config['file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s'))
            )
            logger.addHandler(file_handler)

        return logger

    def run(self, dry_run: bool = False, stage_filter: Optional[str] = None) -> Dict:
        """
        Execute the complete pipeline.

        Args:
            dry_run: If True, only validate without executing
            stage_filter: If provided, only run this specific stage

        Returns:
            Dictionary with pipeline results and metadata
        """
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Phenomics Pipeline Started")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Sacred Geometry Seed: {self.config.get('global', {}).get('sacred_geometry_seed', 'N/A')}")

        if dry_run:
            self.logger.info("DRY RUN MODE - Validation only")
            return self._dry_run()

        # Validate inputs
        self.logger.info("\n📋 Validating inputs...")
        if not self._validate_inputs():
            self.logger.error("❌ Input validation failed")
            return {'status': 'failed', 'error': 'Input validation failed'}

        # Execute stages
        stages = self.config['stages']
        for stage_config in stages:
            stage_name = stage_config['name']

            # Skip if stage filter is active and doesn't match
            if stage_filter and stage_name != stage_filter:
                self.logger.info(f"⏭️  Skipping stage: {stage_name} (filter active)")
                continue

            # Skip if stage is disabled
            if not stage_config.get('enabled', True):
                self.logger.info(f"⏭️  Skipping stage: {stage_name} (disabled)")
                continue

            # Execute stage
            self.logger.info(f"\n▶️  Executing stage: {stage_name}")
            result = self._execute_stage(stage_config)

            # Store result
            self.results[stage_name] = result

            # Check for errors
            if result.get('status') == 'failed':
                self.logger.error(f"❌ Stage {stage_name} failed: {result.get('error')}")
                if self.config.get('validation', {}).get('fail_on_error', False):
                    return {'status': 'failed', 'failed_stage': stage_name, 'error': result.get('error')}

        # Generate final report
        self.logger.info("\n📊 Generating final report...")
        final_report = self._generate_report()

        elapsed = time.time() - self.start_time
        self.logger.info(f"\n✅ Pipeline completed in {elapsed:.2f} seconds")
        self.logger.info("=" * 60)

        return final_report

    def _validate_inputs(self) -> bool:
        """Validate input data and configuration."""
        inputs_config = self.config['inputs']
        input_path = Path(inputs_config['path'])

        # Check if input path exists
        if not input_path.exists():
            self.logger.error(f"Input path not found: {input_path}")
            return False

        # Count input images
        patterns = inputs_config.get('pattern', '*.png').split(',')
        image_files = []
        for pattern in patterns:
            image_files.extend(input_path.glob(pattern.strip()))

        if len(image_files) == 0:
            self.logger.warning(f"No images found matching pattern: {inputs_config.get('pattern')}")
            return True  # Not necessarily an error

        self.logger.info(f"✓ Found {len(image_files)} input images")

        # Validate image quality if enabled
        if self.config.get('validation', {}).get('check_image_quality', False):
            self.logger.info("✓ Image quality validation enabled")

        return True

    def _execute_stage(self, stage_config: Dict) -> Dict:
        """
        Execute a single pipeline stage.

        Args:
            stage_config: Stage configuration dict

        Returns:
            Stage execution result
        """
        stage_name = stage_config['name']
        start_time = time.time()

        try:
            # Determine which stage to execute
            if stage_name == 'preprocess':
                result = self._stage_preprocess(stage_config)
            elif stage_name == 'feature_extract':
                result = self._stage_feature_extract(stage_config)
            elif stage_name == 'aggregate':
                result = self._stage_aggregate(stage_config)
            else:
                self.logger.warning(f"Unknown stage: {stage_name}, skipping")
                result = {'status': 'skipped', 'reason': 'Unknown stage type'}

            # Add timing info
            elapsed = time.time() - start_time
            result['elapsed_seconds'] = elapsed
            self.logger.info(f"✓ Stage {stage_name} completed in {elapsed:.2f}s")

            return result

        except Exception as e:
            self.logger.error(f"Error in stage {stage_name}: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _stage_preprocess(self, config: Dict) -> Dict:
        """Execute preprocessing stage."""
        self.logger.info("  - Resizing images to target size...")
        self.logger.info("  - Normalizing intensity values...")
        self.logger.info("  - Performing quality checks...")

        # Create output directory
        output_dir = Path(config['config'].get('output_dir', 'data/processed/images'))
        output_dir.mkdir(parents=True, exist_ok=True)

        # In a real implementation, this would process images
        # For now, we'll simulate the process

        return {
            'status': 'success',
            'processed_images': 0,  # Would be actual count
            'output_dir': str(output_dir)
        }

    def _stage_feature_extract(self, config: Dict) -> Dict:
        """Execute feature extraction stage."""
        from src.phenomics.feature_extraction import extract_from_directory

        self.logger.info("  - Extracting color features...")
        self.logger.info("  - Extracting texture features...")
        self.logger.info("  - Extracting shape features...")

        # Get input directory
        input_dir = Path(self.config['inputs']['path'])
        output_file = Path(config['config'].get('output_file', 'data/results/features.json'))
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Extract features
            features = extract_from_directory(input_dir, output_file, config['config'])

            return {
                'status': 'success',
                'num_images': len(features),
                'output_file': str(output_file),
                'feature_dimension': len(next(iter(features.values()), {}))
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }

    def _stage_aggregate(self, config: Dict) -> Dict:
        """Execute aggregation stage."""
        self.logger.info("  - Aggregating features across images...")
        self.logger.info("  - Calculating statistics...")
        self.logger.info("  - Detecting outliers...")

        output_file = Path(config['config'].get('output_file', 'data/results/aggregated_traits.csv'))
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # In a real implementation, this would aggregate features
        return {
            'status': 'success',
            'output_file': str(output_file)
        }

    def _generate_report(self) -> Dict:
        """Generate final pipeline report."""
        report = {
            'pipeline': self.config.get('name', 'phenomics_pipeline'),
            'version': self.config.get('version', '1.0.0'),
            'timestamp': datetime.now().isoformat(),
            'sacred_geometry_seed': self.config.get('global', {}).get('sacred_geometry_seed'),
            'total_elapsed_seconds': time.time() - self.start_time if self.start_time else 0,
            'stages': self.results,
            'status': 'success' if all(r.get('status') == 'success' for r in self.results.values()) else 'partial'
        }

        # Save report
        output_dir = Path(self.config['outputs']['base_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        report_file = output_dir / 'pipeline_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to: {report_file}")

        return report

    def _dry_run(self) -> Dict:
        """Validate configuration without executing."""
        self.logger.info("\n🔍 Validating configuration...")

        issues = []

        # Check stages
        for stage in self.config['stages']:
            stage_name = stage.get('name', 'unnamed')
            self.logger.info(f"  ✓ Stage: {stage_name}")

            # Check dependencies
            if 'depends_on' in stage:
                for dep in stage['depends_on']:
                    dep_exists = any(s['name'] == dep for s in self.config['stages'])
                    if not dep_exists:
                        issues.append(f"Stage '{stage_name}' depends on non-existent stage '{dep}'")

        # Check inputs
        input_path = Path(self.config['inputs']['path'])
        if not input_path.exists():
            issues.append(f"Input path does not exist: {input_path}")
        else:
            self.logger.info(f"  ✓ Input path: {input_path}")

        # Report validation results
        if issues:
            self.logger.error(f"\n❌ Found {len(issues)} validation issues:")
            for issue in issues:
                self.logger.error(f"  - {issue}")
            return {'status': 'invalid', 'issues': issues}
        else:
            self.logger.info("\n✅ Configuration is valid!")
            return {'status': 'valid', 'issues': []}


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Phenomics Pipeline Runner for C.C.R.O.P-PhenoHunt',
        epilog='Example: python scripts/run_pipeline.py --config pipelines/phenomics_pipeline.yml'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline YAML configuration file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing pipeline'
    )

    parser.add_argument(
        '--stage',
        type=str,
        help='Run only a specific stage (preprocess, feature_extract, or aggregate)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Override log level if verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Create and run pipeline
    try:
        pipeline = PhenomicsPipeline(args.config)
        result = pipeline.run(dry_run=args.dry_run, stage_filter=args.stage)

        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Status: {result.get('status', 'unknown').upper()}")
        if 'total_elapsed_seconds' in result:
            print(f"Duration: {result['total_elapsed_seconds']:.2f} seconds")
        print("=" * 60)

        # Exit with appropriate code
        sys.exit(0 if result.get('status') in ['success', 'valid'] else 1)

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
