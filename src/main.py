#!/usr/bin/env python3
"""
Main entry point for PhenoHunter Scientific Platform.

This module provides a unified entry point for all PhenoHunter functionality,
whether running as CLI, API, or programmatically.

Sacred Geometry Integration: 3-6-9 numerology maintained throughout
"""

import sys
import os
from pathlib import Path
import argparse
import logging
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from __version__ import __version__, VERSION_INFO
from cli_utils import (
    Colors, print_banner, print_header, print_success,
    print_error, print_warning, print_info
)
from utils.tensor_utils import set_seed, get_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('phenohunter.log')
    ]
)

logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """
    Check that all required dependencies are installed.

    Returns:
        True if all dependencies available, False otherwise
    """
    missing_deps = []
    optional_deps = []

    # Required dependencies
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn'
    }

    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(name)

    # Optional dependencies
    optional = {
        'rdkit': 'RDKit (cheminformatics)',
        'plotly': 'Plotly (visualization)',
        'shap': 'SHAP (explainability)'
    }

    for module, name in optional.items():
        try:
            __import__(module)
        except ImportError:
            optional_deps.append(name)

    # Report
    if missing_deps:
        print_error("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print()
        print_info("Install with: pip install -r requirements.txt")
        return False

    print_success("All required dependencies found")

    if optional_deps:
        print_warning("Missing optional dependencies:")
        for dep in optional_deps:
            print(f"  - {dep}")
        print()
        print_info("Install with: pip install -e .[full]")

    return True


def check_pytorch_version() -> bool:
    """
    Verify PyTorch version is 2.0+.

    Returns:
        True if compatible version
    """
    try:
        import torch
        version = torch.__version__.split('+')[0]  # Remove +cu118 suffix
        major, minor = map(int, version.split('.')[:2])

        if major >= 2:
            print_success(f"PyTorch version {version} detected (compatible)")
            return True
        else:
            print_warning(f"PyTorch version {version} detected (recommend 2.0+)")
            print_info("Update with: pip install --upgrade torch")
            return False
    except Exception as e:
        print_error(f"Could not check PyTorch version: {e}")
        return False


def check_python_version() -> bool:
    """
    Verify Python version is 3.10+.

    Returns:
        True if compatible version
    """
    major = sys.version_info.major
    minor = sys.version_info.minor

    if major == 3 and minor >= 10:
        print_success(f"Python {major}.{minor} detected (compatible)")
        return True
    else:
        print_error(f"Python {major}.{minor} detected (require 3.10+)")
        return False


def system_check() -> bool:
    """
    Run complete system compatibility check.

    Returns:
        True if system is ready
    """
    print_banner()
    print_header("SYSTEM COMPATIBILITY CHECK")

    checks = [
        ("Python Version", check_python_version),
        ("PyTorch Version", check_pytorch_version),
        ("Dependencies", check_dependencies)
    ]

    all_passed = True
    for name, check_func in checks:
        print(f"\n{Colors.CYAN}{name}:{Colors.RESET}")
        if not check_func():
            all_passed = False

    print()
    if all_passed:
        print_success("System ready for PhenoHunter!")
    else:
        print_error("System check failed - please fix issues above")

    return all_passed


def setup_environment(
    seed: int = 369,
    use_gpu: bool = True,
    deterministic: bool = True
) -> None:
    """
    Set up the PhenoHunter environment.

    Args:
        seed: Random seed (369 = sacred geometry)
        use_gpu: Whether to use GPU if available
        deterministic: Whether to use deterministic algorithms
    """
    logger.info("Setting up PhenoHunter environment")

    # Set random seed for reproducibility
    set_seed(seed, deterministic=deterministic)

    # Get device
    device = get_device(prefer_gpu=use_gpu)

    logger.info(f"Environment ready (seed={seed}, device={device})")


def main_cli():
    """
    Main entry point for command-line interface.

    This imports and runs the enhanced CLI.
    """
    # Import CLI after environment setup
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Run the CLI from phenohunt script
    from importlib import import_module
    cli_module = import_module('phenohunt')
    return cli_module.main()


def main():
    """
    Main entry point with environment checks.
    """
    parser = argparse.ArgumentParser(
        description='PhenoHunter Scientific Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--check', action='store_true',
        help='Run system compatibility check'
    )
    parser.add_argument(
        '--version', action='store_true',
        help='Show version information'
    )
    parser.add_argument(
        '--setup', action='store_true',
        help='Run interactive setup'
    )

    # If no arguments, run CLI
    if len(sys.argv) == 1:
        return main_cli()

    args = parser.parse_args()

    if args.version:
        print_banner()
        print(f"{Colors.BOLD}PhenoHunter{Colors.RESET} version {Colors.GREEN}{__version__}{Colors.RESET}")
        print()
        print_info(VERSION_INFO['description'])
        print_info(f"URL: {VERSION_INFO['url']}")
        return 0

    if args.check:
        success = system_check()
        return 0 if success else 1

    if args.setup:
        print_banner()
        print_header("INTERACTIVE SETUP")
        print_info("Running compatibility checks...")

        if not system_check():
            print_error("Please fix compatibility issues first")
            return 1

        print()
        print_success("Setup complete!")
        print_info("You can now run: phenohunt --help")
        return 0

    # Default: run CLI
    return main_cli()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Fatal error: {e}")
        if os.getenv('DEBUG'):
            import traceback
            traceback.print_exc()
        sys.exit(1)
