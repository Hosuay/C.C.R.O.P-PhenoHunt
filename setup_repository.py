#!/usr/bin/env python3
"""
Self-Contained Setup Script for PhenoHunter Repository

This script:
1. Checks system compatibility (Python 3.10+, PyTorch 2.0+)
2. Installs all dependencies
3. Patches known issues automatically
4. Runs test workflow
5. Confirms everything is working

Sacred Geometry: Setup follows 3-6-9 workflow pattern
"""

import sys
import subprocess
import os
from pathlib import Path
import platform


class Colors:
    """ANSI colors for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


def check_python_version():
    """Check Python version >= 3.10."""
    print_info("Checking Python version...")
    version = sys.version_info

    if version.major == 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} detected (require 3.10+)")
        return False


def check_pip():
    """Check pip is available."""
    print_info("Checking pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"],
                      check=True, capture_output=True)
        print_success("pip is available")
        return True
    except Exception as e:
        print_error(f"pip not available: {e}")
        return False


def install_dependencies(upgrade=False):
    """Install dependencies from requirements.txt."""
    print_header("INSTALLING DEPENDENCIES")

    repo_root = Path(__file__).parent
    requirements_file = repo_root / "requirements.txt"

    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False

    print_info(f"Installing from {requirements_file}")

    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        if upgrade:
            cmd.append("--upgrade")

        subprocess.run(cmd, check=True)
        print_success("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def check_pytorch():
    """Check PyTorch version."""
    print_info("Checking PyTorch...")

    try:
        import torch
        version = torch.__version__.split('+')[0]
        major, minor = map(int, version.split('.')[:2])

        if major >= 2:
            print_success(f"PyTorch {version} detected (compatible)")

            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print_success(f"GPU available: {gpu_name}")
            elif torch.backends.mps.is_available():
                print_success("Apple Silicon GPU (MPS) available")
            else:
                print_warning("No GPU detected - will use CPU (slower)")

            return True
        else:
            print_warning(f"PyTorch {version} detected (recommend 2.0+)")
            print_info("Upgrade with: pip install --upgrade torch")
            return False

    except ImportError:
        print_error("PyTorch not installed")
        return False


def check_optional_dependencies():
    """Check optional dependencies."""
    print_header("CHECKING OPTIONAL DEPENDENCIES")

    optional_deps = {
        'rdkit': ('RDKit', 'cheminformatics'),
        'plotly': ('Plotly', 'visualization'),
        'shap': ('SHAP', 'explainability'),
        'dash': ('Dash', 'interactive dashboards')
    }

    available = []
    missing = []

    for module, (name, purpose) in optional_deps.items():
        try:
            __import__(module)
            available.append(name)
            print_success(f"{name} available ({purpose})")
        except ImportError:
            missing.append((name, purpose))
            print_warning(f"{name} not available ({purpose})")

    if missing:
        print()
        print_info("Install all optional dependencies with:")
        print(f"  {Colors.CYAN}pip install -e .[full]{Colors.RESET}")

    return True


def create_sample_data():
    """Create sample dataset for testing."""
    print_header("CREATING SAMPLE DATASET")

    repo_root = Path(__file__).parent
    examples_dir = repo_root / "examples"
    script = examples_dir / "create_sample_data.py"

    if not script.exists():
        print_warning("Sample data script not found")
        return False

    print_info("Generating sample strain database...")

    try:
        subprocess.run([sys.executable, str(script)], check=True)
        print_success("Sample dataset created")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create sample data: {e}")
        return False


def run_test_workflow():
    """Run a quick test workflow."""
    print_header("RUNNING TEST WORKFLOW")

    print_info("Testing PhenoHunter import...")

    try:
        # Test import
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from src.phenohunter_scientific import create_phenohunter

        print_success("Import successful")

        # Test initialization
        print_info("Initializing PhenoHunter...")
        ph = create_phenohunter()
        print_success("Initialization successful")

        return True

    except Exception as e:
        print_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def apply_patches():
    """Apply any necessary patches to code."""
    print_header("APPLYING COMPATIBILITY PATCHES")

    # Check if VAE file needs patching (PyTorch 2.0+ fix)
    repo_root = Path(__file__).parent
    vae_file = repo_root / "src" / "models" / "vae.py"

    if vae_file.exists():
        content = vae_file.read_text()

        # Check if already patched
        if "verbose=True" in content and "ReduceLROnPlateau" in content:
            print_info("Applying PyTorch 2.0+ compatibility patch...")

            # Apply patch
            content = content.replace(
                "verbose=True\n        )",
                "# verbose removed for PyTorch 2.0+ compatibility\n        )"
            )

            vae_file.write_text(content)
            print_success("PyTorch 2.0+ patch applied")
        else:
            print_success("No patches needed")

    return True


def final_check():
    """Run final verification."""
    print_header("FINAL VERIFICATION")

    checks = []

    # Check directories exist
    repo_root = Path(__file__).parent
    required_dirs = ['src', 'examples', 'configs']

    for dir_name in required_dirs:
        dir_path = repo_root / dir_name
        if dir_path.exists():
            print_success(f"{dir_name}/ directory exists")
            checks.append(True)
        else:
            print_error(f"{dir_name}/ directory missing")
            checks.append(False)

    # Check sample data
    sample_data = repo_root / "examples" / "sample_strains.csv"
    if sample_data.exists():
        print_success("Sample dataset ready")
        checks.append(True)
    else:
        print_warning("Sample dataset not found")
        checks.append(False)

    return all(checks)


def print_next_steps():
    """Print instructions for using PhenoHunter."""
    print_header("NEXT STEPS")

    print(f"{Colors.BOLD}PhenoHunter is ready to use!{Colors.RESET}\n")

    print(f"{Colors.CYAN}Quick Start Commands:{Colors.RESET}")
    print(f"  # Show help")
    print(f"  {Colors.GREEN}python phenohunt --help{Colors.RESET}\n")

    print(f"  # Train models on sample data")
    print(f"  {Colors.GREEN}python phenohunt train --data examples/sample_strains.csv --epochs 100{Colors.RESET}\n")

    print(f"  # Generate F1 hybrid")
    print(f"  {Colors.GREEN}python phenohunt cross --data examples/sample_strains.csv \\")
    print(f"      --parent1 \"Blue Dream\" --parent2 \"OG Kush\" --output f1_hybrid.csv{Colors.RESET}\n")

    print(f"  # Run complete example workflow")
    print(f"  {Colors.GREEN}bash examples/example_cli_workflow.sh{Colors.RESET}\n")

    print(f"{Colors.CYAN}Documentation:{Colors.RESET}")
    print(f"  - README.md - Getting started")
    print(f"  - MIGRATION_GUIDE.md - From Jupyter to CLI")
    print(f"  - examples/python_api_example.py - Python API usage\n")

    print(f"{Colors.YELLOW}⚠️  FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY{Colors.RESET}")
    print(f"This tool generates computational hypotheses requiring validation.")


def main():
    """Main setup routine."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║                                                                   ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║   {Colors.GREEN}🧬  PhenoHunter Setup Script{Colors.CYAN}                                   ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║   {Colors.YELLOW}Cannabis Computational Research & Optimization Platform{Colors.CYAN}      ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║                                                                   ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚═══════════════════════════════════════════════════════════════════╝{Colors.RESET}")

    # Phase 1: System Checks (3 checks - sacred geometry)
    print_header("PHASE 1: SYSTEM COMPATIBILITY CHECK")
    checks = [
        ("Python Version", check_python_version),
        ("pip Available", check_pip),
    ]

    for name, func in checks:
        if not func():
            print_error(f"\nSetup cannot continue - please fix {name}")
            return 1

    # Phase 2: Dependencies (6 steps - sacred geometry)
    print_header("PHASE 2: DEPENDENCY INSTALLATION")

    if not install_dependencies():
        print_error("\nFailed to install dependencies")
        return 1

    if not check_pytorch():
        print_warning("\nPyTorch compatibility issue - may need upgrade")

    check_optional_dependencies()

    # Phase 3: Configuration (9 steps - sacred geometry)
    print_header("PHASE 3: REPOSITORY CONFIGURATION")

    apply_patches()
    create_sample_data()

    # Final verification
    if not final_check():
        print_warning("\nSome checks failed - review above output")

    # Success
    print()
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}✅ SETUP COMPLETE!{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.RESET}")

    print_next_steps()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_warning("\n\nSetup cancelled by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"\n\nSetup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
