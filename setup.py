"""
Setup script for PhenoHunter CLI installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="phenohunt",
    version="3.0.0",
    author="Hosuay and Contributors",
    author_email="",
    description="Cannabis Computational Research & Optimization Platform - Professional CLI Edition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hosuay/C.C.R.O.P-PhenoHunt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "scikit-learn>=1.0.0",
        "plotly>=5.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
        "requests>=2.26.0",
        "beautifulsoup4>=4.9.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "full": [
            "rdkit>=2022.9.1",
            "py3Dmol>=2.0.0",
            "shap>=0.41.0",
            "pdfplumber>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "phenohunt=phenohunt_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "configs/*.json"],
    },
    zip_safe=False,
)
