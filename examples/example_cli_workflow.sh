#!/bin/bash
#
# Example PhenoHunter CLI Workflow
# Demonstrates common breeding tasks using the command-line interface
#

set -e  # Exit on error

echo "=========================================="
echo "PhenoHunter CLI - Example Workflow"
echo "=========================================="
echo ""

# Check if phenohunt is installed
if ! command -v phenohunt &> /dev/null; then
    echo "Installing phenohunt..."
    pip install -e .
fi

# Create sample data if it doesn't exist
if [ ! -f "examples/sample_strains.csv" ]; then
    echo "Creating sample data..."
    python examples/create_sample_data.py
fi

echo "1. Training models on sample database..."
echo "=========================================="
phenohunt train \
    --data examples/sample_strains.csv \
    --epochs 200 \
    --show-warnings \
    --verbose

echo ""
echo "2. Generating F1 hybrid: Blue Dream × OG Kush..."
echo "=========================================="
phenohunt cross \
    --data examples/sample_strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --ratio 0.6 \
    --output examples/f1_hybrid.csv

echo ""
echo "3. Generating F2 population..."
echo "=========================================="
phenohunt f2 \
    --data examples/sample_strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --count 10 \
    --trait "Analgesic" \
    --output examples/f2_population.csv

echo ""
echo "4. Generating backcross (BX1)..."
echo "=========================================="
phenohunt backcross \
    --data examples/sample_strains.csv \
    --parent1 "Blue Dream" \
    --parent2 "OG Kush" \
    --backcross-to "Blue Dream" \
    --generation 1 \
    --output examples/bx1_results.csv

echo ""
echo "5. Analyzing specific strains..."
echo "=========================================="
phenohunt analyze \
    --data examples/sample_strains.csv \
    --strains "Blue Dream" "OG Kush" "Sour Diesel"

echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - examples/f1_hybrid.csv"
echo "  - examples/f2_population.csv"
echo "  - examples/bx1_results.csv"
echo ""
echo "For more information, run: phenohunt --help"
