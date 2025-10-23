#!/usr/bin/env python3
"""
PhenoHunter CLI - Command-line interface for cannabis breeding optimization.

This replaces the Jupyter notebook interface with a proper CLI application.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from phenohunter_scientific import create_phenohunter
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_train(args):
    """Train models on a strain database."""
    logger.info(f"Loading strain database from {args.data}")

    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    ph = create_phenohunter(args.config)

    logger.info(f"Loaded {len(df)} strains")
    warnings = ph.load_strain_database(df, validate=not args.no_validate)

    if args.show_warnings and warnings:
        logger.warning("Validation warnings detected:")
        for category, warning_list in warnings.items():
            for warning in warning_list[:5]:
                logger.warning(f"[{category}] {warning}")

    # Train VAE
    logger.info(f"Training VAE for {args.epochs} epochs...")
    history = ph.train_vae(epochs=args.epochs, verbose=args.verbose)
    logger.info(f"VAE training complete. Final loss: {history['train_loss'][-1]:.4f}")

    # Train effect predictors
    if not args.skip_effects:
        logger.info("Training effect predictors...")
        metrics = ph.train_effect_predictors(auto_generate_targets=True)
        logger.info(f"Trained {len(metrics)} effect predictors")

    # Save model if requested
    if args.save:
        import torch
        torch.save(ph.vae.state_dict(), args.save)
        logger.info(f"Model saved to {args.save}")

    # Print summary
    print("\n" + ph.get_summary_report())

    return 0


def cmd_cross(args):
    """Generate a hybrid cross between two parents."""
    logger.info(f"Loading strain database from {args.data}")

    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    ph = create_phenohunter(args.config)
    ph.load_strain_database(df, validate=False)

    # Train models
    logger.info("Training models...")
    ph.train_vae(epochs=args.epochs, verbose=False)
    ph.train_effect_predictors(auto_generate_targets=True)

    # Generate F1 hybrid
    logger.info(f"Generating F1 hybrid: {args.parent1} × {args.parent2}")
    try:
        f1_result = ph.generate_f1_hybrid(
            parent1_name=args.parent1,
            parent2_name=args.parent2,
            parent1_weight=args.ratio,
            n_samples=args.samples
        )
    except Exception as e:
        logger.error(f"Failed to generate hybrid: {e}")
        return 1

    # Print results
    print(f"\n{'='*70}")
    print(f"F1 HYBRID RESULTS")
    print(f"{'='*70}")
    print(f"Parents: {args.parent1} ({args.ratio:.0%}) × {args.parent2} ({1-args.ratio:.0%})")
    print(f"Stability Score: {f1_result.stability_score:.3f}")
    print(f"Heterosis Score: {f1_result.heterosis_score:.3f}")

    print(f"\nChemical Profile:")
    feature_names = ph.feature_columns
    for i, feat in enumerate(feature_names):
        mean = f1_result.offspring_profile[i]
        std = f1_result.offspring_std[i]
        print(f"  {feat:<20}: {mean:6.2f} ± {std:4.2f}")

    if f1_result.predicted_effects:
        print(f"\nPredicted Effects:")
        sorted_effects = sorted(
            f1_result.predicted_effects.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for effect, prob in sorted_effects:
            print(f"  {effect:<20}: {prob:6.1%}")

    # Save if requested
    if args.output:
        ph.export_results([f1_result], args.output)
        logger.info(f"Results saved to {args.output}")

    # Generate visualization
    if args.visualize:
        logger.info("Generating visualization...")
        ph.visualize_breeding_result(f1_result, show_uncertainty=True)

    return 0


def cmd_f2(args):
    """Generate an F2 population from parents."""
    logger.info(f"Loading strain database from {args.data}")

    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    ph = create_phenohunter(args.config)
    ph.load_strain_database(df, validate=False)

    # Train models
    logger.info("Training models...")
    ph.train_vae(epochs=args.epochs, verbose=False)
    ph.train_effect_predictors(auto_generate_targets=True)

    # Generate F1 first
    logger.info(f"Generating F1: {args.parent1} × {args.parent2}")
    f1_result = ph.generate_f1_hybrid(
        parent1_name=args.parent1,
        parent2_name=args.parent2,
        parent1_weight=args.ratio,
        n_samples=100
    )

    # Generate F2 population
    logger.info(f"Generating F2 population with {args.count} offspring")
    f2_population = ph.generate_f2_population(f1_result, n_offspring=args.count)

    # Print results
    print(f"\n{'='*70}")
    print(f"F2 POPULATION RESULTS")
    print(f"{'='*70}")
    print(f"Parents: {args.parent1} × {args.parent2}")
    print(f"F2 Population Size: {len(f2_population)}")
    print(f"Avg Stability: {np.mean([r.stability_score for r in f2_population]):.3f}")

    # Sort by target trait if specified
    if args.trait:
        logger.info(f"Ranking by {args.trait} effect...")
        sorted_pop = sorted(
            f2_population,
            key=lambda x: x.predicted_effects.get(args.trait, 0),
            reverse=True
        )

        print(f"\nTop 5 candidates for {args.trait}:")
        print("-"*70)
        for i, result in enumerate(sorted_pop[:5], 1):
            score = result.predicted_effects.get(args.trait, 0)
            print(f"{i}. Stability: {result.stability_score:.3f}, {args.trait}: {score:.1%}")

    # Save if requested
    if args.output:
        ph.export_results(f2_population, args.output)
        logger.info(f"Results saved to {args.output}")

    return 0


def cmd_backcross(args):
    """Generate backcross offspring."""
    logger.info(f"Loading strain database from {args.data}")

    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    ph = create_phenohunter(args.config)
    ph.load_strain_database(df, validate=False)

    # Train models
    logger.info("Training models...")
    ph.train_vae(epochs=args.epochs, verbose=False)
    ph.train_effect_predictors(auto_generate_targets=True)

    # Generate F1 first
    logger.info(f"Generating F1: {args.parent1} × {args.parent2}")
    f1_result = ph.generate_f1_hybrid(
        parent1_name=args.parent1,
        parent2_name=args.parent2,
        parent1_weight=0.5,
        n_samples=100
    )

    # Generate backcross
    logger.info(f"Backcrossing to {args.backcross_to} (BX{args.generation})")
    bx_result = ph.backcross(
        f1_result,
        parent_name=args.backcross_to,
        backcross_generation=args.generation
    )

    # Print results
    print(f"\n{'='*70}")
    print(f"BACKCROSS RESULTS (BX{args.generation})")
    print(f"{'='*70}")
    print(f"F1: {args.parent1} × {args.parent2}")
    print(f"Backcrossed to: {args.backcross_to}")
    print(f"Stability Score: {bx_result.stability_score:.3f}")
    print(f"Parent Contribution: {bx_result.parent1_weight:.1%}")

    # Save if requested
    if args.output:
        ph.export_results([bx_result], args.output)
        logger.info(f"Results saved to {args.output}")

    return 0


def cmd_analyze(args):
    """Analyze a strain or compare multiple strains."""
    logger.info(f"Loading strain database from {args.data}")

    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    ph = create_phenohunter(args.config)
    ph.load_strain_database(df, validate=False)

    # Train effect predictors
    logger.info("Training effect predictors...")
    ph.train_effect_predictors(auto_generate_targets=True)

    for strain_name in args.strains:
        if strain_name not in df['strain_name'].values:
            logger.warning(f"Strain '{strain_name}' not found in database")
            continue

        strain_data = df[df['strain_name'] == strain_name].iloc[0]

        print(f"\n{'='*70}")
        print(f"STRAIN ANALYSIS: {strain_name}")
        print(f"{'='*70}")
        print(f"Type: {strain_data.get('type', 'Unknown')}")

        print(f"\nChemical Profile:")
        for feat in ph.feature_columns:
            if feat in strain_data:
                print(f"  {feat:<20}: {strain_data[feat]:6.2f}%")

        # Predict effects
        profile = {feat: strain_data[feat] for feat in ph.feature_columns if feat in strain_data}

        # This would require implementing a predict_effects method
        # For now, just show the profile

    return 0


def cmd_import(args):
    """Import strain data from various formats."""
    logger.info(f"Importing from {args.file}")

    try:
        if args.file.endswith('.csv'):
            df = pd.read_csv(args.file)
        elif args.file.endswith('.json'):
            df = pd.read_json(args.file)
        else:
            logger.error(f"Unsupported file format: {args.file}")
            return 1

        logger.info(f"Loaded {len(df)} strains")
        print(f"\nColumns: {', '.join(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())

        # Save to output if specified
        if args.output:
            df.to_csv(args.output, index=False)
            logger.info(f"Saved to {args.output}")

    except Exception as e:
        logger.error(f"Failed to import: {e}")
        return 1

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PhenoHunter - Cannabis Breeding Optimization CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models on your data
  phenohunt train --data strains.csv --epochs 369

  # Generate F1 hybrid
  phenohunt cross --data strains.csv --parent1 "Blue Dream" --parent2 "OG Kush" --output f1.csv

  # Generate F2 population
  phenohunt f2 --data strains.csv --parent1 "Blue Dream" --parent2 "OG Kush" --count 10 --trait Analgesic

  # Backcross
  phenohunt backcross --data strains.csv --parent1 "Blue Dream" --parent2 "OG Kush" --backcross-to "Blue Dream"

  # Analyze strains
  phenohunt analyze --data strains.csv --strains "Blue Dream" "OG Kush"

For more information, visit: https://github.com/Hosuay/C.C.R.O.P-PhenoHunt
        """
    )

    parser.add_argument('--version', action='version', version='PhenoHunter 3.0.0')
    parser.add_argument('--config', type=str, help='Path to config file')

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train models on strain database')
    train_parser.add_argument('--data', required=True, help='Path to strain CSV file')
    train_parser.add_argument('--epochs', type=int, default=369, help='Training epochs (default: 369)')
    train_parser.add_argument('--no-validate', action='store_true', help='Skip data validation')
    train_parser.add_argument('--skip-effects', action='store_true', help='Skip effect predictor training')
    train_parser.add_argument('--save', type=str, help='Save trained model to file')
    train_parser.add_argument('--show-warnings', action='store_true', help='Show validation warnings')
    train_parser.add_argument('--verbose', action='store_true', help='Verbose training output')

    # Cross command
    cross_parser = subparsers.add_parser('cross', help='Generate F1 hybrid cross')
    cross_parser.add_argument('--data', required=True, help='Path to strain CSV file')
    cross_parser.add_argument('--parent1', required=True, help='First parent strain name')
    cross_parser.add_argument('--parent2', required=True, help='Second parent strain name')
    cross_parser.add_argument('--ratio', type=float, default=0.6, help='Parent1 contribution ratio (default: 0.6)')
    cross_parser.add_argument('--samples', type=int, default=100, help='Monte Carlo samples (default: 100)')
    cross_parser.add_argument('--epochs', type=int, default=200, help='Training epochs (default: 200)')
    cross_parser.add_argument('--output', type=str, help='Output CSV file')
    cross_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')

    # F2 command
    f2_parser = subparsers.add_parser('f2', help='Generate F2 population')
    f2_parser.add_argument('--data', required=True, help='Path to strain CSV file')
    f2_parser.add_argument('--parent1', required=True, help='First parent strain name')
    f2_parser.add_argument('--parent2', required=True, help='Second parent strain name')
    f2_parser.add_argument('--ratio', type=float, default=0.5, help='Parent1 contribution ratio (default: 0.5)')
    f2_parser.add_argument('--count', type=int, default=10, help='Number of F2 offspring (default: 10)')
    f2_parser.add_argument('--trait', type=str, help='Trait to optimize for ranking')
    f2_parser.add_argument('--epochs', type=int, default=200, help='Training epochs (default: 200)')
    f2_parser.add_argument('--output', type=str, help='Output CSV file')

    # Backcross command
    bx_parser = subparsers.add_parser('backcross', help='Generate backcross offspring')
    bx_parser.add_argument('--data', required=True, help='Path to strain CSV file')
    bx_parser.add_argument('--parent1', required=True, help='First parent strain name')
    bx_parser.add_argument('--parent2', required=True, help='Second parent strain name')
    bx_parser.add_argument('--backcross-to', required=True, help='Parent to backcross to')
    bx_parser.add_argument('--generation', type=int, default=1, help='Backcross generation number (default: 1)')
    bx_parser.add_argument('--epochs', type=int, default=200, help='Training epochs (default: 200)')
    bx_parser.add_argument('--output', type=str, help='Output CSV file')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze strains')
    analyze_parser.add_argument('--data', required=True, help='Path to strain CSV file')
    analyze_parser.add_argument('--strains', nargs='+', required=True, help='Strain names to analyze')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import strain data')
    import_parser.add_argument('--file', required=True, help='File to import (CSV or JSON)')
    import_parser.add_argument('--output', type=str, help='Output CSV file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Route to appropriate command
    commands = {
        'train': cmd_train,
        'cross': cmd_cross,
        'f2': cmd_f2,
        'backcross': cmd_backcross,
        'analyze': cmd_analyze,
        'import': cmd_import
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
