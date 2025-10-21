"""
PhenoHunter Scientific Edition - Usage Example
Demonstrates the improved scientific features
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from phenohunter_scientific import create_phenohunter


def create_sample_database():
    """Create a sample strain database for demonstration."""
    strains = {
        'Blue Dream': {
            'type': 'hybrid',
            'thc_pct': 19.5, 'cbd_pct': 0.8, 'cbg_pct': 0.9, 'cbc_pct': 0.2, 'cbda_pct': 0.15,
            'thcv_pct': 0.3, 'cbn_pct': 0.5, 'delta8_thc_pct': 0.1, 'thca_pct': 18.0,
            'myrcene_pct': 1.2, 'limonene_pct': 1.5, 'pinene_pct': 0.8, 'linalool_pct': 0.4,
            'caryophyllene_pct': 0.9, 'humulene_pct': 0.3, 'terpinolene_pct': 0.2,
            'ocimene_pct': 0.1, 'camphene_pct': 0.05, 'bisabolol_pct': 0.15
        },
        'OG Kush': {
            'type': 'hybrid',
            'thc_pct': 22.5, 'cbd_pct': 0.3, 'cbg_pct': 0.6, 'cbc_pct': 0.25, 'cbda_pct': 0.1,
            'thcv_pct': 0.4, 'cbn_pct': 0.8, 'delta8_thc_pct': 0.15, 'thca_pct': 21.0,
            'myrcene_pct': 1.8, 'limonene_pct': 1.2, 'pinene_pct': 0.7, 'linalool_pct': 0.6,
            'caryophyllene_pct': 1.4, 'humulene_pct': 0.5, 'terpinolene_pct': 0.15,
            'ocimene_pct': 0.08, 'camphene_pct': 0.04, 'bisabolol_pct': 0.12
        },
        'Sour Diesel': {
            'type': 'sativa',
            'thc_pct': 20.8, 'cbd_pct': 0.2, 'cbg_pct': 0.7, 'cbc_pct': 0.18, 'cbda_pct': 0.12,
            'thcv_pct': 0.5, 'cbn_pct': 0.3, 'delta8_thc_pct': 0.12, 'thca_pct': 19.5,
            'myrcene_pct': 0.8, 'limonene_pct': 2.1, 'pinene_pct': 1.3, 'linalool_pct': 0.3,
            'caryophyllene_pct': 0.9, 'humulene_pct': 0.4, 'terpinolene_pct': 0.25,
            'ocimene_pct': 0.12, 'camphene_pct': 0.06, 'bisabolol_pct': 0.08
        },
        'Girl Scout Cookies': {
            'type': 'hybrid',
            'thc_pct': 24.2, 'cbd_pct': 0.5, 'cbg_pct': 0.8, 'cbc_pct': 0.3, 'cbda_pct': 0.2,
            'thcv_pct': 0.6, 'cbn_pct': 0.6, 'delta8_thc_pct': 0.18, 'thca_pct': 23.0,
            'myrcene_pct': 1.1, 'limonene_pct': 1.4, 'pinene_pct': 0.6, 'linalool_pct': 0.7,
            'caryophyllene_pct': 1.6, 'humulene_pct': 0.45, 'terpinolene_pct': 0.18,
            'ocimene_pct': 0.09, 'camphene_pct': 0.03, 'bisabolol_pct': 0.14
        },
        'Granddaddy Purple': {
            'type': 'indica',
            'thc_pct': 21.5, 'cbd_pct': 0.9, 'cbg_pct': 0.5, 'cbc_pct': 0.22, 'cbda_pct': 0.18,
            'thcv_pct': 0.2, 'cbn_pct': 1.2, 'delta8_thc_pct': 0.08, 'thca_pct': 20.0,
            'myrcene_pct': 2.4, 'limonene_pct': 0.7, 'pinene_pct': 0.5, 'linalool_pct': 1.0,
            'caryophyllene_pct': 1.5, 'humulene_pct': 0.6, 'terpinolene_pct': 0.12,
            'ocimene_pct': 0.06, 'camphene_pct': 0.02, 'bisabolol_pct': 0.18
        },
        'Wedding Cake': {
            'type': 'indica',
            'thc_pct': 25.3, 'cbd_pct': 0.4, 'cbg_pct': 0.9, 'cbc_pct': 0.28, 'cbda_pct': 0.16,
            'thcv_pct': 0.7, 'cbn_pct': 0.9, 'delta8_thc_pct': 0.2, 'thca_pct': 24.0,
            'myrcene_pct': 1.3, 'limonene_pct': 1.8, 'pinene_pct': 0.7, 'linalool_pct': 0.8,
            'caryophyllene_pct': 1.7, 'humulene_pct': 0.55, 'terpinolene_pct': 0.16,
            'ocimene_pct': 0.1, 'camphene_pct': 0.04, 'bisabolol_pct': 0.16
        },
        'Gelato': {
            'type': 'hybrid',
            'thc_pct': 23.8, 'cbd_pct': 0.6, 'cbg_pct': 0.85, 'cbc_pct': 0.26, 'cbda_pct': 0.19,
            'thcv_pct': 0.5, 'cbn_pct': 0.7, 'delta8_thc_pct': 0.16, 'thca_pct': 22.5,
            'myrcene_pct': 1.4, 'limonene_pct': 1.6, 'pinene_pct': 0.8, 'linalool_pct': 0.65,
            'caryophyllene_pct': 1.5, 'humulene_pct': 0.5, 'terpinolene_pct': 0.14,
            'ocimene_pct': 0.09, 'camphene_pct': 0.04, 'bisabolol_pct': 0.13
        },
        'Northern Lights': {
            'type': 'indica',
            'thc_pct': 18.9, 'cbd_pct': 1.2, 'cbg_pct': 0.6, 'cbc_pct': 0.2, 'cbda_pct': 0.14,
            'thcv_pct': 0.2, 'cbn_pct': 1.0, 'delta8_thc_pct': 0.09, 'thca_pct': 17.5,
            'myrcene_pct': 2.1, 'limonene_pct': 0.8, 'pinene_pct': 0.9, 'linalool_pct': 0.9,
            'caryophyllene_pct': 1.3, 'humulene_pct': 0.55, 'terpinolene_pct': 0.11,
            'ocimene_pct': 0.07, 'camphene_pct': 0.03, 'bisabolol_pct': 0.17
        },
        'Jack Herer': {
            'type': 'sativa',
            'thc_pct': 20.3, 'cbd_pct': 0.4, 'cbg_pct': 0.75, 'cbc_pct': 0.21, 'cbda_pct': 0.13,
            'thcv_pct': 0.6, 'cbn_pct': 0.4, 'delta8_thc_pct': 0.14, 'thca_pct': 19.0,
            'myrcene_pct': 0.9, 'limonene_pct': 1.9, 'pinene_pct': 1.6, 'linalool_pct': 0.4,
            'caryophyllene_pct': 1.0, 'humulene_pct': 0.45, 'terpinolene_pct': 0.22,
            'ocimene_pct': 0.11, 'camphene_pct': 0.07, 'bisabolol_pct': 0.09
        },
        'Durban Poison': {
            'type': 'sativa',
            'thc_pct': 18.5, 'cbd_pct': 0.2, 'cbg_pct': 0.9, 'cbc_pct': 0.17, 'cbda_pct': 0.1,
            'thcv_pct': 0.8, 'cbn_pct': 0.2, 'delta8_thc_pct': 0.11, 'thca_pct': 17.0,
            'myrcene_pct': 0.7, 'limonene_pct': 2.0, 'pinene_pct': 1.5, 'linalool_pct': 0.3,
            'caryophyllene_pct': 0.85, 'humulene_pct': 0.35, 'terpinolene_pct': 0.28,
            'ocimene_pct': 0.13, 'camphene_pct': 0.08, 'bisabolol_pct': 0.07
        }
    }

    # Convert to DataFrame
    rows = []
    for name, profile in strains.items():
        row = {'strain_name': name}
        row.update(profile)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    """Main demonstration function."""

    print("=" * 70)
    print("PHENOHUNTER SCIENTIFIC EDITION - DEMONSTRATION")
    print("=" * 70)

    # Step 1: Create PhenoHunter instance
    print("\n[1/7] Initializing PhenoHunter Scientific...")
    ph = create_phenohunter()
    print("✓ Initialized")

    # Step 2: Load strain database
    print("\n[2/7] Loading strain database...")
    strain_data = create_sample_database()
    warnings = ph.load_strain_database(strain_data, validate=True)

    print(f"✓ Loaded {len(strain_data)} strains")
    if warnings:
        print("\nValidation Warnings:")
        for category, warning_list in warnings.items():
            if warning_list:
                print(f"  [{category}]")
                for w in warning_list[:3]:  # Show first 3
                    print(f"    - {w}")

    # Step 3: Train VAE
    print("\n[3/7] Training Variational Autoencoder...")
    history = ph.train_vae(epochs=200, verbose=False)
    final_loss = history['train_loss'][-1]
    print(f"✓ VAE trained (Final loss: {final_loss:.4f})")

    # Step 4: Train effect predictors
    print("\n[4/7] Training therapeutic effect predictors...")
    metrics = ph.train_effect_predictors(auto_generate_targets=True)
    print(f"✓ Trained {len(metrics)} effect predictors")

    # Step 5: Generate F1 hybrid
    print("\n[5/7] Generating F1 hybrid: Blue Dream × OG Kush...")
    f1_result = ph.generate_f1_hybrid(
        parent1_name='Blue Dream',
        parent2_name='OG Kush',
        parent1_weight=0.6,
        n_samples=100
    )

    print(f"✓ F1 Generated:")
    print(f"  - Stability Score: {f1_result.stability_score:.3f}")
    print(f"  - Heterosis Score: {f1_result.heterosis_score:.3f}")
    print(f"  - Parent 1 Contribution: {f1_result.parent1_weight:.1%}")

    # Show top predicted effects
    if f1_result.predicted_effects:
        print(f"\n  Top Predicted Effects:")
        sorted_effects = sorted(
            f1_result.predicted_effects.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for effect, prob in sorted_effects[:3]:
            print(f"    • {effect}: {prob:.1%}")

    # Step 6: Generate F2 population
    print("\n[6/7] Generating F2 population (5 individuals)...")
    f2_population = ph.generate_f2_population(f1_result, n_offspring=5)
    print(f"✓ F2 Population Generated")
    print(f"  - Population size: {len(f2_population)}")
    print(f"  - Avg stability: {np.mean([r.stability_score for r in f2_population]):.3f}")

    # Step 7: Backcross
    print("\n[7/7] Backcrossing F1 to Blue Dream (BX1)...")
    bx1_result = ph.backcross(
        f1_result,
        parent_name='Blue Dream',
        backcross_generation=1
    )
    print(f"✓ BX1 Generated")
    print(f"  - Stability Score: {bx1_result.stability_score:.3f}")
    print(f"  - Blue Dream Contribution: {bx1_result.parent1_weight:.1%}")

    # Export results
    print("\n" + "=" * 70)
    print("Exporting Results...")
    all_results = [f1_result] + f2_population + [bx1_result]
    ph.export_results(all_results, 'scientific_breeding_results.csv')
    print("✓ Results exported to: scientific_breeding_results.csv")

    # Summary report
    print("\n" + "=" * 70)
    print(ph.get_summary_report())

    # Demonstrate visualization (optional - requires display)
    try:
        print("\nGenerating visualizations...")
        ph.visualize_breeding_result(f1_result, show_uncertainty=True)
        print("✓ Visualizations generated")
    except Exception as e:
        print(f"⚠ Visualization skipped (requires display): {e}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Improvements Demonstrated:")
    print("  ✓ Variational Autoencoder with uncertainty quantification")
    print("  ✓ Research-backed therapeutic effect prediction")
    print("  ✓ Comprehensive data validation")
    print("  ✓ Multi-generation breeding (F1, F2, Backcross)")
    print("  ✓ Expanded chemical profile (20 compounds)")
    print("  ✓ Scientific configuration system")
    print("  ✓ Statistical rigor and reproducibility")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
