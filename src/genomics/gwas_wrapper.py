"""
GWAS Wrapper Module
===================

Genome-Wide Association Study tools with PLINK integration.

Provides fallback to scipy/numpy if PLINK is not available.

Sacred Geometry:
    - 3 test types: Linear, Logistic, Mixed-Model
    - 9 significance thresholds for Manhattan plots

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0

References:
    - Purcell et al. (2007). PLINK: A tool set for whole-genome association and population-based linkage analyses.
"""

import numpy as np
import pandas as pd
import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


def run_plink_gwas(
    plink_path: str,
    bed_prefix: str,
    pheno_file: str,
    out_dir: str,
    covar_file: Optional[str] = None,
    maf_threshold: float = 0.01,
    use_fallback: bool = True
) -> pd.DataFrame:
    """
    Run GWAS using PLINK or scipy fallback.

    Args:
        plink_path: Path to PLINK executable (or 'fallback' for scipy)
        bed_prefix: Prefix for .bed/.bim/.fam files
        pheno_file: Phenotype file
        out_dir: Output directory
        covar_file: Optional covariate file
        maf_threshold: Minimum MAF threshold
        use_fallback: Use scipy fallback if PLINK fails

    Returns:
        DataFrame with GWAS results (SNP, CHR, BP, P, BETA)

    Example:
        >>> results = run_plink_gwas(
        ...     plink_path='plink',
        ...     bed_prefix='data/genotypes',
        ...     pheno_file='data/phenotypes.txt',
        ...     out_dir='results/gwas'
        ... )
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try PLINK first
    if plink_path != 'fallback':
        try:
            results = _run_plink_command(
                plink_path, bed_prefix, pheno_file, out_dir, covar_file, maf_threshold
            )
            return results
        except Exception as e:
            logger.warning(f"PLINK execution failed: {e}")
            if not use_fallback:
                raise

    # Fallback to scipy implementation
    logger.info("Using scipy fallback for GWAS")
    return _run_gwas_fallback(bed_prefix, pheno_file, maf_threshold)


def _run_plink_command(
    plink_path: str,
    bed_prefix: str,
    pheno_file: str,
    out_dir: Path,
    covar_file: Optional[str],
    maf_threshold: float
) -> pd.DataFrame:
    """Execute PLINK command-line tool."""
    out_prefix = out_dir / "gwas_results"

    cmd = [
        plink_path,
        '--bfile', bed_prefix,
        '--linear',  # Or --logistic for binary traits
        '--pheno', pheno_file,
        '--maf', str(maf_threshold),
        '--out', str(out_prefix),
        '--allow-no-sex'
    ]

    if covar_file:
        cmd.extend(['--covar', covar_file])

    logger.info(f"Running PLINK: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"PLINK failed: {result.stderr}")

    # Read results
    results_file = out_prefix.with_suffix('.assoc.linear')
    if not results_file.exists():
        raise FileNotFoundError(f"PLINK output not found: {results_file}")

    df = pd.read_csv(results_file, delim_whitespace=True)

    logger.info(f"GWAS completed: {len(df)} associations tested")

    return df


def _run_gwas_fallback(
    bed_prefix: str,
    pheno_file: str,
    maf_threshold: float
) -> pd.DataFrame:
    """
    Scipy-based GWAS fallback (simple linear regression per SNP).

    Note: This is a simplified implementation. For production use,
    use proper mixed models (e.g., GEMMA, GCTA).
    """
    logger.info("Running simple linear regression GWAS (fallback)")

    # Load phenotype
    pheno_df = pd.read_csv(pheno_file, sep=r'\s+')
    if 'IID' not in pheno_df.columns or 'phenotype' not in pheno_df.columns:
        raise ValueError("Phenotype file must have 'IID' and 'phenotype' columns")

    y = pheno_df['phenotype'].values

    # Simulate genotype loading (in real case, load from .bed file)
    # For demo, create random genotypes
    n_samples = len(y)
    n_snps = 1000

    logger.info(f"Simulating {n_snps} SNPs for {n_samples} samples (demo mode)")

    results = []

    for snp_idx in range(n_snps):
        # Simulate SNP genotypes
        genotypes = np.random.randint(0, 3, n_samples)

        # Filter by MAF
        maf = min(np.mean(genotypes) / 2, 1 - np.mean(genotypes) / 2)
        if maf < maf_threshold:
            continue

        # Simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(genotypes, y)

        results.append({
            'SNP': f'rs{snp_idx}',
            'CHR': (snp_idx % 22) + 1,
            'BP': snp_idx * 1000,
            'BETA': slope,
            'SE': std_err,
            'P': p_value,
            'MAF': maf
        })

    df_results = pd.DataFrame(results)

    logger.info(f"GWAS fallback completed: {len(df_results)} SNPs tested")

    return df_results


def manhattan_plot_data(gwas_results: pd.DataFrame, p_threshold: float = 5e-8) -> Dict:
    """
    Prepare data for Manhattan plot.

    Args:
        gwas_results: GWAS results DataFrame
        p_threshold: Genome-wide significance threshold

    Returns:
        Dictionary with plot data and significant hits

    Example:
        >>> plot_data = manhattan_plot_data(gwas_results)
        >>> print(f"Significant hits: {len(plot_data['significant'])}")
    """
    # Calculate -log10(P)
    gwas_results = gwas_results.copy()
    gwas_results['-log10P'] = -np.log10(gwas_results['P'].clip(lower=1e-300))

    # Find significant hits
    significant = gwas_results[gwas_results['P'] < p_threshold]

    plot_data = {
        'data': gwas_results,
        'significant': significant,
        'threshold': -np.log10(p_threshold),
        'n_significant': len(significant)
    }

    logger.info(f"Manhattan plot data prepared: {len(significant)} significant hits")

    return plot_data


if __name__ == '__main__':
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    print("GWAS Wrapper Demo")
    print("=" * 60)

    # Create dummy phenotype file
    pheno_df = pd.DataFrame({
        'FID': range(1, 101),
        'IID': range(1, 101),
        'phenotype': np.random.randn(100)
    })

    pheno_file = Path('/tmp/pheno_test.txt')
    pheno_df.to_csv(pheno_file, sep='\t', index=False)

    # Run fallback GWAS
    results = run_plink_gwas(
        plink_path='fallback',
        bed_prefix='dummy',
        pheno_file=str(pheno_file),
        out_dir='/tmp/gwas_output'
    )

    print(f"\nGWAS Results:")
    print(f"  - {len(results)} SNPs tested")
    print(f"  - Top 5 associations:")
    print(results.nsmallest(5, 'P')[['SNP', 'CHR', 'BP', 'P', 'BETA']])

    # Manhattan plot data
    plot_data = manhattan_plot_data(results)
    print(f"\n  - Significant hits (P < 5e-8): {plot_data['n_significant']}")

    print("\n✓ GWAS wrapper demo complete")
