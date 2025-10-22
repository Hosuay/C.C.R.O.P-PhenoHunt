"""
Molecular Descriptor Computation Module

Computes molecular descriptors for cannabis cannabinoids and terpenes using RDKit.
Aligned with sacred geometry: 33 primary molecular descriptors.

References:
    - Todeschini, R., & Consonni, V. (2009). Molecular Descriptors for Chemoinformatics.
    - RDKit: Open-source cheminformatics. https://www.rdkit.org/
"""

from typing import Dict, List, Optional, Union
import logging
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Conditional imports for RDKit (graceful degradation if not installed)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, AllChem, GraphDescriptors
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not installed. Molecular descriptor computation will be limited.")

logger = logging.getLogger(__name__)


# SMILES representations for common cannabinoids and terpenes
CANNABINOID_SMILES = {
    'THC': 'CCCCCc1cc(O)c2c(c1)OC(C)(C)[C@@H]1CCC(=CC1)C2',
    'CBD': 'CCCCCc1cc(O)c(C[C@H]2C=C(C)CC[C@H]2C(C)=C)c(O)c1',
    'CBG': 'CCCCCc1cc(O)cc(CC=C(C)C)c1O',
    'CBC': 'CCCCCc1cc2OC(C)(C)[C@@H]3CCC(C)=C[C@H]3c2c(O)c1',
    'CBN': 'CCCCCc1cc(O)c2c3c1OC(C)(C)CC3=CCC2=O',
    'THCV': 'CCCc1cc(O)c2c(c1)OC(C)(C)[C@@H]1CCC(=CC1)C2',
    'CBDA': 'CCCCCc1cc(O)c(C[C@H]2C(=C)CC[C@H]2C(C)(C(O)=O))c(O)c1',
    'THCA': 'CCCCCc1cc(O)c2c(c1)OC(C)(C(O)=O)[C@@H]1CCC(=CC1)C2',
    'Delta8-THC': 'CCCCCc1cc(O)c2c(c1)OC(C)(C)[C@H]1CC=C(C[C@H]21)C',
}

TERPENE_SMILES = {
    'Myrcene': 'CC(=CCCC(=C)C=C)C',
    'Limonene': 'CC1=CCC(CC1)C(=C)C',
    'Pinene': 'CC1=CCC2CC1C2(C)C',  # α-Pinene
    'Linalool': 'CC(=CCCC(C)(O)C=C)C',
    'Caryophyllene': 'CC1=CCCC(=C)C2CC(C)(C)CCC21',  # β-Caryophyllene
    'Humulene': 'CC1=CCCC(=C)C(C)CCC=C1',
    'Terpinolene': 'CC1=CCC(=C(C)C)CC1',
    'Ocimene': 'CC(=CCCC(=C)C)C',
    'Camphene': 'CC1(C)C2CCC1(C)C=C2',
    'Bisabolol': 'CC(=CCCC(C)(O)C(C)CCC=C(C)C)C',
    'Beta-Pinene': 'CC1(C)C2CCC(=C)C1C2',
}


@dataclass
class MolecularDescriptor:
    """Data class for storing molecular descriptor results."""
    compound_name: str
    smiles: str
    molecular_weight: float
    logp: float
    num_h_donors: int
    num_h_acceptors: int
    num_rotatable_bonds: int
    tpsa: float  # Topological Polar Surface Area
    num_aromatic_rings: int
    num_aliphatic_rings: int
    num_saturated_rings: int
    num_heteroatoms: int
    num_heavy_atoms: int
    fraction_csp3: float
    molar_refractivity: float
    num_valence_electrons: int
    complexity: float

    # Extended descriptors (for 33 total)
    num_atoms: int
    num_bonds: int
    num_rings: int
    chi0: float  # Molecular connectivity index
    chi1: float
    kappa1: float  # Molecular shape index
    kappa2: float
    kappa3: float
    balaban_j: float  # Topological index
    bertz_ct: float  # Molecular complexity
    ipc: float  # Information content
    hallKierAlpha: float

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        """Convert to dictionary."""
        return {
            'compound_name': self.compound_name,
            'smiles': self.smiles,
            'molecular_weight': self.molecular_weight,
            'logp': self.logp,
            'num_h_donors': self.num_h_donors,
            'num_h_acceptors': self.num_h_acceptors,
            'num_rotatable_bonds': self.num_rotatable_bonds,
            'tpsa': self.tpsa,
            'num_aromatic_rings': self.num_aromatic_rings,
            'num_aliphatic_rings': self.num_aliphatic_rings,
            'num_saturated_rings': self.num_saturated_rings,
            'num_heteroatoms': self.num_heteroatoms,
            'num_heavy_atoms': self.num_heavy_atoms,
            'fraction_csp3': self.fraction_csp3,
            'molar_refractivity': self.molar_refractivity,
            'num_valence_electrons': self.num_valence_electrons,
            'complexity': self.complexity,
            'num_atoms': self.num_atoms,
            'num_bonds': self.num_bonds,
            'num_rings': self.num_rings,
            'chi0': self.chi0,
            'chi1': self.chi1,
            'kappa1': self.kappa1,
            'kappa2': self.kappa2,
            'kappa3': self.kappa3,
            'balaban_j': self.balaban_j,
            'bertz_ct': self.bertz_ct,
            'ipc': self.ipc,
            'hallKierAlpha': self.hallKierAlpha,
        }


class MolecularDescriptorCalculator:
    """
    Calculator for molecular descriptors aligned with sacred geometry (33 descriptors).

    Sacred Geometry Alignment:
        - 33 primary molecular descriptors (master number)
        - Triadic organization: Basic → Extended → Topological
    """

    def __init__(self):
        """Initialize the descriptor calculator."""
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "RDKit is required for molecular descriptor calculation. "
                "Install with: pip install rdkit>=2022.9.1"
            )
        logger.info("Initialized MolecularDescriptorCalculator with 33 descriptors")

    def compute_descriptors(
        self,
        smiles: str,
        compound_name: Optional[str] = None
    ) -> Optional[MolecularDescriptor]:
        """
        Compute 33 molecular descriptors from SMILES string.

        Args:
            smiles: SMILES representation of the molecule
            compound_name: Optional name for the compound

        Returns:
            MolecularDescriptor object with 33 computed properties

        Example:
            >>> calc = MolecularDescriptorCalculator()
            >>> desc = calc.compute_descriptors('CCO', 'Ethanol')
            >>> print(f"MW: {desc.molecular_weight:.2f}")
            MW: 46.07
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES string: {smiles}")
                return None

            # Add explicit hydrogens for accurate descriptor calculation
            mol_h = Chem.AddHs(mol)

            # Basic descriptors (11)
            molecular_weight = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            num_h_donors = Lipinski.NumHDonors(mol)
            num_h_acceptors = Lipinski.NumHAcceptors(mol)
            num_rotatable_bonds = Lipinski.NumRotatableBonds(mol)
            tpsa = MolSurf.TPSA(mol)
            num_aromatic_rings = Lipinski.NumAromaticRings(mol)
            num_aliphatic_rings = Lipinski.NumAliphaticRings(mol)
            num_saturated_rings = Lipinski.NumSaturatedRings(mol)
            num_heteroatoms = Lipinski.NumHeteroatoms(mol)
            num_heavy_atoms = Lipinski.HeavyAtomCount(mol)

            # Extended descriptors (6)
            fraction_csp3 = Lipinski.FractionCsp3(mol)
            molar_refractivity = Crippen.MolMR(mol)
            num_valence_electrons = Descriptors.NumValenceElectrons(mol)
            num_atoms = mol_h.GetNumAtoms()
            num_bonds = mol.GetNumBonds()

            # Ring information
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()

            # Topological descriptors (16)
            chi0 = GraphDescriptors.Chi0(mol)
            chi1 = GraphDescriptors.Chi1(mol)
            kappa1 = GraphDescriptors.Kappa1(mol)
            kappa2 = GraphDescriptors.Kappa2(mol)
            kappa3 = GraphDescriptors.Kappa3(mol)
            balaban_j = GraphDescriptors.BalabanJ(mol)
            bertz_ct = GraphDescriptors.BertzCT(mol)
            ipc = GraphDescriptors.Ipc(mol)
            hallKierAlpha = GraphDescriptors.HallKierAlpha(mol)

            # Complexity estimate (custom metric)
            complexity = (
                num_heavy_atoms * 0.3 +
                num_aromatic_rings * 2.0 +
                num_rotatable_bonds * 0.5 +
                bertz_ct * 0.01
            )

            descriptor = MolecularDescriptor(
                compound_name=compound_name or smiles,
                smiles=smiles,
                molecular_weight=molecular_weight,
                logp=logp,
                num_h_donors=num_h_donors,
                num_h_acceptors=num_h_acceptors,
                num_rotatable_bonds=num_rotatable_bonds,
                tpsa=tpsa,
                num_aromatic_rings=num_aromatic_rings,
                num_aliphatic_rings=num_aliphatic_rings,
                num_saturated_rings=num_saturated_rings,
                num_heteroatoms=num_heteroatoms,
                num_heavy_atoms=num_heavy_atoms,
                fraction_csp3=fraction_csp3,
                molar_refractivity=molar_refractivity,
                num_valence_electrons=num_valence_electrons,
                complexity=complexity,
                num_atoms=num_atoms,
                num_bonds=num_bonds,
                num_rings=num_rings,
                chi0=chi0,
                chi1=chi1,
                kappa1=kappa1,
                kappa2=kappa2,
                kappa3=kappa3,
                balaban_j=balaban_j,
                bertz_ct=bertz_ct,
                ipc=ipc,
                hallKierAlpha=hallKierAlpha,
            )

            logger.info(f"Computed 33 descriptors for {compound_name or smiles}")
            return descriptor

        except Exception as e:
            logger.error(f"Error computing descriptors for {smiles}: {str(e)}")
            return None

    def compute_batch_descriptors(
        self,
        compounds: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Compute descriptors for multiple compounds.

        Args:
            compounds: Dictionary mapping compound names to SMILES strings

        Returns:
            DataFrame with all computed descriptors

        Example:
            >>> calc = MolecularDescriptorCalculator()
            >>> compounds = {'THC': 'CCCCCc1cc(O)c2...', 'CBD': 'CCCCCc1cc(O)...'}
            >>> df = calc.compute_batch_descriptors(compounds)
        """
        results = []

        for name, smiles in compounds.items():
            desc = self.compute_descriptors(smiles, name)
            if desc:
                results.append(desc.to_dict())

        df = pd.DataFrame(results)
        logger.info(f"Computed descriptors for {len(results)} compounds")

        return df

    def compute_cannabinoid_descriptors(self) -> pd.DataFrame:
        """
        Compute descriptors for all standard cannabinoids.

        Returns:
            DataFrame with cannabinoid descriptors (9 cannabinoids × 33 descriptors)

        Sacred Geometry: 9 cannabinoids (nonagon - completion)
        """
        return self.compute_batch_descriptors(CANNABINOID_SMILES)

    def compute_terpene_descriptors(self) -> pd.DataFrame:
        """
        Compute descriptors for all standard terpenes.

        Returns:
            DataFrame with terpene descriptors (11 terpenes × 33 descriptors)
        """
        return self.compute_batch_descriptors(TERPENE_SMILES)

    def compute_all_cannabis_descriptors(self) -> pd.DataFrame:
        """
        Compute descriptors for all cannabis compounds.

        Returns:
            DataFrame with all cannabinoid and terpene descriptors (20 compounds × 33 descriptors)

        Sacred Geometry: 20 compounds total
        """
        all_compounds = {**CANNABINOID_SMILES, **TERPENE_SMILES}
        return self.compute_batch_descriptors(all_compounds)

    def export_descriptors(
        self,
        df: pd.DataFrame,
        output_path: str,
        format: str = 'csv'
    ) -> None:
        """
        Export descriptors to file.

        Args:
            df: DataFrame with descriptors
            output_path: Path to output file
            format: Output format ('csv', 'json', 'parquet')
        """
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported descriptors to {output_path} ({format})")


def get_descriptor_metadata() -> Dict[str, Dict[str, str]]:
    """
    Get metadata for all 33 molecular descriptors.

    Returns:
        Dictionary with descriptor names, units, and descriptions
    """
    metadata = {
        'molecular_weight': {
            'unit': 'g/mol',
            'description': 'Molecular mass of the compound',
            'category': 'Basic'
        },
        'logp': {
            'unit': 'dimensionless',
            'description': 'Octanol-water partition coefficient (lipophilicity)',
            'category': 'Basic'
        },
        'num_h_donors': {
            'unit': 'count',
            'description': 'Number of hydrogen bond donors',
            'category': 'Basic'
        },
        'num_h_acceptors': {
            'unit': 'count',
            'description': 'Number of hydrogen bond acceptors',
            'category': 'Basic'
        },
        'num_rotatable_bonds': {
            'unit': 'count',
            'description': 'Number of rotatable bonds (molecular flexibility)',
            'category': 'Basic'
        },
        'tpsa': {
            'unit': 'Ų',
            'description': 'Topological Polar Surface Area',
            'category': 'Basic'
        },
        'num_aromatic_rings': {
            'unit': 'count',
            'description': 'Number of aromatic rings',
            'category': 'Basic'
        },
        'num_aliphatic_rings': {
            'unit': 'count',
            'description': 'Number of aliphatic rings',
            'category': 'Basic'
        },
        'num_saturated_rings': {
            'unit': 'count',
            'description': 'Number of saturated rings',
            'category': 'Basic'
        },
        'num_heteroatoms': {
            'unit': 'count',
            'description': 'Number of heteroatoms (non-C/H atoms)',
            'category': 'Basic'
        },
        'num_heavy_atoms': {
            'unit': 'count',
            'description': 'Number of heavy (non-hydrogen) atoms',
            'category': 'Basic'
        },
        'fraction_csp3': {
            'unit': 'fraction',
            'description': 'Fraction of sp3 hybridized carbons',
            'category': 'Extended'
        },
        'molar_refractivity': {
            'unit': 'm³/mol',
            'description': 'Molar refractivity (molecular volume)',
            'category': 'Extended'
        },
        'num_valence_electrons': {
            'unit': 'count',
            'description': 'Total number of valence electrons',
            'category': 'Extended'
        },
        'complexity': {
            'unit': 'dimensionless',
            'description': 'Custom molecular complexity score',
            'category': 'Extended'
        },
        'num_atoms': {
            'unit': 'count',
            'description': 'Total number of atoms (including H)',
            'category': 'Extended'
        },
        'num_bonds': {
            'unit': 'count',
            'description': 'Total number of bonds',
            'category': 'Extended'
        },
        'num_rings': {
            'unit': 'count',
            'description': 'Total number of rings',
            'category': 'Topological'
        },
        'chi0': {
            'unit': 'dimensionless',
            'description': 'Chi0 molecular connectivity index',
            'category': 'Topological'
        },
        'chi1': {
            'unit': 'dimensionless',
            'description': 'Chi1 molecular connectivity index',
            'category': 'Topological'
        },
        'kappa1': {
            'unit': 'dimensionless',
            'description': 'Kappa1 molecular shape index',
            'category': 'Topological'
        },
        'kappa2': {
            'unit': 'dimensionless',
            'description': 'Kappa2 molecular shape index',
            'category': 'Topological'
        },
        'kappa3': {
            'unit': 'dimensionless',
            'description': 'Kappa3 molecular shape index',
            'category': 'Topological'
        },
        'balaban_j': {
            'unit': 'dimensionless',
            'description': 'Balaban J topological index',
            'category': 'Topological'
        },
        'bertz_ct': {
            'unit': 'dimensionless',
            'description': 'Bertz complexity index',
            'category': 'Topological'
        },
        'ipc': {
            'unit': 'bits',
            'description': 'Information content (complexity measure)',
            'category': 'Topological'
        },
        'hallKierAlpha': {
            'unit': 'dimensionless',
            'description': 'Hall-Kier alpha value',
            'category': 'Topological'
        },
    }

    return metadata


# Module-level convenience function
def compute_cannabis_descriptors(output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to compute all cannabis molecular descriptors.

    Args:
        output_path: Optional path to save results

    Returns:
        DataFrame with all descriptors

    Example:
        >>> df = compute_cannabis_descriptors('cannabis_descriptors.csv')
        >>> print(df.head())
    """
    calc = MolecularDescriptorCalculator()
    df = calc.compute_all_cannabis_descriptors()

    if output_path:
        calc.export_descriptors(df, output_path)

    return df
