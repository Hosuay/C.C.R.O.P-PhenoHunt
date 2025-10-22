"""
3D Molecular Visualization Module

Provides 3D rendering of molecular structures for cannabinoids and terpenes
using RDKit and py3Dmol for interactive visualization.

Sacred Geometry Alignment: 3D spatial representation aligned with natural forms
"""

from typing import Optional, Dict, List, Union, Tuple
import logging

# Conditional imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import IPythonConsole
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. 3D structure generation limited.")

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False
    logging.warning("py3Dmol not available. Interactive 3D visualization limited.")

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import numpy as np

logger = logging.getLogger(__name__)


class Molecular3DVisualizer:
    """
    3D molecular structure visualizer for cannabis compounds.

    Features:
        - Generate 3D conformers from SMILES
        - Interactive 3D rendering with py3Dmol
        - Export to multiple formats (PNG, SVG, PDB, MOL)
        - Customizable styling with harmonic color schemes
    """

    def __init__(self, size: Tuple[int, int] = (600, 600)):
        """
        Initialize the 3D visualizer.

        Args:
            size: Tuple of (width, height) for rendered images
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required. Install with: pip install rdkit>=2022.9.1")

        self.size = size
        logger.info(f"Initialized Molecular3DVisualizer with size {size}")

    def generate_3d_conformer(
        self,
        smiles: str,
        optimize: bool = True,
        num_conformers: int = 1,
        random_seed: int = 27  # Harmonic seed
    ) -> Optional[Chem.Mol]:
        """
        Generate 3D conformer from SMILES string.

        Args:
            smiles: SMILES representation
            optimize: Whether to optimize geometry with MMFF
            num_conformers: Number of conformers to generate
            random_seed: Random seed (default 27 for harmonic alignment)

        Returns:
            RDKit molecule object with 3D coordinates

        Example:
            >>> viz = Molecular3DVisualizer()
            >>> mol_3d = viz.generate_3d_conformer('CCO')
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return None

            # Add explicit hydrogens
            mol_h = Chem.AddHs(mol)

            # Generate 3D coordinates
            if num_conformers == 1:
                AllChem.EmbedMolecule(mol_h, randomSeed=random_seed)
            else:
                AllChem.EmbedMultipleConfs(
                    mol_h,
                    numConfs=num_conformers,
                    randomSeed=random_seed
                )

            # Optimize geometry with MMFF force field
            if optimize:
                if num_conformers == 1:
                    AllChem.MMFFOptimizeMolecule(mol_h)
                else:
                    for conf_id in range(num_conformers):
                        AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id)

            logger.info(f"Generated 3D conformer for {smiles}")
            return mol_h

        except Exception as e:
            logger.error(f"Error generating 3D conformer: {str(e)}")
            return None

    def visualize_3d_interactive(
        self,
        smiles: str,
        compound_name: str = "",
        style: str = "stick",
        color_scheme: str = "default",
        width: int = 600,
        height: int = 600,
        background_color: str = "white"
    ):
        """
        Create interactive 3D visualization using py3Dmol.

        Args:
            smiles: SMILES representation
            compound_name: Name of compound (for display)
            style: Visualization style ('stick', 'sphere', 'cartoon', 'line')
            color_scheme: Color scheme ('default', 'harmonic', 'cpk', 'rainbow')
            width: Display width in pixels
            height: Display height in pixels
            background_color: Background color

        Returns:
            py3Dmol view object (renders in Jupyter)

        Example:
            >>> viz = Molecular3DVisualizer()
            >>> view = viz.visualize_3d_interactive('CCO', 'Ethanol', style='sphere')
            >>> view.show()
        """
        if not PY3DMOL_AVAILABLE:
            logger.error("py3Dmol not available. Install with: pip install py3Dmol")
            return None

        try:
            # Generate 3D conformer
            mol_3d = self.generate_3d_conformer(smiles)
            if mol_3d is None:
                return None

            # Convert to MOL block for py3Dmol
            mol_block = Chem.MolToMolBlock(mol_3d)

            # Create py3Dmol view
            view = py3Dmol.view(width=width, height=height)
            view.addModel(mol_block, 'mol')

            # Apply style
            style_config = self._get_style_config(style, color_scheme)
            view.setStyle(style_config)

            # Set background
            view.setBackgroundColor(background_color)

            # Add label if compound name provided
            if compound_name:
                view.addLabel(
                    compound_name,
                    {'position': {'x': 0, 'y': 0, 'z': 5}},
                    {'backgroundColor': 'lightgray', 'fontColor': 'black'}
                )

            # Zoom to fit
            view.zoomTo()

            logger.info(f"Created 3D visualization for {compound_name or smiles}")
            return view

        except Exception as e:
            logger.error(f"Error creating 3D visualization: {str(e)}")
            return None

    def _get_style_config(self, style: str, color_scheme: str) -> Dict:
        """
        Get py3Dmol style configuration.

        Args:
            style: Visualization style
            color_scheme: Color scheme

        Returns:
            Dictionary with style configuration
        """
        if color_scheme == "harmonic":
            # Custom harmonic color scheme (aligned with sacred geometry)
            colors = {
                'C': '#FF6B6B',  # Red (base, grounding)
                'O': '#4ECDC4',  # Cyan (oxygen, life)
                'N': '#45B7D1',  # Blue (nitrogen, sky)
                'H': '#F7DC6F',  # Light yellow (hydrogen, light)
            }
        elif color_scheme == "cpk":
            colors = None  # Use default CPK colors
        elif color_scheme == "rainbow":
            colors = None  # Use rainbow coloring
        else:
            colors = None  # Default coloring

        if style == "stick":
            return {'stick': {'colorscheme': color_scheme if color_scheme != "harmonic" else colors}}
        elif style == "sphere":
            return {'sphere': {'colorscheme': color_scheme if color_scheme != "harmonic" else colors}}
        elif style == "cartoon":
            return {'cartoon': {'color': 'spectrum'}}
        elif style == "line":
            return {'line': {'colorscheme': color_scheme if color_scheme != "harmonic" else colors}}
        else:
            return {'stick': {}}

    def render_2d_image(
        self,
        smiles: str,
        output_path: Optional[str] = None,
        image_size: Tuple[int, int] = (500, 500),
        highlight_atoms: Optional[List[int]] = None
    ):
        """
        Render 2D molecular structure image.

        Args:
            smiles: SMILES representation
            output_path: Path to save image (PNG)
            image_size: Image dimensions
            highlight_atoms: List of atom indices to highlight

        Returns:
            PIL Image object

        Example:
            >>> viz = Molecular3DVisualizer()
            >>> img = viz.render_2d_image('CCO', 'ethanol.png')
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error(f"Invalid SMILES: {smiles}")
                return None

            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)

            # Draw molecule
            if highlight_atoms:
                img = Draw.MolToImage(
                    mol,
                    size=image_size,
                    highlightAtoms=highlight_atoms
                )
            else:
                img = Draw.MolToImage(mol, size=image_size)

            # Save if path provided
            if output_path and PIL_AVAILABLE:
                img.save(output_path)
                logger.info(f"Saved 2D image to {output_path}")

            return img

        except Exception as e:
            logger.error(f"Error rendering 2D image: {str(e)}")
            return None

    def export_3d_structure(
        self,
        smiles: str,
        output_path: str,
        format: str = 'pdb'
    ) -> bool:
        """
        Export 3D structure to file.

        Args:
            smiles: SMILES representation
            output_path: Path to output file
            format: Output format ('pdb', 'mol', 'sdf', 'xyz')

        Returns:
            True if successful, False otherwise

        Example:
            >>> viz = Molecular3DVisualizer()
            >>> viz.export_3d_structure('CCO', 'ethanol.pdb', format='pdb')
        """
        try:
            mol_3d = self.generate_3d_conformer(smiles)
            if mol_3d is None:
                return False

            if format == 'pdb':
                Chem.MolToPDBFile(mol_3d, output_path)
            elif format == 'mol':
                writer = Chem.SDWriter(output_path)
                writer.write(mol_3d)
                writer.close()
            elif format == 'sdf':
                writer = Chem.SDWriter(output_path)
                writer.write(mol_3d)
                writer.close()
            elif format == 'xyz':
                Chem.MolToXYZFile(mol_3d, output_path)
            else:
                logger.error(f"Unsupported format: {format}")
                return False

            logger.info(f"Exported 3D structure to {output_path} ({format})")
            return True

        except Exception as e:
            logger.error(f"Error exporting 3D structure: {str(e)}")
            return False

    def compare_structures_3d(
        self,
        smiles_list: List[str],
        names: List[str],
        grid_size: Tuple[int, int] = (2, 2)
    ):
        """
        Compare multiple structures side-by-side in 3D.

        Args:
            smiles_list: List of SMILES strings
            names: List of compound names
            grid_size: Grid dimensions for layout

        Returns:
            py3Dmol view with grid of structures

        Example:
            >>> viz = Molecular3DVisualizer()
            >>> smiles = ['CCO', 'CCC', 'CCCC']
            >>> names = ['Ethanol', 'Propane', 'Butane']
            >>> view = viz.compare_structures_3d(smiles, names)
        """
        if not PY3DMOL_AVAILABLE:
            logger.error("py3Dmol not available")
            return None

        try:
            # Create grid view
            rows, cols = grid_size
            view = py3Dmol.view(
                width=300 * cols,
                height=300 * rows,
                linked=False,
                viewergrid=(rows, cols)
            )

            # Add molecules to grid
            for idx, (smiles, name) in enumerate(zip(smiles_list, names)):
                if idx >= rows * cols:
                    break

                mol_3d = self.generate_3d_conformer(smiles)
                if mol_3d is None:
                    continue

                mol_block = Chem.MolToMolBlock(mol_3d)

                row = idx // cols
                col = idx % cols

                view.addModel(mol_block, 'mol', viewer=(row, col))
                view.setStyle({'stick': {}}, viewer=(row, col))
                view.addLabel(
                    name,
                    {'position': {'x': 0, 'y': 0, 'z': 3}},
                    viewer=(row, col)
                )
                view.zoomTo(viewer=(row, col))

            logger.info(f"Created comparison grid with {len(smiles_list)} structures")
            return view

        except Exception as e:
            logger.error(f"Error creating comparison grid: {str(e)}")
            return None


def visualize_cannabinoid_3d(
    cannabinoid_name: str,
    style: str = "stick",
    interactive: bool = True
):
    """
    Convenience function to visualize a cannabinoid in 3D.

    Args:
        cannabinoid_name: Name of cannabinoid (e.g., 'THC', 'CBD')
        style: Visualization style
        interactive: Whether to return interactive view or image

    Returns:
        py3Dmol view (if interactive) or PIL Image

    Example:
        >>> view = visualize_cannabinoid_3d('THC', style='sphere')
        >>> view.show()
    """
    from ..cheminformatics.molecular_descriptors import CANNABINOID_SMILES

    if cannabinoid_name not in CANNABINOID_SMILES:
        logger.error(f"Unknown cannabinoid: {cannabinoid_name}")
        return None

    smiles = CANNABINOID_SMILES[cannabinoid_name]
    viz = Molecular3DVisualizer()

    if interactive:
        return viz.visualize_3d_interactive(smiles, cannabinoid_name, style=style)
    else:
        return viz.render_2d_image(smiles)


def visualize_terpene_3d(
    terpene_name: str,
    style: str = "stick",
    interactive: bool = True
):
    """
    Convenience function to visualize a terpene in 3D.

    Args:
        terpene_name: Name of terpene (e.g., 'Myrcene', 'Limonene')
        style: Visualization style
        interactive: Whether to return interactive view or image

    Returns:
        py3Dmol view (if interactive) or PIL Image

    Example:
        >>> view = visualize_terpene_3d('Limonene', style='sphere')
        >>> view.show()
    """
    from ..cheminformatics.molecular_descriptors import TERPENE_SMILES

    if terpene_name not in TERPENE_SMILES:
        logger.error(f"Unknown terpene: {terpene_name}")
        return None

    smiles = TERPENE_SMILES[terpene_name]
    viz = Molecular3DVisualizer()

    if interactive:
        return viz.visualize_3d_interactive(smiles, terpene_name, style=style)
    else:
        return viz.render_2d_image(smiles)
