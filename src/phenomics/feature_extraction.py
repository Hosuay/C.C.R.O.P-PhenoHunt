"""
Feature Extraction Module for Phenomics Pipeline
=================================================

Extracts morphological traits from cannabis plant images:
    - Trichome density and distribution
    - Bud structure and compactness
    - Leaf morphology and color
    - Plant height and biomass estimation

Sacred Geometry:
    - 9 primary phenotypic features
    - 27-dimensional feature space (3³)

Author: C.C.R.O.P-PhenoHunt Team
Version: 1.0.0
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try to import image processing libraries
try:
    from PIL import Image
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available. Using basic feature extraction.")


class FeatureExtractor:
    """
    Extract phenotypic features from plant images.

    Sacred Geometry Alignment:
        - 9 primary features extracted
        - 27-dimensional embedding (with extended features)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature extractor.

        Args:
            config: Configuration dict with extraction parameters
        """
        self.config = config or self._default_config()
        logger.info("FeatureExtractor initialized")

    def _default_config(self) -> Dict:
        """Default configuration with harmonic parameters."""
        return {
            'target_size': (369, 369),  # Sacred 369 dimension
            'color_space': 'RGB',
            'normalize': True,
            'extract_color': True,
            'extract_texture': True,
            'extract_shape': True,
            'trichome_detection': True
        }

    def extract_features(self, image_path: Path) -> Dict[str, float]:
        """
        Extract all features from a single image.

        Args:
            image_path: Path to plant image

        Returns:
            Dictionary of extracted features
        """
        logger.info(f"Extracting features from: {image_path.name}")

        # Load image
        if not HAS_CV2:
            return self._extract_basic_features(image_path)

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        features = {}

        # Extract 9 primary features (sacred geometry)
        features.update(self._extract_color_features(img_rgb))  # 3 features
        features.update(self._extract_texture_features(img_rgb))  # 3 features
        features.update(self._extract_shape_features(img_rgb))  # 3 features

        # Extended features for 27-dimensional space
        if self.config.get('trichome_detection', True):
            features.update(self._extract_trichome_features(img_rgb))

        logger.info(f"Extracted {len(features)} features")
        return features

    def _extract_basic_features(self, image_path: Path) -> Dict[str, float]:
        """Basic feature extraction without OpenCV."""
        img = Image.open(image_path)
        img_array = np.array(img)

        features = {
            # Color features (3)
            'mean_red': float(np.mean(img_array[:, :, 0])) if img_array.ndim == 3 else 0.0,
            'mean_green': float(np.mean(img_array[:, :, 1])) if img_array.ndim == 3 else 0.0,
            'mean_blue': float(np.mean(img_array[:, :, 2])) if img_array.ndim == 3 else 0.0,

            # Intensity features (3)
            'mean_intensity': float(np.mean(img_array)),
            'std_intensity': float(np.std(img_array)),
            'intensity_range': float(np.ptp(img_array)),

            # Shape features (3)
            'image_height': float(img_array.shape[0]),
            'image_width': float(img_array.shape[1]),
            'aspect_ratio': float(img_array.shape[1] / img_array.shape[0]) if img_array.shape[0] > 0 else 1.0
        }

        return features

    def _extract_color_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract color-based features (3 primary features)."""
        # Convert to HSV for better color analysis
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        features = {
            'mean_hue': float(np.mean(img_hsv[:, :, 0])),
            'mean_saturation': float(np.mean(img_hsv[:, :, 1])),
            'mean_value': float(np.mean(img_hsv[:, :, 2]))
        }

        return features

    def _extract_texture_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract texture features (3 primary features)."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Calculate texture metrics
        features = {
            'texture_contrast': float(np.std(gray)),
            'texture_homogeneity': float(1.0 / (1.0 + np.std(gray))),
            'texture_entropy': self._calculate_entropy(gray)
        }

        return features

    def _extract_shape_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features (3 primary features)."""
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return {
                'plant_area': 0.0,
                'perimeter': 0.0,
                'compactness': 0.0
            }

        # Get largest contour (assume it's the plant)
        largest_contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)

        # Compactness: 4π * area / perimeter²
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

        features = {
            'plant_area': float(area),
            'perimeter': float(perimeter),
            'compactness': float(compactness)
        }

        return features

    def _extract_trichome_features(self, img: np.ndarray) -> Dict[str, float]:
        """
        Extract trichome-related features (advanced).

        Uses high-frequency texture analysis to estimate trichome density.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply Laplacian for edge detection (trichomes appear as high-frequency details)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        trichome_intensity = float(np.mean(np.abs(laplacian)))

        # Estimate density from high-intensity pixels
        threshold = np.percentile(np.abs(laplacian), 90)
        trichome_density = float(np.sum(np.abs(laplacian) > threshold) / laplacian.size)

        features = {
            'trichome_intensity': trichome_intensity,
            'trichome_density': trichome_density,
            'trichome_coverage': trichome_density * 100  # As percentage
        }

        return features

    def _calculate_entropy(self, img: np.ndarray) -> float:
        """Calculate Shannon entropy of image."""
        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)


def extract_from_directory(
    image_dir: Path,
    output_file: Path,
    config: Optional[Dict] = None
) -> Dict:
    """
    Extract features from all images in a directory.

    Args:
        image_dir: Directory containing images
        output_file: Output JSON file for features
        config: Optional configuration dict

    Returns:
        Dictionary with all extracted features
    """
    extractor = FeatureExtractor(config)

    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    # Find all images
    image_files = [
        f for f in Path(image_dir).glob('*')
        if f.suffix.lower() in image_extensions
    ]

    logger.info(f"Found {len(image_files)} images in {image_dir}")

    # Extract features from each image
    all_features = {}
    for img_path in image_files:
        try:
            features = extractor.extract_features(img_path)
            all_features[img_path.name] = features
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            all_features[img_path.name] = {'error': str(e)}

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=2)

    logger.info(f"Saved features to {output_file}")

    return all_features


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Feature Extractor Demo")
    print("=" * 50)

    # Create sample image (for testing without real images)
    sample_img = np.random.randint(0, 255, (369, 369, 3), dtype=np.uint8)
    sample_path = Path('sample_plant.png')

    if HAS_CV2:
        cv2.imwrite(str(sample_path), sample_img)

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_path)

        print(f"\nExtracted {len(features)} features:")
        for name, value in features.items():
            print(f"  {name}: {value:.4f}")

        sample_path.unlink()  # Clean up
    else:
        print("OpenCV not available. Install with: pip install opencv-python")
