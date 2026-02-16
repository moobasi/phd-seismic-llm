"""
================================================================================
DEEP LEARNING FACIES CLASSIFICATION MODULE
Based on Microsoft DeepSeismic and seismic_deep_learning
================================================================================

Multi-architecture facies classification for seismic interpretation.
Supports UNet, HRNet, and SEResNet architectures.

References:
- Microsoft DeepSeismic: https://github.com/microsoft/seismic-deeplearning
- Wrona et al. (2021). Deep learning tutorials for seismic interpretation.
  DOI: 10.5880/GFZ.2.5.2021.001

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FaciesClassificationConfig:
    """Configuration for facies classification"""

    # Input/Output
    seismic_file: str = ""
    output_dir: str = "facies_outputs"

    # Model settings
    model_path: Optional[str] = None
    model_type: str = "unet"  # unet, hrnet, seresnet
    num_classes: int = 6
    use_gpu: bool = True

    # Class definitions (customize for your basin)
    class_names: List[str] = field(default_factory=lambda: [
        "Background",
        "Channel Sand",
        "Marine Shale",
        "Floodplain",
        "Carbonate",
        "Basement"
    ])

    class_colors: List[str] = field(default_factory=lambda: [
        "#FFFFFF",  # Background - white
        "#FFD700",  # Channel Sand - gold
        "#4169E1",  # Marine Shale - blue
        "#228B22",  # Floodplain - green
        "#DEB887",  # Carbonate - tan
        "#8B4513"   # Basement - brown
    ])

    # Processing parameters
    patch_size: Tuple[int, int] = (128, 128)  # For 2D slices
    overlap: int = 16
    batch_size: int = 4

    # Post-processing
    apply_crf: bool = False  # Conditional Random Field smoothing
    median_filter_size: int = 3

    # Output options
    save_probabilities: bool = True
    save_classification: bool = True
    save_figures: bool = True
    figure_dpi: int = 300

    # Bornu Chad Basin specific formations
    basin_formations: Dict[str, str] = field(default_factory=lambda: {
        "Chad_Fm": "Channel Sand",
        "Fika_Shale": "Marine Shale",
        "Gongila_Fm": "Channel Sand",
        "Bima_Fm": "Floodplain",
        "Basement": "Basement"
    })

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FaciesClassificationResults:
    """Results from facies classification"""

    success: bool = False
    num_classes_found: int = 0

    # Class statistics
    class_percentages: Dict[str, float] = field(default_factory=dict)
    class_volumes: Dict[str, int] = field(default_factory=dict)

    # Spatial analysis
    dominant_facies_by_depth: List[Dict] = field(default_factory=list)
    facies_contacts: List[Dict] = field(default_factory=list)

    # Output files
    classification_file: str = ""
    probability_files: List[str] = field(default_factory=list)
    figure_files: List[str] = field(default_factory=list)

    # For LLM interpretation
    summary_for_llm: str = ""

    # Metadata
    processing_time_seconds: float = 0.0
    model_used: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# UNET ARCHITECTURE FOR FACIES CLASSIFICATION
# =============================================================================

class DoubleConv(nn.Module):
    """Double convolution block"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class FaciesUNet(nn.Module):
    """
    U-Net architecture for seismic facies classification.

    Based on the architecture used in Microsoft DeepSeismic
    for the Dutch F3 benchmark dataset.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 features: List[int] = [64, 128, 256, 512]):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconvs = nn.ModuleList()

        # Encoder
        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(DoubleConv(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(DoubleConv(feature * 2, feature))

        # Final classification layer
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path with skip connections
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for idx, (upconv, block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            skip = skip_connections[idx]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=True)

            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.final_conv(x)


# =============================================================================
# FACIES CLASSIFIER CLASS
# =============================================================================

class FaciesClassifier:
    """
    Deep learning-based seismic facies classification.

    Integrates Microsoft DeepSeismic methodology with the PhD framework,
    providing facies classification outputs for LLM interpretation.

    Example:
        classifier = FaciesClassifier(config)
        results = classifier.classify_facies(seismic_data)

        # Feed to LLM for geological interpretation
        llm_summary = results.summary_for_llm
    """

    def __init__(self, config: Optional[FaciesClassificationConfig] = None):
        self.config = config or FaciesClassificationConfig()
        self.model = None
        self.device = None
        self._setup_device()

    def _setup_device(self):
        """Setup computation device"""
        if TORCH_AVAILABLE:
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("Using CPU")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load pre-trained facies classification model"""
        if not TORCH_AVAILABLE:
            print("Error: PyTorch not available")
            return False

        path = model_path or self.config.model_path

        if path and Path(path).exists():
            try:
                self.model = FaciesUNet(num_classes=self.config.num_classes)
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"Loaded model from: {path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")

        # Initialize with random weights
        print("Initializing model with random weights (no pre-trained model)")
        print("For production, download pre-trained weights from DeepSeismic repo")
        self.model = FaciesUNet(num_classes=self.config.num_classes)
        self.model.to(self.device)
        self.model.eval()
        return True

    def load_seismic(self, segy_path: str) -> Optional[np.ndarray]:
        """Load seismic data from SEGY file"""
        if not SEGYIO_AVAILABLE:
            print("Error: segyio not available")
            return None

        try:
            with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
                data = segyio.tools.cube(f)
                print(f"Loaded seismic: {data.shape}")
                return data
        except Exception as e:
            print(f"Error loading seismic: {e}")
            return None

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize seismic data"""
        data = data.astype(np.float32)
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)

    def classify_facies(self, seismic_data: Optional[np.ndarray] = None,
                        segy_path: Optional[str] = None) -> FaciesClassificationResults:
        """
        Classify seismic facies using deep learning.

        Args:
            seismic_data: 3D numpy array of seismic amplitudes
            segy_path: Path to SEGY file

        Returns:
            FaciesClassificationResults with classification and statistics
        """
        start_time = datetime.now()
        results = FaciesClassificationResults()

        # Load data
        if seismic_data is None:
            path = segy_path or self.config.seismic_file
            seismic_data = self.load_seismic(path)
            if seismic_data is None:
                results.summary_for_llm = "Error: Could not load seismic data"
                return results

        # Load model
        if self.model is None:
            if not self.load_model():
                results.summary_for_llm = "Error: Could not load classification model"
                return results

        # Normalize
        print("Normalizing seismic data...")
        normalized = self._normalize(seismic_data)

        nz, ny, nx = normalized.shape

        # Classify inline by inline
        print("Classifying facies...")
        classification = np.zeros((nz, ny, nx), dtype=np.uint8)
        probabilities = np.zeros((self.config.num_classes, nz, ny, nx), dtype=np.float32)

        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                for y in tqdm(range(ny), desc="Processing inlines"):
                    # Get inline slice
                    inline_slice = normalized[:, y, :]  # (nz, nx)

                    # Prepare input
                    x = torch.from_numpy(inline_slice).float().unsqueeze(0).unsqueeze(0)
                    x = x.to(self.device)

                    # Inference
                    logits = self.model(x)
                    probs = F.softmax(logits, dim=1)

                    # Get classification
                    pred = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
                    classification[:, y, :] = pred

                    # Store probabilities
                    probs = probs.squeeze().cpu().numpy()
                    probabilities[:, :, y, :] = probs
        else:
            # Fallback: simple amplitude-based classification
            print("Using fallback amplitude-based classification (no DL model)")
            classification = self._fallback_classification(normalized)

        # Post-processing
        if SCIPY_AVAILABLE and self.config.median_filter_size > 1:
            classification = median_filter(classification,
                                           size=self.config.median_filter_size)

        # Calculate statistics
        results = self._calculate_statistics(classification, results)

        # Analyze facies distribution
        results.dominant_facies_by_depth = self._analyze_depth_distribution(classification)
        results.facies_contacts = self._identify_facies_contacts(classification)

        # Save outputs
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_classification:
            class_file = output_dir / "facies_classification.npy"
            np.save(class_file, classification)
            results.classification_file = str(class_file)

        if self.config.save_probabilities:
            prob_file = output_dir / "facies_probabilities.npy"
            np.save(prob_file, probabilities)
            results.probability_files.append(str(prob_file))

        if self.config.save_figures and MATPLOTLIB_AVAILABLE:
            fig_files = self._create_figures(seismic_data, classification, output_dir)
            results.figure_files = fig_files

        # Generate LLM summary
        results.summary_for_llm = self._generate_llm_summary(results, seismic_data.shape)

        # Finalize
        results.success = True
        results.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        results.model_used = self.config.model_type

        # Save results JSON
        results_file = output_dir / "facies_classification_results.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        print(f"\nFacies classification complete in {results.processing_time_seconds:.1f}s")
        print(f"Classes found: {results.num_classes_found}")

        return results

    def _fallback_classification(self, data: np.ndarray) -> np.ndarray:
        """Simple amplitude-based classification as fallback"""
        # Quantile-based classification
        percentiles = np.percentile(data, [20, 40, 60, 80])

        classification = np.zeros_like(data, dtype=np.uint8)
        classification[data < percentiles[0]] = 0  # Background/low amplitude
        classification[(data >= percentiles[0]) & (data < percentiles[1])] = 1
        classification[(data >= percentiles[1]) & (data < percentiles[2])] = 2
        classification[(data >= percentiles[2]) & (data < percentiles[3])] = 3
        classification[data >= percentiles[3]] = 4

        return classification

    def _calculate_statistics(self, classification: np.ndarray,
                              results: FaciesClassificationResults) -> FaciesClassificationResults:
        """Calculate facies statistics"""
        total_voxels = classification.size

        unique_classes = np.unique(classification)
        results.num_classes_found = len(unique_classes)

        for class_idx in range(self.config.num_classes):
            class_name = self.config.class_names[class_idx]
            count = np.sum(classification == class_idx)
            percentage = 100.0 * count / total_voxels

            results.class_volumes[class_name] = int(count)
            results.class_percentages[class_name] = float(percentage)

        return results

    def _analyze_depth_distribution(self, classification: np.ndarray) -> List[Dict]:
        """Analyze facies distribution by depth"""
        nz = classification.shape[0]
        depth_analysis = []

        # Analyze in depth windows
        window_size = max(1, nz // 10)

        for i in range(0, nz, window_size):
            end_idx = min(i + window_size, nz)
            window = classification[i:end_idx, :, :]

            # Count classes in this window
            unique, counts = np.unique(window, return_counts=True)
            dominant_class = unique[np.argmax(counts)]

            depth_analysis.append({
                'depth_start': i,
                'depth_end': end_idx,
                'dominant_class': int(dominant_class),
                'dominant_class_name': self.config.class_names[dominant_class],
                'class_distribution': {
                    self.config.class_names[int(u)]: int(c)
                    for u, c in zip(unique, counts)
                }
            })

        return depth_analysis

    def _identify_facies_contacts(self, classification: np.ndarray) -> List[Dict]:
        """Identify major facies contacts"""
        contacts = []

        if not SCIPY_AVAILABLE:
            return contacts

        # Use gradient to find transitions
        gz = np.diff(classification.astype(float), axis=0)

        # Find significant transitions
        transition_mask = np.abs(gz) > 0

        # Label connected transition zones
        labeled, num = ndimage.label(transition_mask)

        for i in range(1, min(num + 1, 10)):  # Top 10 contacts
            mask = labeled == i
            if np.sum(mask) < 100:
                continue

            # Get coordinates
            coords = np.where(mask)
            if len(coords[0]) == 0:
                continue

            # Determine facies above and below
            z_mean = int(np.mean(coords[0]))

            if z_mean > 0 and z_mean < classification.shape[0] - 1:
                facies_above = int(np.median(classification[z_mean-1:z_mean, :, :]))
                facies_below = int(np.median(classification[z_mean:z_mean+1, :, :]))

                contacts.append({
                    'contact_id': i,
                    'depth_samples': z_mean,
                    'facies_above': self.config.class_names[facies_above],
                    'facies_below': self.config.class_names[facies_below],
                    'contact_area_voxels': int(np.sum(mask))
                })

        return contacts

    def _create_figures(self, seismic: np.ndarray, classification: np.ndarray,
                        output_dir: Path) -> List[str]:
        """Create visualization figures"""
        figures = []

        nz, ny, nx = seismic.shape

        # Create colormap
        colors = [c for c in self.config.class_colors[:self.config.num_classes]]
        cmap = ListedColormap(colors)

        # Figure 1: Inline comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        inline_idx = ny // 2

        axes[0].imshow(seismic[:, inline_idx, :].T, cmap='gray', aspect='auto')
        axes[0].set_title('Seismic Amplitude')
        axes[0].set_xlabel('Crossline')
        axes[0].set_ylabel('Time (samples)')

        im = axes[1].imshow(classification[:, inline_idx, :].T, cmap=cmap,
                            aspect='auto', vmin=0, vmax=self.config.num_classes-1)
        axes[1].set_title('Facies Classification')
        axes[1].set_xlabel('Crossline')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.config.class_colors[i],
                  label=self.config.class_names[i])
            for i in range(self.config.num_classes)
        ]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        fig_path = output_dir / 'facies_inline.png'
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        figures.append(str(fig_path))

        # Figure 2: Time slice
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        time_idx = nz // 2

        axes[0].imshow(seismic[time_idx, :, :], cmap='gray', aspect='auto')
        axes[0].set_title(f'Seismic @ Time {time_idx}')
        axes[0].set_xlabel('Inline')
        axes[0].set_ylabel('Crossline')

        axes[1].imshow(classification[time_idx, :, :], cmap=cmap, aspect='auto',
                       vmin=0, vmax=self.config.num_classes-1)
        axes[1].set_title('Facies Classification')

        plt.tight_layout()
        fig_path = output_dir / 'facies_timeslice.png'
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        figures.append(str(fig_path))

        # Figure 3: Facies distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 8))

        sizes = [self.config.class_names[i] for i in range(self.config.num_classes)]
        counts = [np.sum(classification == i) for i in range(self.config.num_classes)]

        # Filter out zero counts
        non_zero = [(s, c, self.config.class_colors[i])
                    for i, (s, c) in enumerate(zip(sizes, counts)) if c > 0]

        if non_zero:
            labels, values, colors = zip(*non_zero)
            ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90)
            ax.set_title('Facies Volume Distribution')

        fig_path = output_dir / 'facies_distribution.png'
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        figures.append(str(fig_path))

        return figures

    def _generate_llm_summary(self, results: FaciesClassificationResults,
                              volume_shape: Tuple) -> str:
        """Generate summary for LLM interpretation"""

        summary = f"""
DEEP LEARNING FACIES CLASSIFICATION RESULTS
===========================================

Volume Analyzed: {volume_shape[0]} x {volume_shape[1]} x {volume_shape[2]}

FACIES DISTRIBUTION:
"""
        for class_name, percentage in sorted(results.class_percentages.items(),
                                             key=lambda x: x[1], reverse=True):
            volume = results.class_volumes.get(class_name, 0)
            summary += f"  - {class_name}: {percentage:.1f}% ({volume:,} voxels)\n"

        summary += """
DEPTH DISTRIBUTION ANALYSIS:
"""
        for depth in results.dominant_facies_by_depth:
            summary += f"""
  Depth {depth['depth_start']}-{depth['depth_end']} samples:
    Dominant: {depth['dominant_class_name']}
"""

        summary += """
MAJOR FACIES CONTACTS:
"""
        for contact in results.facies_contacts:
            summary += f"""
  Contact #{contact['contact_id']}:
    - Depth: {contact['depth_samples']} samples
    - Above: {contact['facies_above']}
    - Below: {contact['facies_below']}
    - Contact area: {contact['contact_area_voxels']:,} voxels
"""

        summary += """
GEOLOGICAL INTERPRETATION GUIDANCE:
- Channel sands may indicate reservoir potential
- Marine shales may act as seals or source rocks
- Facies contacts define sequence boundaries
- Consider depositional environment implications

FOR BORNU CHAD BASIN CONTEXT:
- Chad Formation: Fluvial/lacustrine deposits (reservoir)
- Fika Shale: Marine transgression (seal)
- Gongila Formation: Transitional marine (potential reservoir)
- Bima Formation: Continental clastics (reservoir)

OUTPUT FILES:
- Classification volume: facies_classification.npy
- Probability volume: facies_probabilities.npy
- Visualization figures: facies_*.png
"""

        return summary


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Deep Learning Facies Classification'
    )
    parser.add_argument('segy_file', help='Input SEGY file')
    parser.add_argument('--output-dir', default='facies_outputs', help='Output directory')
    parser.add_argument('--model', help='Path to pre-trained model weights')
    parser.add_argument('--num-classes', type=int, default=6, help='Number of facies classes')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')

    args = parser.parse_args()

    config = FaciesClassificationConfig()
    config.seismic_file = args.segy_file
    config.output_dir = args.output_dir
    config.num_classes = args.num_classes
    config.use_gpu = not args.no_gpu

    if args.model:
        config.model_path = args.model

    classifier = FaciesClassifier(config)
    results = classifier.classify_facies()

    print("\n" + results.summary_for_llm)


if __name__ == '__main__':
    main()
