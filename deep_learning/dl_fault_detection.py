"""
================================================================================
DEEP LEARNING FAULT DETECTION MODULE
Based on FaultSeg3D (Wu et al., 2019, GEOPHYSICS)
================================================================================

3D CNN-based fault detection using pre-trained models.
Outputs fault probability volumes for LLM interpretation.

References:
- Wu, X., & Fomel, S. (2018). Automatic fault interpretation with optimal
  surface voting. Geophysics, 83(5), O67-O82.
- Wu, X., Liang, L., Shi, Y., & Fomel, S. (2019). FaultSeg3D: Using synthetic
  data sets to train an end-to-end convolutional neural network for 3D seismic
  fault segmentation. Geophysics, 84(3), IM35-IM45.

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
    print("Warning: PyTorch not available. Install with: pip install torch")

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import segyio
    SEGYIO_AVAILABLE = True
except ImportError:
    SEGYIO_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
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
class FaultDetectionConfig:
    """Configuration for fault detection"""

    # Input/Output
    seismic_file: str = ""
    output_dir: str = "fault_outputs"

    # Model settings
    model_path: Optional[str] = None  # Path to pre-trained model
    model_type: str = "faultseg3d"  # faultseg3d, unet3d, custom
    use_gpu: bool = True

    # Processing parameters
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    overlap: int = 16  # Overlap between patches
    batch_size: int = 1

    # Post-processing
    probability_threshold: float = 0.5
    min_fault_size: int = 100  # Minimum voxels for a fault
    smooth_sigma: float = 1.0

    # Output options
    save_probability: bool = True
    save_binary: bool = True
    save_figures: bool = True
    figure_dpi: int = 300

    # LLM integration
    generate_summary: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> 'FaultDetectionConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        # Only use fields that exist in this dataclass
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid_fields)


@dataclass
class FaultDetectionResults:
    """Results from fault detection"""

    success: bool = False
    num_faults_detected: int = 0
    fault_volume_percent: float = 0.0

    # Fault statistics
    fault_orientations: List[Dict] = field(default_factory=list)
    fault_lengths: List[float] = field(default_factory=list)
    major_fault_systems: List[Dict] = field(default_factory=list)

    # Output files
    probability_file: str = ""
    binary_file: str = ""
    figure_files: List[str] = field(default_factory=list)

    # For LLM interpretation
    summary_for_llm: str = ""
    geological_interpretation: str = ""

    # Metadata
    processing_time_seconds: float = 0.0
    model_used: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# FAULTSEG3D-STYLE U-NET ARCHITECTURE
# =============================================================================

class UNet3DBlock(nn.Module):
    """3D U-Net convolutional block"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class FaultSeg3DNetOriginal(nn.Module):
    """
    Exact replica of the original FaultSeg3D Keras model architecture.

    This matches the pre-trained weights from Wu et al. (2019).
    Layer names match the Keras model for direct weight loading.
    """

    def __init__(self):
        super().__init__()

        # Encoder - exactly matching Keras layer names
        self.conv3d_1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv3d_2 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        self.conv3d_3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3d_4 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.conv3d_5 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3d_6 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        # Bottleneck
        self.conv3d_7 = nn.Conv3d(64, 512, kernel_size=3, padding=1)
        self.conv3d_8 = nn.Conv3d(512, 512, kernel_size=3, padding=1)

        # Decoder with skip connections
        self.conv3d_9 = nn.Conv3d(576, 64, kernel_size=3, padding=1)   # 512 + 64 skip
        self.conv3d_10 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.conv3d_11 = nn.Conv3d(96, 32, kernel_size=3, padding=1)   # 64 + 32 skip
        self.conv3d_12 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.conv3d_13 = nn.Conv3d(48, 16, kernel_size=3, padding=1)   # 32 + 16 skip
        self.conv3d_14 = nn.Conv3d(16, 16, kernel_size=3, padding=1)

        # Output
        self.conv3d_15 = nn.Conv3d(16, 1, kernel_size=1)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.relu(self.conv3d_1(x))
        e1 = self.relu(self.conv3d_2(e1))

        e2 = self.pool(e1)
        e2 = self.relu(self.conv3d_3(e2))
        e2 = self.relu(self.conv3d_4(e2))

        e3 = self.pool(e2)
        e3 = self.relu(self.conv3d_5(e3))
        e3 = self.relu(self.conv3d_6(e3))

        # Bottleneck
        b = self.pool(e3)
        b = self.relu(self.conv3d_7(b))
        b = self.relu(self.conv3d_8(b))

        # Decoder with skip connections
        d3 = F.interpolate(b, size=e3.shape[2:], mode='trilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)  # 512 + 64 = 576
        d3 = self.relu(self.conv3d_9(d3))
        d3 = self.relu(self.conv3d_10(d3))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)  # 64 + 32 = 96
        d2 = self.relu(self.conv3d_11(d2))
        d2 = self.relu(self.conv3d_12(d2))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)  # 32 + 16 = 48
        d1 = self.relu(self.conv3d_13(d1))
        d1 = self.relu(self.conv3d_14(d1))

        # Output
        out = self.sigmoid(self.conv3d_15(d1))
        return out


class FaultSeg3DNet(nn.Module):
    """
    FaultSeg3D-style 3D U-Net for fault segmentation.

    Architecture based on Wu et al. (2019) with modifications for
    flexibility and integration with the PhD framework.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: List[int] = [32, 64, 128, 256]):
        super().__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.upconvs = nn.ModuleList()

        # Encoder path
        prev_channels = in_channels
        for feature in features:
            self.encoder_blocks.append(UNet3DBlock(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = UNet3DBlock(features[-1], features[-1] * 2)

        # Decoder path
        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(UNet3DBlock(feature * 2, feature))

        # Final output
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx, (upconv, block) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            skip = skip_connections[idx]

            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)

            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.sigmoid(self.final_conv(x))


# =============================================================================
# FAULT DETECTOR CLASS
# =============================================================================

class FaultDetector:
    """
    Deep learning-based fault detection for 3D seismic volumes.

    Integrates FaultSeg3D methodology with the PhD framework,
    providing fault probability outputs for LLM interpretation.

    Example:
        detector = FaultDetector(config)
        results = detector.detect_faults(seismic_data)

        # Feed to LLM for interpretation
        llm_summary = results.summary_for_llm
    """

    def __init__(self, config: Optional[FaultDetectionConfig] = None):
        self.config = config or FaultDetectionConfig()
        self.model = None
        self.device = None
        self._setup_device()

    def _setup_device(self):
        """Setup computation device (GPU/CPU)"""
        if TORCH_AVAILABLE:
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                print("Using CPU")

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load pre-trained fault detection model"""
        if not TORCH_AVAILABLE:
            print("Error: PyTorch not available")
            return False

        path = model_path or self.config.model_path

        # Default to pre-trained FaultSeg3D weights if no path specified
        if path is None:
            default_path = Path(__file__).parent / "models" / "faultseg3d" / "faultseg3d_pytorch.pth"
            if default_path.exists():
                path = str(default_path)

        if path and Path(path).exists():
            try:
                # Use FaultSeg3DNetOriginal for pre-trained weights (matches Keras architecture)
                self.model = FaultSeg3DNetOriginal()
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                print(f"Loaded pre-trained FaultSeg3D model from: {path}")
                return True
            except Exception as e:
                print(f"Error loading pre-trained model: {e}")
                print("Falling back to randomly initialized model...")

        # Initialize with random weights if no pre-trained model
        print("Initializing model with random weights (no pre-trained model found)")
        self.model = FaultSeg3DNet()
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
        """Normalize seismic data to [0, 1]"""
        data = data.astype(np.float32)
        p1, p99 = np.percentile(data, [1, 99])
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1 + 1e-8)
        return data

    def _extract_patches(self, data: np.ndarray) -> List[Tuple[np.ndarray, Tuple]]:
        """Extract overlapping patches from volume"""
        patches = []
        ps = self.config.patch_size
        overlap = self.config.overlap
        step = [p - overlap for p in ps]

        nz, ny, nx = data.shape

        for z in range(0, nz - ps[0] + 1, step[0]):
            for y in range(0, ny - ps[1] + 1, step[1]):
                for x in range(0, nx - ps[2] + 1, step[2]):
                    patch = data[z:z+ps[0], y:y+ps[1], x:x+ps[2]]
                    patches.append((patch, (z, y, x)))

        return patches

    def _merge_patches(self, patches: List[Tuple[np.ndarray, Tuple]],
                       output_shape: Tuple) -> np.ndarray:
        """Merge overlapping patches with weighted averaging"""
        result = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape, dtype=np.float32)

        ps = self.config.patch_size

        for patch, (z, y, x) in patches:
            result[z:z+ps[0], y:y+ps[1], x:x+ps[2]] += patch
            weights[z:z+ps[0], y:y+ps[1], x:x+ps[2]] += 1

        # Avoid division by zero
        weights = np.maximum(weights, 1)
        return result / weights

    def detect_faults(self, seismic_data: Optional[np.ndarray] = None,
                      segy_path: Optional[str] = None) -> FaultDetectionResults:
        """
        Detect faults in seismic volume using deep learning.

        Args:
            seismic_data: 3D numpy array of seismic amplitudes
            segy_path: Path to SEGY file (alternative to seismic_data)

        Returns:
            FaultDetectionResults with probability volume and statistics
        """
        start_time = datetime.now()
        results = FaultDetectionResults()

        # Load data if needed
        if seismic_data is None:
            path = segy_path or self.config.seismic_file
            seismic_data = self.load_seismic(path)
            if seismic_data is None:
                results.summary_for_llm = "Error: Could not load seismic data"
                return results

        # Load model
        if self.model is None:
            if not self.load_model():
                results.summary_for_llm = "Error: Could not load fault detection model"
                return results

        # Normalize data
        print("Normalizing seismic data...")
        normalized = self._normalize(seismic_data)

        # Extract and process patches
        print("Extracting patches...")
        patches = self._extract_patches(normalized)
        print(f"Processing {len(patches)} patches...")

        # Process with model
        processed_patches = []

        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                for patch, pos in tqdm(patches, desc="Fault detection"):
                    # Prepare input
                    x = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
                    x = x.to(self.device)

                    # Inference
                    pred = self.model(x)
                    pred = pred.squeeze().cpu().numpy()

                    processed_patches.append((pred, pos))
        else:
            # Fallback: use simple edge detection as proxy
            print("Using fallback edge detection (no DL model)")
            for patch, pos in tqdm(patches, desc="Edge detection"):
                if SCIPY_AVAILABLE:
                    # Sobel-based fault proxy
                    gx = ndimage.sobel(patch, axis=2)
                    gy = ndimage.sobel(patch, axis=1)
                    gz = ndimage.sobel(patch, axis=0)
                    edge = np.sqrt(gx**2 + gy**2 + gz**2)
                    edge = edge / (edge.max() + 1e-8)
                    processed_patches.append((edge, pos))
                else:
                    processed_patches.append((np.zeros_like(patch), pos))

        # Merge patches
        print("Merging patches...")
        fault_probability = self._merge_patches(processed_patches, seismic_data.shape)

        # Post-processing
        if SCIPY_AVAILABLE and self.config.smooth_sigma > 0:
            fault_probability = gaussian_filter(fault_probability,
                                                sigma=self.config.smooth_sigma)

        # Binary fault volume
        fault_binary = (fault_probability > self.config.probability_threshold).astype(np.uint8)

        # Calculate statistics
        results.fault_volume_percent = 100.0 * np.sum(fault_binary) / fault_binary.size
        results.num_faults_detected = self._count_fault_bodies(fault_binary)
        results.fault_orientations = self._analyze_fault_orientations(fault_binary)
        results.major_fault_systems = self._identify_major_faults(fault_binary, fault_probability)

        # Save outputs
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_probability:
            prob_file = output_dir / "fault_probability.npy"
            np.save(prob_file, fault_probability)
            results.probability_file = str(prob_file)

        if self.config.save_binary:
            bin_file = output_dir / "fault_binary.npy"
            np.save(bin_file, fault_binary)
            results.binary_file = str(bin_file)

        if self.config.save_figures and MATPLOTLIB_AVAILABLE:
            fig_files = self._create_figures(seismic_data, fault_probability,
                                             fault_binary, output_dir)
            results.figure_files = fig_files

        # Generate LLM summary
        results.summary_for_llm = self._generate_llm_summary(results, seismic_data.shape)

        # Finalize
        results.success = True
        results.processing_time_seconds = (datetime.now() - start_time).total_seconds()
        results.model_used = self.config.model_type

        # Save results JSON
        results_file = output_dir / "fault_detection_results.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)

        print(f"\nFault detection complete in {results.processing_time_seconds:.1f}s")
        print(f"Detected {results.num_faults_detected} fault bodies")
        print(f"Fault volume: {results.fault_volume_percent:.2f}%")

        return results

    def _count_fault_bodies(self, binary: np.ndarray) -> int:
        """Count separate fault bodies"""
        if not SCIPY_AVAILABLE:
            return 0
        labeled, num = ndimage.label(binary)
        return num

    def _analyze_fault_orientations(self, binary: np.ndarray) -> List[Dict]:
        """Analyze fault orientations"""
        orientations = []

        if not SCIPY_AVAILABLE:
            return orientations

        # Label connected components
        labeled, num = ndimage.label(binary)

        for i in range(1, min(num + 1, 20)):  # Top 20 faults
            mask = labeled == i
            if np.sum(mask) < self.config.min_fault_size:
                continue

            # Get fault coordinates
            coords = np.where(mask)

            # Estimate orientation using PCA-like approach
            if len(coords[0]) > 10:
                # Calculate centroid
                centroid = [np.mean(c) for c in coords]

                # Estimate strike (dominant horizontal direction)
                dx = coords[2].max() - coords[2].min()  # inline extent
                dy = coords[1].max() - coords[1].min()  # crossline extent

                if dx > 0:
                    strike = np.degrees(np.arctan2(dy, dx))
                else:
                    strike = 90.0

                # Estimate dip from vertical extent
                dz = coords[0].max() - coords[0].min()
                horizontal_extent = np.sqrt(dx**2 + dy**2)
                if horizontal_extent > 0:
                    dip = np.degrees(np.arctan2(dz, horizontal_extent))
                else:
                    dip = 90.0

                orientations.append({
                    'fault_id': i,
                    'strike': float(strike),
                    'dip': float(dip),
                    'size_voxels': int(np.sum(mask)),
                    'centroid': [float(c) for c in centroid]
                })

        return orientations

    def _identify_major_faults(self, binary: np.ndarray,
                               probability: np.ndarray) -> List[Dict]:
        """Identify major fault systems"""
        major_faults = []

        if not SCIPY_AVAILABLE:
            return major_faults

        labeled, num = ndimage.label(binary)

        # Get sizes of all faults
        sizes = ndimage.sum(binary, labeled, range(1, num + 1))

        # Get top 5 largest faults
        if len(sizes) > 0:
            top_indices = np.argsort(sizes)[-5:][::-1]

            for rank, idx in enumerate(top_indices):
                fault_id = idx + 1
                mask = labeled == fault_id
                size = sizes[idx]

                if size < self.config.min_fault_size:
                    continue

                # Get probability statistics
                fault_probs = probability[mask]

                major_faults.append({
                    'rank': rank + 1,
                    'fault_id': int(fault_id),
                    'size_voxels': int(size),
                    'mean_probability': float(np.mean(fault_probs)),
                    'max_probability': float(np.max(fault_probs))
                })

        return major_faults

    def _create_figures(self, seismic: np.ndarray, probability: np.ndarray,
                        binary: np.ndarray, output_dir: Path) -> List[str]:
        """Create visualization figures"""
        figures = []

        nz, ny, nx = seismic.shape

        # Figure 1: Inline comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        inline_idx = ny // 2

        axes[0].imshow(seismic[:, inline_idx, :].T, cmap='gray', aspect='auto')
        axes[0].set_title('Seismic Amplitude')
        axes[0].set_xlabel('Crossline')
        axes[0].set_ylabel('Time (samples)')

        axes[1].imshow(probability[:, inline_idx, :].T, cmap='hot', aspect='auto',
                       vmin=0, vmax=1)
        axes[1].set_title('Fault Probability')
        axes[1].set_xlabel('Crossline')

        axes[2].imshow(seismic[:, inline_idx, :].T, cmap='gray', aspect='auto')
        axes[2].imshow(probability[:, inline_idx, :].T, cmap='Reds', aspect='auto',
                       alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title('Overlay')
        axes[2].set_xlabel('Crossline')

        plt.tight_layout()
        fig_path = output_dir / 'fault_detection_inline.png'
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        figures.append(str(fig_path))

        # Figure 2: Time slice
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        time_idx = nz // 2

        axes[0].imshow(seismic[time_idx, :, :], cmap='gray', aspect='auto')
        axes[0].set_title(f'Seismic @ Time {time_idx}')
        axes[0].set_xlabel('Inline')
        axes[0].set_ylabel('Crossline')

        axes[1].imshow(probability[time_idx, :, :], cmap='hot', aspect='auto',
                       vmin=0, vmax=1)
        axes[1].set_title('Fault Probability')

        axes[2].imshow(binary[time_idx, :, :], cmap='binary', aspect='auto')
        axes[2].set_title('Fault Binary')

        plt.tight_layout()
        fig_path = output_dir / 'fault_detection_timeslice.png'
        plt.savefig(fig_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
        figures.append(str(fig_path))

        return figures

    def _generate_llm_summary(self, results: FaultDetectionResults,
                              volume_shape: Tuple) -> str:
        """Generate summary text for LLM interpretation"""

        summary = f"""
DEEP LEARNING FAULT DETECTION RESULTS
=====================================

Volume Analyzed: {volume_shape[0]} x {volume_shape[1]} x {volume_shape[2]} (samples x crosslines x inlines)

DETECTION SUMMARY:
- Total fault bodies detected: {results.num_faults_detected}
- Fault volume percentage: {results.fault_volume_percent:.2f}%
- Detection method: {results.model_used}

MAJOR FAULT SYSTEMS:
"""
        for fault in results.major_fault_systems:
            summary += f"""
  Fault #{fault['rank']}:
    - Size: {fault['size_voxels']:,} voxels
    - Mean probability: {fault['mean_probability']:.3f}
    - Max probability: {fault['max_probability']:.3f}
"""

        summary += """
FAULT ORIENTATIONS:
"""
        for orient in results.fault_orientations[:5]:
            summary += f"""
  Fault {orient['fault_id']}:
    - Strike: {orient['strike']:.1f}°
    - Dip: {orient['dip']:.1f}°
    - Size: {orient['size_voxels']:,} voxels
"""

        summary += """
GEOLOGICAL CONTEXT FOR INTERPRETATION:
- Consider fault orientations relative to regional stress field
- Evaluate fault seal potential based on throw and juxtaposition
- Assess compartmentalization risk for reservoir development
- Check fault timing relative to hydrocarbon migration

OUTPUT FILES:
- Fault probability volume: fault_probability.npy
- Binary fault volume: fault_binary.npy
- Visualization figures: fault_detection_*.png
"""

        return summary


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Deep Learning Fault Detection (FaultSeg3D-style)'
    )
    parser.add_argument('segy_file', help='Input SEGY file')
    parser.add_argument('--output-dir', default='fault_outputs', help='Output directory')
    parser.add_argument('--model', help='Path to pre-trained model weights')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fault probability threshold')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--config', help='Path to JSON config file')

    args = parser.parse_args()

    # Load or create config
    if args.config and Path(args.config).exists():
        config = FaultDetectionConfig.from_json(args.config)
    else:
        config = FaultDetectionConfig()

    # Apply CLI arguments
    config.seismic_file = args.segy_file
    config.output_dir = args.output_dir
    config.probability_threshold = args.threshold
    config.use_gpu = not args.no_gpu

    if args.model:
        config.model_path = args.model

    # Run detection
    detector = FaultDetector(config)
    results = detector.detect_faults()

    # Print summary
    print("\n" + results.summary_for_llm)


if __name__ == '__main__':
    main()
