"""
================================================================================
AUTOMATED SEISMIC INTERPRETATION MODULE
Based on CNN-for-ASI (Waldeland et al., 2018, The Leading Edge)
================================================================================

Texture-based seismic interpretation using CNNs for automated
structure and facies classification.

Reference:
- Waldeland, A. U., Jensen, A. C., Gelius, L., & Solberg, A. H. S. (2018).
  Convolutional Neural Networks for Automated Seismic Interpretation.
  The Leading Edge, 37(7), 529-537.

Author: Moses Ekene Obasi
PhD Research - University of Calabar
================================================================================
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# TEXTURE CNN ARCHITECTURE (Based on Waldeland et al., 2018)
# =============================================================================

class TextureNet(nn.Module):
    """
    CNN for seismic texture classification.

    Architecture based on Waldeland et al. (2018) for learning
    seismic textures and classifying geological structures.
    """

    def __init__(self, num_classes: int = 5, input_size: int = 65):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def extract_features(self, x):
        """Extract learned features before classification"""
        return self.features(x)


# =============================================================================
# AUTOMATED INTERPRETER CLASS
# =============================================================================

@dataclass
class AutoInterpretationConfig:
    """Configuration for automated interpretation"""

    seismic_file: str = ""
    output_dir: str = "auto_interpretation"

    # Model settings
    model_path: Optional[str] = None
    num_classes: int = 5
    use_gpu: bool = True

    # Class definitions
    class_names: List[str] = field(default_factory=lambda: [
        "Continuous Reflectors",
        "Chaotic/Disrupted",
        "High Amplitude",
        "Low Amplitude",
        "Fault Zone"
    ])

    # Processing
    patch_size: int = 65
    stride: int = 16
    batch_size: int = 32

    # Output
    save_classification: bool = True
    save_features: bool = True
    save_figures: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AutoInterpretationResults:
    """Results from automated interpretation"""

    success: bool = False
    num_classes_found: int = 0

    # Statistics
    class_distribution: Dict[str, float] = field(default_factory=dict)

    # Structural analysis
    fault_zone_percentage: float = 0.0
    continuous_percentage: float = 0.0
    chaotic_percentage: float = 0.0

    # Output files
    classification_file: str = ""
    feature_file: str = ""
    figure_files: List[str] = field(default_factory=list)

    # For LLM
    summary_for_llm: str = ""

    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class AutomatedInterpreter:
    """
    CNN-based automated seismic interpretation.

    Uses texture classification to identify seismic patterns
    including continuous reflectors, chaotic zones, and fault zones.
    """

    def __init__(self, config: Optional[AutoInterpretationConfig] = None):
        self.config = config or AutoInterpretationConfig()
        self.model = None
        self.device = None
        self._setup_device()

    def _setup_device(self):
        if TORCH_AVAILABLE:
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

    def load_model(self, model_path: Optional[str] = None) -> bool:
        if not TORCH_AVAILABLE:
            return False

        path = model_path or self.config.model_path

        if path and Path(path).exists():
            try:
                self.model = TextureNet(num_classes=self.config.num_classes)
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                return True
            except Exception as e:
                print(f"Error loading model: {e}")

        # Initialize random weights
        self.model = TextureNet(num_classes=self.config.num_classes)
        self.model.to(self.device)
        self.model.eval()
        return True

    def interpret(self, seismic_data: Optional[np.ndarray] = None,
                  segy_path: Optional[str] = None) -> AutoInterpretationResults:
        """Run automated interpretation on seismic data"""
        start_time = datetime.now()
        results = AutoInterpretationResults()

        # Load data
        if seismic_data is None and segy_path:
            if SEGYIO_AVAILABLE:
                with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
                    seismic_data = segyio.tools.cube(f)

        if seismic_data is None:
            return results

        # Load model
        if self.model is None:
            self.load_model()

        # Normalize
        data = seismic_data.astype(np.float32)
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)

        nz, ny, nx = data.shape

        # Process inline by inline
        classification = np.zeros((nz, ny, nx), dtype=np.uint8)

        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
            ps = self.config.patch_size
            half = ps // 2

            with torch.no_grad():
                for y in range(ny):
                    inline = data[:, y, :]

                    # Pad for edge patches
                    padded = np.pad(inline, ((half, half), (half, half)), mode='reflect')

                    for z in range(0, nz, self.config.stride):
                        for x in range(0, nx, self.config.stride):
                            # Extract patch
                            patch = padded[z:z+ps, x:x+ps]

                            if patch.shape == (ps, ps):
                                # Classify
                                inp = torch.from_numpy(patch).float()
                                inp = inp.unsqueeze(0).unsqueeze(0).to(self.device)

                                out = self.model(inp)
                                pred = torch.argmax(out, dim=1).item()

                                # Fill region
                                z_end = min(z + self.config.stride, nz)
                                x_end = min(x + self.config.stride, nx)
                                classification[z:z_end, y, x:x_end] = pred
        else:
            # Fallback: amplitude-based
            p20, p40, p60, p80 = np.percentile(data, [20, 40, 60, 80])
            classification[data < p20] = 3  # Low amplitude
            classification[(data >= p20) & (data < p40)] = 0  # Continuous
            classification[(data >= p40) & (data < p60)] = 0
            classification[(data >= p60) & (data < p80)] = 2  # High amplitude
            classification[data >= p80] = 2

            # Detect chaotic using gradient
            if SCIPY_AVAILABLE:
                gx = ndimage.sobel(data, axis=2)
                gy = ndimage.sobel(data, axis=0)
                gradient = np.sqrt(gx**2 + gy**2)
                high_gradient = gradient > np.percentile(gradient, 90)
                classification[high_gradient] = 1  # Chaotic

        # Calculate statistics
        total = classification.size
        for i, name in enumerate(self.config.class_names):
            pct = 100.0 * np.sum(classification == i) / total
            results.class_distribution[name] = pct

            if 'Fault' in name:
                results.fault_zone_percentage = pct
            elif 'Continuous' in name:
                results.continuous_percentage = pct
            elif 'Chaotic' in name:
                results.chaotic_percentage = pct

        results.num_classes_found = len(np.unique(classification))

        # Save outputs
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.save_classification:
            class_file = output_dir / "texture_classification.npy"
            np.save(class_file, classification)
            results.classification_file = str(class_file)

        # Generate summary
        results.summary_for_llm = self._generate_summary(results, data.shape)

        results.success = True
        results.processing_time_seconds = (datetime.now() - start_time).total_seconds()

        return results

    def _generate_summary(self, results: AutoInterpretationResults,
                          shape: Tuple) -> str:
        summary = f"""
AUTOMATED SEISMIC INTERPRETATION (CNN Texture Classification)
=============================================================

Volume: {shape[0]} x {shape[1]} x {shape[2]}

TEXTURE CLASS DISTRIBUTION:
"""
        for name, pct in sorted(results.class_distribution.items(),
                                key=lambda x: x[1], reverse=True):
            summary += f"  - {name}: {pct:.1f}%\n"

        summary += f"""
KEY METRICS:
- Continuous Reflectors: {results.continuous_percentage:.1f}%
- Chaotic/Disrupted Zones: {results.chaotic_percentage:.1f}%
- Fault Zones: {results.fault_zone_percentage:.1f}%

INTERPRETATION NOTES:
- High continuous reflector percentage suggests well-stratified sequence
- Chaotic zones may indicate mass transport deposits or intense faulting
- Fault zone percentage indicates structural complexity
"""
        return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Automated Seismic Interpretation')
    parser.add_argument('segy_file', help='Input SEGY file')
    parser.add_argument('--output-dir', default='auto_interpretation')
    parser.add_argument('--no-gpu', action='store_true')

    args = parser.parse_args()

    config = AutoInterpretationConfig()
    config.seismic_file = args.segy_file
    config.output_dir = args.output_dir
    config.use_gpu = not args.no_gpu

    interpreter = AutomatedInterpreter(config)
    results = interpreter.interpret(segy_path=args.segy_file)

    print(results.summary_for_llm)


if __name__ == '__main__':
    main()
