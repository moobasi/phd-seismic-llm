# PhD Workflow Optimization Plan

## IMPLEMENTATION STATUS: COMPLETED

### Seismic Viewer Optimizations
All three phases have been implemented:
- [x] Phase 1: Fix redundant data loading
- [x] Phase 2: Create unified fault detection system
- [x] Phase 3: Add 3D visualization capabilities

### AI Assistant Redesign
Complete redesign implemented:
- [x] Created SeismicAgent framework with tool-calling capabilities (`seismic_agent.py`)
- [x] Redesigned AI Assistant from 8 tabs to 3 tabs (`seismic_ai_assistant_v3.py`)
  - Tab 1: Chat & Agent (primary interface with inline images, quick actions)
  - Tab 2: Visual Analysis (gallery of generated outputs)
  - Tab 3: Reports (generate and export professional reports)
- [x] Added progress panel to main workflow GUI (`phd_workflow_gui.py`)
  - Real-time step status display
  - Progress bar with percentage
  - Live output streaming with colored logs

## Overview
This plan addresses three key issues:
1. Redundant data loading between main GUI and seismic viewer
2. Two separate fault detection systems needing unification
3. 2D-only display for 3D data

---

## PHASE 1: Fix Redundant Data Loading

### Problem
- `phd_workflow_gui.py` asks for data paths in Project Configuration
- `seismic_viewer.py` ignores these and re-asks with file dialogs
- User enters paths twice

### Solution
Pass ProjectConfig to SeismicViewer on launch and auto-load configured data.

### Implementation

#### 1.1 Modify Viewer Launch in `phd_workflow_gui.py`

**File:** `phd_workflow_gui.py`
**Location:** Lines 866-868 (where seismic viewer is launched)

**Current Code:**
```python
subprocess.Popen([sys.executable, "seismic_viewer.py"])
```

**New Code:**
```python
# Pass config paths as command line arguments
config = get_config()
args = [sys.executable, "seismic_viewer.py"]
if config.seismic_3d_path:
    args.extend(["--segy-3d", config.seismic_3d_path])
if config.seismic_2d_directory:
    args.extend(["--segy-2d-dir", config.seismic_2d_directory])
subprocess.Popen(args)
```

#### 1.2 Modify Seismic Viewer to Accept Arguments

**File:** `seismic_viewer.py`
**Location:** Main entry point (end of file)

**Add argument parsing:**
```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Seismic Viewer")
    parser.add_argument("--segy-3d", type=str, help="Path to 3D SEGY file")
    parser.add_argument("--segy-2d-dir", type=str, help="Directory containing 2D SEGY files")
    parser.add_argument("--auto-load", action="store_true", default=True,
                        help="Auto-load data on startup")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = SeismicViewerGUI(root,
                           initial_3d_path=args.segy_3d,
                           initial_2d_dir=args.segy_2d_dir,
                           auto_load=args.auto_load)
    root.mainloop()
```

#### 1.3 Add Auto-Load in SeismicViewerGUI.__init__

**File:** `seismic_viewer.py`
**Location:** `SeismicViewerGUI.__init__` method

**Add to constructor:**
```python
def __init__(self, root, initial_3d_path=None, initial_2d_dir=None, auto_load=True):
    # ... existing init code ...

    # Auto-load data if paths provided
    if auto_load:
        if initial_3d_path and os.path.exists(initial_3d_path):
            self._auto_load_3d(initial_3d_path)
        if initial_2d_dir and os.path.isdir(initial_2d_dir):
            self._auto_load_2d_directory(initial_2d_dir)

def _auto_load_3d(self, filepath):
    """Auto-load 3D volume from configured path."""
    try:
        self.loader.load_3d_volume(filepath)
        self._update_sliders_from_geometry()
        self._display_section()
        self._update_status(f"Loaded: {os.path.basename(filepath)}")
    except Exception as e:
        self._update_status(f"Auto-load failed: {e}")

def _auto_load_2d_directory(self, directory):
    """Auto-load 2D lines from configured directory."""
    segy_files = glob.glob(os.path.join(directory, "*.sgy")) + \
                 glob.glob(os.path.join(directory, "*.segy"))
    for f in segy_files[:10]:  # Load first 10
        try:
            self.loader.load_2d_line(f)
        except:
            pass
    if segy_files:
        self._update_status(f"Loaded {len(segy_files)} 2D lines")
```

---

## PHASE 2: Unify Fault Detection Systems

### Problem
- **Deep Learning** (`dl_fault_detection.py`): FaultSeg3D CNN, outputs probability volume
- **Classical** (`real_interpretation.py`): Variance + gradient, outputs binary mask per horizon
- Different output formats, not interoperable
- No clear way to compare or combine results

### Solution
Create a unified fault detection interface with:
1. Common output format (UnifiedFaultResult)
2. Method selection (DL primary, classical for validation)
3. Confidence scoring
4. Single integration point for the viewer

### Implementation

#### 2.1 Create Unified Fault Data Model

**New File:** `fault_detection_unified.py`

```python
"""
Unified Fault Detection Interface
Combines Deep Learning and Classical approaches with consistent output format.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import numpy as np
import json
import os

class FaultDetectionMethod(Enum):
    DEEP_LEARNING = "deep_learning"      # FaultSeg3D
    CLASSICAL_VARIANCE = "classical_variance"
    CLASSICAL_GRADIENT = "classical_gradient"
    ENSEMBLE = "ensemble"                 # Combined methods


@dataclass
class FaultSegment:
    """Individual fault segment with unified properties."""
    fault_id: int
    method: FaultDetectionMethod

    # Spatial extent
    inline_range: Tuple[int, int]         # (min_il, max_il)
    crossline_range: Tuple[int, int]      # (min_xl, max_xl)
    time_range: Tuple[float, float]       # (min_twt, max_twt) in ms

    # Geometry
    strike_azimuth: float                 # Degrees from north (0-360)
    dip_angle: float                      # Degrees from horizontal (0-90)
    throw_estimate: Optional[float]       # Vertical displacement in ms
    length_km: Optional[float]            # Fault trace length

    # Confidence metrics
    confidence: float                     # 0-1, unified confidence score
    dl_probability: Optional[float]       # Deep learning probability (if available)
    classical_score: Optional[float]      # Classical attribute score (if available)

    # Trace data for visualization
    trace_points: List[Dict]              # [{"il": x, "xl": y, "twt": z}, ...]

    # 3D volume indices (for volumetric rendering)
    voxel_indices: Optional[np.ndarray]   # Shape (N, 3) array of [il, xl, sample] indices

    # Metadata
    interpreter_notes: str = ""
    validated: bool = False
    validation_well: Optional[str] = None


@dataclass
class UnifiedFaultResult:
    """Container for all fault detection results."""
    # Identification
    survey_name: str
    detection_timestamp: str
    methods_used: List[FaultDetectionMethod]

    # Volume info
    volume_shape: Tuple[int, int, int]    # (n_inlines, n_crosslines, n_samples)
    inline_range: Tuple[int, int]
    crossline_range: Tuple[int, int]
    sample_rate_ms: float

    # Fault data
    faults: List[FaultSegment] = field(default_factory=list)

    # Full probability volume (from DL)
    probability_volume_path: Optional[str] = None

    # Statistics
    total_faults: int = 0
    major_faults: int = 0                 # Faults with length > threshold
    fault_volume_percent: float = 0.0

    # Confidence summary
    high_confidence_count: int = 0        # confidence > 0.8
    medium_confidence_count: int = 0      # 0.5 < confidence <= 0.8
    low_confidence_count: int = 0         # confidence <= 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "survey_name": self.survey_name,
            "detection_timestamp": self.detection_timestamp,
            "methods_used": [m.value for m in self.methods_used],
            "volume_shape": list(self.volume_shape),
            "inline_range": list(self.inline_range),
            "crossline_range": list(self.crossline_range),
            "sample_rate_ms": self.sample_rate_ms,
            "faults": [self._fault_to_dict(f) for f in self.faults],
            "probability_volume_path": self.probability_volume_path,
            "statistics": {
                "total_faults": self.total_faults,
                "major_faults": self.major_faults,
                "fault_volume_percent": self.fault_volume_percent,
                "high_confidence_count": self.high_confidence_count,
                "medium_confidence_count": self.medium_confidence_count,
                "low_confidence_count": self.low_confidence_count
            }
        }

    def _fault_to_dict(self, fault: FaultSegment) -> Dict:
        return {
            "fault_id": fault.fault_id,
            "method": fault.method.value,
            "inline_range": list(fault.inline_range),
            "crossline_range": list(fault.crossline_range),
            "time_range": list(fault.time_range),
            "strike_azimuth": fault.strike_azimuth,
            "dip_angle": fault.dip_angle,
            "throw_estimate": fault.throw_estimate,
            "length_km": fault.length_km,
            "confidence": fault.confidence,
            "dl_probability": fault.dl_probability,
            "classical_score": fault.classical_score,
            "trace_points": fault.trace_points,
            "validated": fault.validated,
            "validation_well": fault.validation_well
        }

    def save(self, filepath: str):
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'UnifiedFaultResult':
        """Load results from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Reconstruct object from dict
        result = cls(
            survey_name=data["survey_name"],
            detection_timestamp=data["detection_timestamp"],
            methods_used=[FaultDetectionMethod(m) for m in data["methods_used"]],
            volume_shape=tuple(data["volume_shape"]),
            inline_range=tuple(data["inline_range"]),
            crossline_range=tuple(data["crossline_range"]),
            sample_rate_ms=data["sample_rate_ms"],
            probability_volume_path=data.get("probability_volume_path")
        )
        # Load faults
        for fd in data.get("faults", []):
            fault = FaultSegment(
                fault_id=fd["fault_id"],
                method=FaultDetectionMethod(fd["method"]),
                inline_range=tuple(fd["inline_range"]),
                crossline_range=tuple(fd["crossline_range"]),
                time_range=tuple(fd["time_range"]),
                strike_azimuth=fd["strike_azimuth"],
                dip_angle=fd["dip_angle"],
                throw_estimate=fd.get("throw_estimate"),
                length_km=fd.get("length_km"),
                confidence=fd["confidence"],
                dl_probability=fd.get("dl_probability"),
                classical_score=fd.get("classical_score"),
                trace_points=fd.get("trace_points", []),
                voxel_indices=None,  # Not serialized
                validated=fd.get("validated", False),
                validation_well=fd.get("validation_well")
            )
            result.faults.append(fault)

        # Load stats
        stats = data.get("statistics", {})
        result.total_faults = stats.get("total_faults", len(result.faults))
        result.major_faults = stats.get("major_faults", 0)
        result.fault_volume_percent = stats.get("fault_volume_percent", 0.0)
        result.high_confidence_count = stats.get("high_confidence_count", 0)
        result.medium_confidence_count = stats.get("medium_confidence_count", 0)
        result.low_confidence_count = stats.get("low_confidence_count", 0)

        return result


class UnifiedFaultDetector:
    """
    Unified interface for fault detection.
    Combines Deep Learning (primary) with Classical methods (validation).
    """

    def __init__(self, seismic_data: np.ndarray, geometry: Dict):
        """
        Args:
            seismic_data: 3D numpy array (inline, crossline, samples)
            geometry: Dict with keys: il_min, il_max, xl_min, xl_max, sample_rate_ms
        """
        self.seismic_data = seismic_data
        self.geometry = geometry
        self.results: Optional[UnifiedFaultResult] = None

        # Method availability
        self._dl_available = self._check_dl_available()

    def _check_dl_available(self) -> bool:
        """Check if deep learning dependencies are available."""
        try:
            import torch
            from deep_learning.dl_fault_detection import FaultDetector
            return True
        except ImportError:
            return False

    def detect_faults(self,
                      method: FaultDetectionMethod = FaultDetectionMethod.DEEP_LEARNING,
                      probability_threshold: float = 0.5,
                      min_fault_size: int = 100,
                      use_ensemble: bool = False) -> UnifiedFaultResult:
        """
        Run fault detection with specified method.

        Args:
            method: Detection method to use
            probability_threshold: Threshold for DL probability (0-1)
            min_fault_size: Minimum fault size in voxels
            use_ensemble: If True, run both DL and classical, combine results

        Returns:
            UnifiedFaultResult with detected faults
        """
        from datetime import datetime

        methods_used = []
        all_faults = []
        probability_volume_path = None

        # Run Deep Learning detection
        if method in [FaultDetectionMethod.DEEP_LEARNING, FaultDetectionMethod.ENSEMBLE]:
            if self._dl_available:
                dl_faults, prob_path = self._run_deep_learning(
                    probability_threshold, min_fault_size
                )
                all_faults.extend(dl_faults)
                methods_used.append(FaultDetectionMethod.DEEP_LEARNING)
                probability_volume_path = prob_path
            else:
                print("Warning: Deep learning not available, falling back to classical")
                method = FaultDetectionMethod.CLASSICAL_VARIANCE

        # Run Classical detection (for validation or primary)
        if method in [FaultDetectionMethod.CLASSICAL_VARIANCE,
                      FaultDetectionMethod.CLASSICAL_GRADIENT,
                      FaultDetectionMethod.ENSEMBLE]:
            classical_faults = self._run_classical(method, probability_threshold)
            all_faults.extend(classical_faults)
            if FaultDetectionMethod.CLASSICAL_VARIANCE not in methods_used:
                methods_used.append(FaultDetectionMethod.CLASSICAL_VARIANCE)

        # If ensemble, merge overlapping faults
        if use_ensemble and len(all_faults) > 0:
            all_faults = self._merge_ensemble_faults(all_faults)

        # Build result
        self.results = UnifiedFaultResult(
            survey_name=self.geometry.get("survey_name", "Unknown"),
            detection_timestamp=datetime.now().isoformat(),
            methods_used=methods_used,
            volume_shape=self.seismic_data.shape,
            inline_range=(self.geometry["il_min"], self.geometry["il_max"]),
            crossline_range=(self.geometry["xl_min"], self.geometry["xl_max"]),
            sample_rate_ms=self.geometry.get("sample_rate_ms", 4.0),
            faults=all_faults,
            probability_volume_path=probability_volume_path
        )

        # Calculate statistics
        self._calculate_statistics()

        return self.results

    def _run_deep_learning(self, threshold: float, min_size: int):
        """Run FaultSeg3D deep learning detection."""
        from deep_learning.dl_fault_detection import FaultDetector, FaultDetectionConfig

        config = FaultDetectionConfig(
            probability_threshold=threshold,
            min_fault_size=min_size
        )
        detector = FaultDetector(config)
        dl_results = detector.detect_faults(seismic_data=self.seismic_data)

        # Convert DL results to unified format
        faults = []
        for i, fault_data in enumerate(dl_results.fault_orientations):
            fault = FaultSegment(
                fault_id=i,
                method=FaultDetectionMethod.DEEP_LEARNING,
                inline_range=self._extract_il_range(fault_data),
                crossline_range=self._extract_xl_range(fault_data),
                time_range=self._extract_time_range(fault_data),
                strike_azimuth=fault_data.get("strike", 0.0),
                dip_angle=fault_data.get("dip", 90.0),
                throw_estimate=None,
                length_km=None,
                confidence=fault_data.get("mean_probability", 0.5),
                dl_probability=fault_data.get("mean_probability"),
                classical_score=None,
                trace_points=self._extract_trace_points(fault_data)
            )
            faults.append(fault)

        return faults, dl_results.probability_file

    def _run_classical(self, method: FaultDetectionMethod, threshold: float):
        """Run classical variance/gradient detection."""
        from scipy import ndimage

        faults = []

        # Calculate variance attribute on time slices
        n_il, n_xl, n_samples = self.seismic_data.shape
        variance_volume = np.zeros_like(self.seismic_data)

        for t in range(n_samples):
            slice_data = self.seismic_data[:, :, t]
            # Local variance with 3x3 window
            mean_local = ndimage.uniform_filter(slice_data, size=3)
            sqr_mean = ndimage.uniform_filter(slice_data**2, size=3)
            variance_volume[:, :, t] = sqr_mean - mean_local**2

        # Normalize variance
        variance_volume = (variance_volume - variance_volume.min()) / \
                         (variance_volume.max() - variance_volume.min() + 1e-10)

        # Threshold to get fault candidates
        fault_mask = variance_volume > threshold

        # Label connected components
        labeled, n_features = ndimage.label(fault_mask)

        for fault_id in range(1, min(n_features + 1, 51)):  # Top 50
            coords = np.where(labeled == fault_id)
            if len(coords[0]) < 100:  # Skip small features
                continue

            fault = FaultSegment(
                fault_id=fault_id + 1000,  # Offset to distinguish from DL
                method=FaultDetectionMethod.CLASSICAL_VARIANCE,
                inline_range=(coords[0].min(), coords[0].max()),
                crossline_range=(coords[1].min(), coords[1].max()),
                time_range=(coords[2].min() * self.geometry.get("sample_rate_ms", 4.0),
                           coords[2].max() * self.geometry.get("sample_rate_ms", 4.0)),
                strike_azimuth=self._estimate_strike(coords),
                dip_angle=self._estimate_dip(coords),
                throw_estimate=None,
                length_km=None,
                confidence=0.6,  # Lower confidence for classical
                dl_probability=None,
                classical_score=float(variance_volume[coords].mean()),
                trace_points=self._coords_to_trace(coords)
            )
            faults.append(fault)

        return faults

    def _merge_ensemble_faults(self, faults: List[FaultSegment]) -> List[FaultSegment]:
        """Merge overlapping faults from different methods."""
        # Simple spatial overlap check
        merged = []
        used = set()

        for i, f1 in enumerate(faults):
            if i in used:
                continue

            # Find overlapping faults
            overlapping = [f1]
            for j, f2 in enumerate(faults[i+1:], start=i+1):
                if j in used:
                    continue
                if self._faults_overlap(f1, f2):
                    overlapping.append(f2)
                    used.add(j)

            # Merge overlapping faults
            if len(overlapping) > 1:
                merged_fault = self._merge_fault_group(overlapping)
                merged.append(merged_fault)
            else:
                merged.append(f1)
            used.add(i)

        return merged

    def _faults_overlap(self, f1: FaultSegment, f2: FaultSegment) -> bool:
        """Check if two faults spatially overlap."""
        # Check inline overlap
        il_overlap = not (f1.inline_range[1] < f2.inline_range[0] or
                         f2.inline_range[1] < f1.inline_range[0])
        # Check crossline overlap
        xl_overlap = not (f1.crossline_range[1] < f2.crossline_range[0] or
                         f2.crossline_range[1] < f1.crossline_range[0])
        return il_overlap and xl_overlap

    def _merge_fault_group(self, faults: List[FaultSegment]) -> FaultSegment:
        """Merge a group of overlapping faults into one."""
        # Use highest confidence fault as base
        base = max(faults, key=lambda f: f.confidence)

        # Expand ranges
        il_min = min(f.inline_range[0] for f in faults)
        il_max = max(f.inline_range[1] for f in faults)
        xl_min = min(f.crossline_range[0] for f in faults)
        xl_max = max(f.crossline_range[1] for f in faults)
        t_min = min(f.time_range[0] for f in faults)
        t_max = max(f.time_range[1] for f in faults)

        # Average confidence (boost for ensemble agreement)
        avg_conf = sum(f.confidence for f in faults) / len(faults)
        boosted_conf = min(1.0, avg_conf * 1.2)  # 20% boost for agreement

        return FaultSegment(
            fault_id=base.fault_id,
            method=FaultDetectionMethod.ENSEMBLE,
            inline_range=(il_min, il_max),
            crossline_range=(xl_min, xl_max),
            time_range=(t_min, t_max),
            strike_azimuth=base.strike_azimuth,
            dip_angle=base.dip_angle,
            throw_estimate=base.throw_estimate,
            length_km=base.length_km,
            confidence=boosted_conf,
            dl_probability=next((f.dl_probability for f in faults if f.dl_probability), None),
            classical_score=next((f.classical_score for f in faults if f.classical_score), None),
            trace_points=base.trace_points
        )

    def _calculate_statistics(self):
        """Calculate summary statistics for results."""
        if not self.results:
            return

        self.results.total_faults = len(self.results.faults)
        self.results.high_confidence_count = sum(
            1 for f in self.results.faults if f.confidence > 0.8
        )
        self.results.medium_confidence_count = sum(
            1 for f in self.results.faults if 0.5 < f.confidence <= 0.8
        )
        self.results.low_confidence_count = sum(
            1 for f in self.results.faults if f.confidence <= 0.5
        )

    # Helper methods for coordinate extraction
    def _extract_il_range(self, data: Dict) -> Tuple[int, int]:
        centroid = data.get("centroid", [0, 0, 0])
        size = data.get("size_voxels", 100)
        extent = int(np.sqrt(size) / 2)
        return (int(centroid[0]) - extent, int(centroid[0]) + extent)

    def _extract_xl_range(self, data: Dict) -> Tuple[int, int]:
        centroid = data.get("centroid", [0, 0, 0])
        size = data.get("size_voxels", 100)
        extent = int(np.sqrt(size) / 2)
        return (int(centroid[1]) - extent, int(centroid[1]) + extent)

    def _extract_time_range(self, data: Dict) -> Tuple[float, float]:
        centroid = data.get("centroid", [0, 0, 0])
        sr = self.geometry.get("sample_rate_ms", 4.0)
        return (centroid[2] * sr - 50, centroid[2] * sr + 50)

    def _extract_trace_points(self, data: Dict) -> List[Dict]:
        centroid = data.get("centroid", [0, 0, 0])
        return [{"il": int(centroid[0]), "xl": int(centroid[1]), "twt": centroid[2]}]

    def _estimate_strike(self, coords) -> float:
        if len(coords[0]) < 2:
            return 0.0
        dx = coords[0].max() - coords[0].min()
        dy = coords[1].max() - coords[1].min()
        return np.degrees(np.arctan2(dy, dx))

    def _estimate_dip(self, coords) -> float:
        if len(coords[0]) < 2:
            return 90.0
        horiz = np.sqrt((coords[0].max() - coords[0].min())**2 +
                       (coords[1].max() - coords[1].min())**2)
        vert = coords[2].max() - coords[2].min()
        if horiz == 0:
            return 90.0
        return np.degrees(np.arctan2(vert, horiz))

    def _coords_to_trace(self, coords) -> List[Dict]:
        # Sample every 10th point
        points = []
        for i in range(0, len(coords[0]), 10):
            points.append({
                "il": int(coords[0][i]),
                "xl": int(coords[1][i]),
                "twt": float(coords[2][i] * self.geometry.get("sample_rate_ms", 4.0))
            })
        return points[:100]  # Limit to 100 points
```

#### 2.2 Update Seismic Viewer to Use Unified Format

**File:** `seismic_viewer.py`
**Location:** `_load_fault_data()` method

**Updated to load unified format:**
```python
def _load_fault_data(self):
    """Load fault data from unified fault detection results."""
    self.faults = []

    # Try unified format first
    unified_path = os.path.join("interpretation", "real_outputs", "unified_faults.json")
    if os.path.exists(unified_path):
        try:
            from fault_detection_unified import UnifiedFaultResult
            results = UnifiedFaultResult.load(unified_path)
            self.faults = results.faults
            self._fault_result = results
            return
        except Exception as e:
            print(f"Failed to load unified faults: {e}")

    # Fall back to legacy format
    legacy_path = os.path.join("interpretation", "real_outputs", "interpretation_results.json")
    if os.path.exists(legacy_path):
        # ... existing legacy loading code ...
```

---

## PHASE 3: Add 3D Visualization

### Problem
- Current viewer shows only 2D slices (inline, crossline, timeslice)
- No true volumetric 3D rendering
- No overlay of 2D and 3D data

### Solution
Add optional 3D visualization using PyVista/VTK with fallback to matplotlib 3D.

### Implementation

#### 3.1 Create 3D Viewer Module

**New File:** `viewer_3d.py`

```python
"""
3D Seismic Visualization Module
Provides volumetric rendering using PyVista (VTK) with matplotlib fallback.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, List, Tuple
import os

# Check for 3D libraries
try:
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MPL_3D_AVAILABLE = True
except ImportError:
    MPL_3D_AVAILABLE = False


class Seismic3DViewer:
    """
    3D visualization for seismic data and fault surfaces.
    Uses PyVista if available, falls back to matplotlib 3D.
    """

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.seismic_data: Optional[np.ndarray] = None
        self.fault_data: Optional[np.ndarray] = None
        self.geometry: Dict = {}

        # Display settings
        self.opacity = 0.3
        self.colormap = "seismic"
        self.show_faults = True
        self.show_seismic = True

        # Initialize viewer
        if PYVISTA_AVAILABLE:
            self._init_pyvista()
        elif MPL_3D_AVAILABLE:
            self._init_matplotlib_3d()
        else:
            self._init_fallback()

    def _init_pyvista(self):
        """Initialize PyVista 3D viewer."""
        self.viewer_type = "pyvista"

        # Create frame for controls
        self.control_frame = ttk.Frame(self.parent)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Opacity slider
        ttk.Label(self.control_frame, text="Opacity:").pack(side=tk.LEFT)
        self.opacity_var = tk.DoubleVar(value=0.3)
        self.opacity_slider = ttk.Scale(
            self.control_frame, from_=0.0, to=1.0,
            variable=self.opacity_var, orient=tk.HORIZONTAL,
            command=self._update_opacity
        )
        self.opacity_slider.pack(side=tk.LEFT, padx=5)

        # View buttons
        ttk.Button(self.control_frame, text="Reset View",
                  command=self._reset_view).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.control_frame, text="Top View",
                  command=lambda: self._set_view("top")).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.control_frame, text="Front View",
                  command=lambda: self._set_view("front")).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.control_frame, text="Side View",
                  command=lambda: self._set_view("side")).pack(side=tk.LEFT, padx=2)

        # Toggle checkboxes
        self.show_seismic_var = tk.BooleanVar(value=True)
        self.show_faults_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Seismic",
                       variable=self.show_seismic_var,
                       command=self._update_visibility).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(self.control_frame, text="Faults",
                       variable=self.show_faults_var,
                       command=self._update_visibility).pack(side=tk.LEFT, padx=5)

        # PyVista plotter (will be embedded in Tkinter)
        self.plotter_frame = ttk.Frame(self.parent)
        self.plotter_frame.pack(fill=tk.BOTH, expand=True)

        # Note: BackgroundPlotter requires Qt, for pure Tkinter we'll use off-screen rendering
        self.plotter = None
        self._seismic_actor = None
        self._fault_actor = None

    def _init_matplotlib_3d(self):
        """Initialize matplotlib 3D fallback."""
        self.viewer_type = "matplotlib"

        # Control frame
        self.control_frame = ttk.Frame(self.parent)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Slice selection for 3D wireframe
        ttk.Label(self.control_frame, text="IL Skip:").pack(side=tk.LEFT)
        self.il_skip_var = tk.IntVar(value=10)
        ttk.Spinbox(self.control_frame, from_=1, to=50, width=5,
                   textvariable=self.il_skip_var).pack(side=tk.LEFT, padx=5)

        ttk.Label(self.control_frame, text="XL Skip:").pack(side=tk.LEFT)
        self.xl_skip_var = tk.IntVar(value=10)
        ttk.Spinbox(self.control_frame, from_=1, to=50, width=5,
                   textvariable=self.xl_skip_var).pack(side=tk.LEFT, padx=5)

        ttk.Button(self.control_frame, text="Render",
                  command=self._render_mpl_3d).pack(side=tk.LEFT, padx=10)

        # Matplotlib figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas_frame = ttk.Frame(self.parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _init_fallback(self):
        """Fallback when no 3D library available."""
        self.viewer_type = "none"
        label = ttk.Label(
            self.parent,
            text="3D Visualization requires PyVista or matplotlib.\n"
                 "Install with: pip install pyvista pyvistaqt",
            justify=tk.CENTER
        )
        label.pack(expand=True)

    def set_seismic_data(self, data: np.ndarray, geometry: Dict):
        """Set seismic volume data."""
        self.seismic_data = data
        self.geometry = geometry

    def set_fault_data(self, fault_probability: np.ndarray, threshold: float = 0.5):
        """Set fault probability volume."""
        self.fault_data = fault_probability
        self.fault_threshold = threshold

    def render(self):
        """Render the 3D scene."""
        if self.viewer_type == "pyvista":
            self._render_pyvista()
        elif self.viewer_type == "matplotlib":
            self._render_mpl_3d()

    def _render_pyvista(self):
        """Render using PyVista."""
        if self.seismic_data is None:
            return

        # Create plotter if needed
        if self.plotter is None:
            self.plotter = pv.Plotter(off_screen=True)
        else:
            self.plotter.clear()

        # Add seismic volume
        if self.show_seismic_var.get() and self.seismic_data is not None:
            grid = pv.ImageData()
            grid.dimensions = np.array(self.seismic_data.shape) + 1
            grid.spacing = (1, 1, 1)
            grid.cell_data["amplitude"] = self.seismic_data.flatten(order="F")

            self._seismic_actor = self.plotter.add_volume(
                grid, scalars="amplitude",
                cmap=self.colormap,
                opacity="sigmoid",
                shade=True
            )

        # Add fault surfaces
        if self.show_faults_var.get() and self.fault_data is not None:
            fault_surface = self._extract_fault_surface(self.fault_data)
            if fault_surface is not None:
                self._fault_actor = self.plotter.add_mesh(
                    fault_surface,
                    color="yellow",
                    opacity=0.8,
                    show_edges=True
                )

        # Render to image and display in Tkinter
        self.plotter.show(auto_close=False)
        img = self.plotter.screenshot(return_img=True)
        self._display_image_in_tkinter(img)

    def _render_mpl_3d(self):
        """Render using matplotlib 3D (simplified wireframe)."""
        if self.seismic_data is None:
            return

        self.ax.clear()

        n_il, n_xl, n_samples = self.seismic_data.shape
        il_skip = self.il_skip_var.get()
        xl_skip = self.xl_skip_var.get()

        # Plot selected inlines as surfaces
        for il in range(0, n_il, il_skip):
            X = np.arange(n_xl)
            Z = np.arange(n_samples)
            X, Z = np.meshgrid(X, Z)
            Y = np.full_like(X, il)
            C = self.seismic_data[il, :, :].T

            # Normalize colors
            C_norm = (C - C.min()) / (C.max() - C.min() + 1e-10)

            self.ax.plot_surface(
                X, Y, Z, facecolors=plt.cm.seismic(C_norm),
                alpha=0.3, linewidth=0, antialiased=False
            )

        # Plot selected crosslines
        for xl in range(0, n_xl, xl_skip):
            Y = np.arange(n_il)
            Z = np.arange(n_samples)
            Y, Z = np.meshgrid(Y, Z)
            X = np.full_like(Y, xl)
            C = self.seismic_data[:, xl, :].T

            C_norm = (C - C.min()) / (C.max() - C.min() + 1e-10)

            self.ax.plot_surface(
                X, Y, Z, facecolors=plt.cm.seismic(C_norm),
                alpha=0.3, linewidth=0, antialiased=False
            )

        # Plot faults if available
        if self.fault_data is not None:
            fault_mask = self.fault_data > self.fault_threshold
            fault_coords = np.where(fault_mask)
            if len(fault_coords[0]) > 0:
                # Subsample for performance
                step = max(1, len(fault_coords[0]) // 5000)
                self.ax.scatter(
                    fault_coords[1][::step],  # XL
                    fault_coords[0][::step],  # IL
                    fault_coords[2][::step],  # Sample
                    c='yellow', s=1, alpha=0.5, label='Faults'
                )

        self.ax.set_xlabel('Crossline')
        self.ax.set_ylabel('Inline')
        self.ax.set_zlabel('Time (samples)')
        self.ax.invert_zaxis()

        self.canvas.draw()

    def _extract_fault_surface(self, fault_prob: np.ndarray):
        """Extract isosurface from fault probability volume."""
        if not PYVISTA_AVAILABLE:
            return None

        grid = pv.ImageData()
        grid.dimensions = np.array(fault_prob.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.cell_data["probability"] = fault_prob.flatten(order="F")

        # Extract isosurface at threshold
        surface = grid.contour(
            isosurfaces=[self.fault_threshold],
            scalars="probability"
        )
        return surface

    def _display_image_in_tkinter(self, img: np.ndarray):
        """Display rendered image in Tkinter."""
        from PIL import Image, ImageTk

        pil_img = Image.fromarray(img)
        tk_img = ImageTk.PhotoImage(pil_img)

        # Create or update label
        if not hasattr(self, 'img_label'):
            self.img_label = ttk.Label(self.plotter_frame, image=tk_img)
            self.img_label.pack(fill=tk.BOTH, expand=True)
        else:
            self.img_label.configure(image=tk_img)

        self.img_label.image = tk_img  # Keep reference

    def _update_opacity(self, *args):
        self.opacity = self.opacity_var.get()
        self.render()

    def _update_visibility(self):
        self.render()

    def _reset_view(self):
        if self.viewer_type == "pyvista" and self.plotter:
            self.plotter.reset_camera()
            self.render()
        elif self.viewer_type == "matplotlib":
            self.ax.view_init(elev=30, azim=-60)
            self.canvas.draw()

    def _set_view(self, view: str):
        if self.viewer_type == "matplotlib":
            views = {
                "top": (90, 0),
                "front": (0, 0),
                "side": (0, 90)
            }
            elev, azim = views.get(view, (30, -60))
            self.ax.view_init(elev=elev, azim=azim)
            self.canvas.draw()


class Overlay2D3DViewer:
    """
    Overlay viewer for combined 2D and 3D visualization.
    Shows 2D lines overlaid on 3D volume context.
    """

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.seismic_3d: Optional[np.ndarray] = None
        self.seismic_2d_lines: Dict[str, np.ndarray] = {}
        self.geometry_3d: Dict = {}
        self.geometry_2d: Dict[str, Dict] = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup the overlay viewer UI."""
        # Control panel
        control_frame = ttk.Frame(self.parent)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 2D line selection
        ttk.Label(control_frame, text="2D Line:").pack(side=tk.LEFT)
        self.line_var = tk.StringVar()
        self.line_combo = ttk.Combobox(
            control_frame, textvariable=self.line_var, state='readonly', width=20
        )
        self.line_combo.pack(side=tk.LEFT, padx=5)
        self.line_combo.bind('<<ComboboxSelected>>', self._on_line_selected)

        # Overlay opacity
        ttk.Label(control_frame, text="2D Opacity:").pack(side=tk.LEFT, padx=(20, 0))
        self.overlay_opacity = tk.DoubleVar(value=1.0)
        ttk.Scale(control_frame, from_=0.0, to=1.0,
                 variable=self.overlay_opacity,
                 command=self._update_overlay).pack(side=tk.LEFT, padx=5)

        # Display mode
        self.mode_var = tk.StringVar(value="overlay")
        ttk.Radiobutton(control_frame, text="Overlay",
                       variable=self.mode_var, value="overlay",
                       command=self._update_display).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(control_frame, text="Side-by-Side",
                       variable=self.mode_var, value="sidebyside",
                       command=self._update_display).pack(side=tk.LEFT, padx=5)

        # Canvas
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        self.canvas_frame = ttk.Frame(self.parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def set_3d_data(self, data: np.ndarray, geometry: Dict):
        """Set 3D seismic volume."""
        self.seismic_3d = data
        self.geometry_3d = geometry

    def add_2d_line(self, name: str, data: np.ndarray, geometry: Dict):
        """Add a 2D seismic line."""
        self.seismic_2d_lines[name] = data
        self.geometry_2d[name] = geometry

        # Update combo box
        self.line_combo['values'] = list(self.seismic_2d_lines.keys())
        if not self.line_var.get() and self.seismic_2d_lines:
            self.line_var.set(list(self.seismic_2d_lines.keys())[0])

    def _on_line_selected(self, event):
        self._update_display()

    def _update_overlay(self, *args):
        self._update_display()

    def _update_display(self):
        """Update the overlay display."""
        line_name = self.line_var.get()
        if not line_name or line_name not in self.seismic_2d_lines:
            return

        line_2d = self.seismic_2d_lines[line_name]

        for ax in self.axes:
            ax.clear()

        mode = self.mode_var.get()

        if mode == "overlay" and self.seismic_3d is not None:
            # Extract nearest inline from 3D for overlay
            # Assume 2D line corresponds to a specific inline
            inline_idx = self.geometry_2d[line_name].get("inline", 0)
            if 0 <= inline_idx < self.seismic_3d.shape[0]:
                slice_3d = self.seismic_3d[inline_idx, :, :].T

                # Show 3D section with reduced opacity
                self.axes[0].imshow(
                    slice_3d, cmap='seismic', aspect='auto', alpha=0.5
                )
                # Overlay 2D with adjustable opacity
                self.axes[0].imshow(
                    line_2d.T, cmap='seismic', aspect='auto',
                    alpha=self.overlay_opacity.get()
                )
                self.axes[0].set_title(f"Overlay: 3D IL {inline_idx} + 2D {line_name}")

            # Show difference
            if slice_3d.shape == line_2d.T.shape:
                diff = line_2d.T - slice_3d
                self.axes[1].imshow(diff, cmap='RdBu', aspect='auto')
                self.axes[1].set_title("Difference (2D - 3D)")

        elif mode == "sidebyside":
            # 2D line
            self.axes[0].imshow(line_2d.T, cmap='seismic', aspect='auto')
            self.axes[0].set_title(f"2D Line: {line_name}")

            # Corresponding 3D section
            if self.seismic_3d is not None:
                inline_idx = self.geometry_2d[line_name].get("inline", 0)
                if 0 <= inline_idx < self.seismic_3d.shape[0]:
                    slice_3d = self.seismic_3d[inline_idx, :, :].T
                    self.axes[1].imshow(slice_3d, cmap='seismic', aspect='auto')
                    self.axes[1].set_title(f"3D Inline: {inline_idx}")

        self.fig.tight_layout()
        self.canvas.draw()
```

#### 3.2 Integrate 3D Viewer into Seismic Viewer

**File:** `seismic_viewer.py`
**Location:** Add new tab for 3D view

**Add to `_create_display_panel()` method:**
```python
def _create_display_panel(self, parent):
    # Create notebook for multiple views
    self.view_notebook = ttk.Notebook(parent)
    self.view_notebook.pack(fill=tk.BOTH, expand=True)

    # Tab 1: 2D Section Viewer (existing)
    self.section_frame = ttk.Frame(self.view_notebook)
    self.view_notebook.add(self.section_frame, text="2D Sections")
    self._create_2d_viewer(self.section_frame)

    # Tab 2: 3D Volume Viewer (new)
    self.volume_frame = ttk.Frame(self.view_notebook)
    self.view_notebook.add(self.volume_frame, text="3D Volume")
    self._create_3d_viewer(self.volume_frame)

    # Tab 3: 2D/3D Overlay (new)
    self.overlay_frame = ttk.Frame(self.view_notebook)
    self.view_notebook.add(self.overlay_frame, text="2D/3D Overlay")
    self._create_overlay_viewer(self.overlay_frame)

def _create_3d_viewer(self, parent):
    """Create 3D volume visualization tab."""
    from viewer_3d import Seismic3DViewer

    self.viewer_3d = Seismic3DViewer(parent)

    # Connect to data
    if self.loader.segy_3d is not None:
        self.viewer_3d.set_seismic_data(
            self.loader.segy_3d,
            self.loader.geometry
        )

    # Connect to faults
    fault_prob_path = os.path.join(
        "interpretation", "real_outputs", "fault_probability.npy"
    )
    if os.path.exists(fault_prob_path):
        fault_prob = np.load(fault_prob_path)
        self.viewer_3d.set_fault_data(fault_prob)

def _create_overlay_viewer(self, parent):
    """Create 2D/3D overlay comparison tab."""
    from viewer_3d import Overlay2D3DViewer

    self.overlay_viewer = Overlay2D3DViewer(parent)

    if self.loader.segy_3d is not None:
        self.overlay_viewer.set_3d_data(
            self.loader.segy_3d,
            self.loader.geometry
        )

    for name, data in self.loader.segy_2d_files.items():
        self.overlay_viewer.add_2d_line(name, data, {})
```

---

## PHASE 4: Integration Testing & Validation

### 4.1 Test Cases

1. **Data Loading Test**
   - Start main GUI, configure 3D and 2D paths
   - Launch seismic viewer
   - Verify data auto-loads from config
   - Verify manual load buttons still work

2. **Unified Fault Detection Test**
   - Run deep learning fault detection
   - Run classical fault detection
   - Compare outputs in unified format
   - Verify viewer displays both correctly

3. **3D Visualization Test**
   - Load 3D volume
   - Open 3D Volume tab
   - Verify rendering works (PyVista or matplotlib)
   - Toggle fault overlay
   - Test view controls

4. **Overlay Test**
   - Load both 2D and 3D data
   - Open 2D/3D Overlay tab
   - Select 2D line
   - Verify overlay and side-by-side modes

### 4.2 Performance Optimization

1. **Lazy Loading**: Only load data when tab is selected
2. **Downsampling**: For 3D rendering, use every Nth sample
3. **Caching**: Cache rendered images for quick view switching
4. **Threading**: Move heavy rendering to background thread

---

## Implementation Order

### Week 1: Phase 1 (Data Loading)
1. Add argument parsing to `seismic_viewer.py`
2. Modify viewer launch in `phd_workflow_gui.py`
3. Implement auto-load functionality
4. Test integration

### Week 2: Phase 2 (Fault Detection)
1. Create `fault_detection_unified.py`
2. Update `seismic_viewer.py` to use unified format
3. Modify deep learning output to unified format
4. Test both methods produce compatible output

### Week 3: Phase 3 (3D Visualization)
1. Create `viewer_3d.py`
2. Add 3D tab to seismic viewer
3. Add overlay tab
4. Test with sample data

### Week 4: Integration & Polish
1. Full integration testing
2. Performance optimization
3. Error handling
4. Documentation

---

## Dependencies to Add

```
# In requirements.txt
pyvista>=0.40.0        # 3D visualization (optional)
pyvistaqt>=0.9.0       # Qt integration for PyVista (optional)
vtk>=9.2.0             # VTK backend (optional)
```

**Note**: PyVista/VTK are optional. The system falls back to matplotlib 3D if not installed.

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `fault_detection_unified.py` | CREATE | Unified fault detection interface |
| `viewer_3d.py` | CREATE | 3D visualization module |
| `seismic_viewer.py` | MODIFY | Add arg parsing, auto-load, 3D tabs |
| `phd_workflow_gui.py` | MODIFY | Pass config to viewer on launch |
| `deep_learning/dl_fault_detection.py` | MODIFY | Output unified format |
| `interpretation/real_interpretation.py` | MODIFY | Output unified format |

---

## Summary

This optimization plan addresses all three issues:

1. **Redundant Data Loading**: Fixed by passing ProjectConfig paths to viewer via command-line args and auto-loading on startup.

2. **Dual Fault Detection**: Unified by creating a common data model (`UnifiedFaultResult`) that both DL and classical methods output to, with ensemble mode for combining results.

3. **2D-Only Display**: Fixed by adding PyVista/matplotlib 3D visualization tabs with volumetric rendering and 2D/3D overlay comparison.

The implementation is modular, backward-compatible, and includes fallbacks for missing dependencies.
